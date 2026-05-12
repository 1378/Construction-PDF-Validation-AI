"""
llm_checker.py — LLM-агент для сравнения элементов чертежа с ТЗ.

Алгоритм:
1. Берёт текст чертежа (OCR + LLaVA caption + PDF-текст).
2. Ищет релевантные чанки ТЗ через index_rag (RAG).
3. Формирует строгий антигаллюцинационный промпт.
4. Вызывает локальную LLM через Ollama.
5. Парсит JSON-ответ в DrawingReport.

LLM не получает информацию о требованиях кроме той, что найдена RAG-поиском.
Это ключевая мера против галлюцинаций.

Пример запуска:
    python -m scripts.llm_checker --drawing-id 123-А --text "Штуцер M16x1.5 Ст3сп"
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Optional

import requests

from scripts.ollama_util import get_ollama_generate_url
from scripts.models import (
    ComplianceReport,
    ConfidenceLevel,
    DrawingElement,
    DrawingReport,
    SourceType,
    ValidationIssue,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "mistral"   # можно заменить на llama3.2, qwen2.5 и т.п.
TIMEOUT_SECONDS = 180

# ---------------------------------------------------------------------------
# Промпт — самая важная часть антигаллюцинационной защиты
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
Ты — инженер-контролёр качества. Твоя задача: сравнить элементы чертежа с требованиями ТЗ.

СТРОГИЕ ПРАВИЛА:
1. Анализируй ТОЛЬКО данные из "ОПИСАНИЕ ЧЕРТЕЖА" и "ВЫДЕРЖКИ ИЗ ТЗ" ниже.
2. НЕ ПРИДУМЫВАЙ нормы, размеры, материалы, коды которых нет в предоставленных данных.
3. Если информации недостаточно — пиши: "нельзя подтвердить", "не указано в ТЗ", "не видно на чертеже".
4. При каждом утверждении о соответствии/несоответствии — цитируй фрагмент из ТЗ.
5. Если ТЗ не содержит нормы для данного параметра — пиши "не указано в ТЗ".
6. Поле is_compliant = null если нельзя однозначно определить.
7. В JSON поля compliance.missing_info и compliance.citations — это массивы СТРОК (string[]):
   каждый элемент — одна строка текста, не объект вида {"field":...} или {"text":...}.

Верни ответ СТРОГО в JSON формате без пояснений вне JSON.\
"""

USER_PROMPT_TEMPLATE = """\
=== ОПИСАНИЕ ЧЕРТЕЖА (drawing_id: {drawing_id}) ===
{drawing_text}

=== ВЫДЕРЖКИ ИЗ ТЗ (найдено RAG-поиском, топ-{top_k}) ===
{tz_chunks_text}

=== ЗАДАЧА ===
Извлеки все элементы чертежа и проверь их соответствие ТЗ.

Формат compliance: missing_info — массив СТРОК с отсутствующими данными / причинами;
citations — массив СТРОК с цитатами из ТЗ (не вложенные объекты).

Верни JSON строго по этой схеме:
{{
  "drawing_id": "{drawing_id}",
  "elements": [
    {{
      "item_id": "строка или null",
      "name": "строка или null",
      "element_type": "строка или null",
      "size": "строка или null",
      "material": "строка или null",
      "designation": "строка или null",
      "note": "строка или null",
      "source": "pdf_text|ocr|llava_caption|combined",
      "confidence": "high|medium|low",
      "raw_text_fragment": "цитата из описания чертежа или null"
    }}
  ],
  "compliance": {{
    "standard": "название стандарта или null",
    "is_compliant": true/false/null,
    "issues": [
      {{
        "field": "имя поля",
        "actual_value": "значение из чертежа или null",
        "expected": "ожидаемое значение или диапазон из ТЗ или null",
        "message": "конкретное описание расхождения",
        "tz_reference": "цитата из ТЗ подтверждающая требование"
      }}
    ],
    "missing_info": ["строка: что нельзя проверить и почему"],
    "citations": ["строка: прямая цитата из ТЗ, использованная для проверки"]
  }}
}}\
"""


def _format_tz_chunks(chunks: list[dict]) -> str:
    """Форматирует чанки ТЗ для включения в промпт."""
    if not chunks:
        return "[Релевантные разделы ТЗ не найдены. Проверка по ТЗ невозможна.]"

    lines = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "неизвестно")
        score = chunk.get("score", 0)
        text = chunk.get("text", "")
        lines.append(f"[{i}] Источник: {source} (релевантность: {score:.2f})\n{text}")

    return "\n\n---\n\n".join(lines)


def _call_ollama(prompt: str, model: str, ollama_url: str, timeout: int) -> Optional[str]:
    """Вызывает Ollama и возвращает текст ответа или None при ошибке."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.05,   # максимально детерминированно
            "top_p": 0.9,
            "num_predict": 4096,
        },
    }

    try:
        response = requests.post(ollama_url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.exceptions.ConnectionError:
        logger.error("Ollama недоступна по адресу %s", ollama_url)
        return None
    except requests.exceptions.Timeout:
        logger.error("Таймаут Ollama при проверке чертежа")
        return None
    except Exception as e:
        logger.error("Ошибка Ollama API: %s", e)
        return None


def _parse_llm_response(raw: str, drawing_id: str) -> DrawingReport:
    """
    Парсит JSON-ответ LLM в DrawingReport.

    При ошибках парсинга возвращает частичный результат с пометкой low confidence.
    """
    # Извлекаем JSON даже если LLM обернул его в markdown
    json_match = re.search(r"\{[\s\S]+\}", raw)
    if not json_match:
        logger.error("LLM не вернул валидный JSON")
        return DrawingReport(
            drawing_id=drawing_id,
            pdf_path="",
            compliance=ComplianceReport(
                is_compliant=None,
                missing_info=["LLM не вернул валидный JSON — результат ненадёжен"],
            ),
            overall_confidence=ConfidenceLevel.LOW,
        )

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        logger.error("Ошибка парсинга JSON от LLM: %s", e)
        return DrawingReport(
            drawing_id=drawing_id,
            pdf_path="",
            compliance=ComplianceReport(
                is_compliant=None,
                missing_info=[f"Ошибка парсинга JSON от LLM: {e}"],
            ),
            overall_confidence=ConfidenceLevel.LOW,
        )

    # Строим элементы
    elements: list[DrawingElement] = []
    for el_data in data.get("elements", []):
        try:
            source_str = el_data.get("source", "pdf_text")
            try:
                source = SourceType(source_str)
            except ValueError:
                source = SourceType.PDF_TEXT

            conf_str = el_data.get("confidence", "medium")
            try:
                conf = ConfidenceLevel(conf_str)
            except ValueError:
                conf = ConfidenceLevel.MEDIUM

            elements.append(DrawingElement(
                item_id=el_data.get("item_id"),
                name=el_data.get("name"),
                element_type=el_data.get("element_type"),
                size=el_data.get("size"),
                material=el_data.get("material"),
                designation=el_data.get("designation"),
                note=el_data.get("note"),
                source=source,
                confidence=conf,
                raw_text_fragment=el_data.get("raw_text_fragment"),
            ))
        except Exception as e:
            logger.warning("Не удалось разобрать элемент от LLM: %s | %s", el_data, e)

    # Строим compliance
    comp_data = data.get("compliance", {})
    issues: list[ValidationIssue] = []
    for issue_data in comp_data.get("issues", []):
        try:
            issues.append(ValidationIssue(
                field=issue_data.get("field", "unknown"),
                actual_value=issue_data.get("actual_value"),
                expected=issue_data.get("expected"),
                message=issue_data.get("message", ""),
                tz_reference=issue_data.get("tz_reference"),
            ))
        except Exception as e:
            logger.warning("Не удалось разобрать issue от LLM: %s | %s", issue_data, e)

    is_compliant_raw = comp_data.get("is_compliant")
    is_compliant: Optional[bool] = None
    if isinstance(is_compliant_raw, bool):
        is_compliant = is_compliant_raw
    elif isinstance(is_compliant_raw, str):
        if is_compliant_raw.lower() == "true":
            is_compliant = True
        elif is_compliant_raw.lower() == "false":
            is_compliant = False

    logger.info(
        "LLM compliance (сырые missing_info / citations до Pydantic): %s / %s",
        comp_data.get("missing_info"),
        comp_data.get("citations"),
    )
    logger.debug(
        "LLM raw compliance до валидации Pydantic: %s",
        json.dumps(comp_data, ensure_ascii=False, default=str),
    )

    compliance = ComplianceReport(
        standard=comp_data.get("standard"),
        is_compliant=is_compliant,
        issues=issues,
        missing_info=comp_data.get("missing_info", []),
        citations=comp_data.get("citations", []),
    )

    # Общая уверенность
    low_conf_elements = sum(1 for e in elements if e.confidence == ConfidenceLevel.LOW)
    if not elements or low_conf_elements > len(elements) // 2:
        overall_conf = ConfidenceLevel.LOW
    elif low_conf_elements > 0 or is_compliant is None:
        overall_conf = ConfidenceLevel.MEDIUM
    else:
        overall_conf = ConfidenceLevel.HIGH

    return DrawingReport(
        drawing_id=drawing_id,
        pdf_path="",
        elements=elements,
        compliance=compliance,
        overall_confidence=overall_conf,
    )


def check_drawing(
    drawing_id: str,
    drawing_text: str,
    tz_chunks: list[dict],
    model: str = DEFAULT_MODEL,
    ollama_url: str | None = None,
    timeout: int = TIMEOUT_SECONDS,
) -> DrawingReport:
    """
    Сравнивает описание чертежа с ТЗ через LLM.

    Args:
        drawing_id: Идентификатор чертежа.
        drawing_text: Объединённый текст (PDF + OCR + LLaVA caption).
        tz_chunks: Чанки ТЗ, найденные RAG-поиском.
        model: Модель Ollama.
        ollama_url: URL Ollama API.
        timeout: Таймаут в секундах.

    Returns:
        DrawingReport с элементами и отчётом о соответствии.
    """
    if ollama_url is None:
        ollama_url = get_ollama_generate_url()

    tz_text = _format_tz_chunks(tz_chunks)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        drawing_id=drawing_id,
        drawing_text=drawing_text[:3000],    # ограничиваем контекст
        tz_chunks_text=tz_text[:2000],
        top_k=len(tz_chunks),
    )
    full_prompt = SYSTEM_PROMPT + "\n\n" + user_prompt

    logger.info("Отправка в LLM: модель=%s, текст=%d симв.", model, len(full_prompt))
    raw_response = _call_ollama(full_prompt, model, ollama_url, timeout)

    if raw_response is None:
        return DrawingReport(
            drawing_id=drawing_id,
            pdf_path="",
            llm_used=False,
            tz_chunks_used=[c.get("text", "") for c in tz_chunks],
            compliance=ComplianceReport(
                is_compliant=None,
                missing_info=[
                    "Текстовая LLM (Ollama) недоступна: запустите ollama serve и установите модель "
                    f"(например: ollama pull {model}). Дальше включится эвристика по тексту и rules.json."
                ],
            ),
            overall_confidence=ConfidenceLevel.LOW,
        )

    report = _parse_llm_response(raw_response, drawing_id)
    report.llm_used = True
    report.tz_chunks_used = [c.get("text", "") for c in tz_chunks]
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="LLM-проверка чертежа по ТЗ")
    parser.add_argument("--drawing-id", required=True)
    parser.add_argument("--text", required=True, help="Текст описания чертежа")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    from scripts.index_rag import search_tz
    tz_chunks = search_tz(args.text, top_k=args.top_k)

    report = check_drawing(
        drawing_id=args.drawing_id,
        drawing_text=args.text,
        tz_chunks=tz_chunks,
        model=args.model,
    )

    print(f"\n{report.summary()}")
    print(f"\nJSON:\n{report.model_dump_json(indent=2, exclude={'tz_chunks_used'})}")


if __name__ == "__main__":
    main()
