"""
multimodal_caption.py — получение текстового описания чертежа через LLaVA (Ollama).

Отправляет изображение страницы в локальную multimodal-модель (LLaVA или аналог)
и получает структурированное описание видимых элементов чертежа.

Промпт намеренно строгий: модель описывает только то, что видит.
Если что-то неразборчиво — явно пишет "неразборчиво".

Пример запуска:
    python -m scripts.multimodal_caption --image data/images/123-А_page_001.png
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
from pathlib import Path

import requests

from scripts.ollama_util import get_ollama_generate_url, is_ollama_reachable

logger = logging.getLogger(__name__)
DEFAULT_MODEL = "llava"
TIMEOUT_SECONDS = 120

# Промпт: явная инструкция не домысливать
CAPTION_PROMPT = """\
Ты — технический эксперт по чертежам. Тебе показано изображение технического чертежа.

Задача: опиши только то, что ВИДНО на чертеже. Не домысливай и не придумывай.

Перечисли:
1. Видимые элементы (детали, узлы, позиции) с их обозначениями и размерами если они указаны
2. Текстовые подписи, надписи, маркировки
3. Размерные линии и значения размеров (если видны и разборчивы)
4. Материалы (если указаны на чертеже или в штампе)
5. Примечания и технические требования (если есть)

Правила:
- Если текст неразборчив — пиши "неразборчиво"
- Если элемент виден, но его назначение непонятно — опиши форму/расположение
- НЕ ПРИДУМЫВАЙ размеры, марки, коды которых нет на изображении
- Отвечай на русском языке
- Будь конкретен и лаконичен

Опиши чертёж:\
"""


def _encode_image_base64(image_path: str) -> str:
    """Кодирует изображение в base64 для передачи в Ollama API."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def caption_drawing(
    image_path: str,
    model: str = DEFAULT_MODEL,
    ollama_url: str | None = None,
    timeout: int = TIMEOUT_SECONDS,
) -> str:
    """
    Отправляет изображение чертежа в LLaVA и получает текстовое описание.

    Args:
        image_path: Путь к PNG/JPG изображению.
        model: Имя модели в Ollama (напр. "llava", "llava:13b").
        ollama_url: URL Ollama API.
        timeout: Таймаут запроса в секундах.

    Returns:
        Текстовое описание чертежа или сообщение об ошибке с пометкой.
    """
    if ollama_url is None:
        ollama_url = get_ollama_generate_url()

    if not Path(image_path).exists():
        logger.error("Изображение не найдено: %s", image_path)
        return "[ОШИБКА: изображение не найдено — нельзя сгенерировать описание]"

    try:
        image_b64 = _encode_image_base64(image_path)
    except Exception as e:
        logger.error("Не удалось прочитать изображение %s: %s", image_path, e)
        return f"[ОШИБКА: не удалось прочитать изображение — {e}]"

    payload = {
        "model": model,
        "prompt": CAPTION_PROMPT,
        "images": [image_b64],
        "stream": False,
        "options": {
            "temperature": 0.1,   # низкая температура = меньше выдумок
            "top_p": 0.9,
        },
    }

    try:
        response = requests.post(ollama_url, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        caption = data.get("response", "").strip()

        if not caption:
            logger.warning("LLaVA вернул пустой ответ для %s", image_path)
            return "[LLaVA вернул пустой ответ — нельзя подтвердить содержимое чертежа]"

        logger.info("LLaVA описание получено: %d символов", len(caption))
        return caption

    except requests.exceptions.ConnectionError:
        logger.error("Ollama недоступна по адресу %s", ollama_url)
        return "[ОШИБКА: Ollama недоступна — описание чертежа невозможно]"
    except requests.exceptions.Timeout:
        logger.error("Таймаут при запросе к Ollama для %s", image_path)
        return "[ОШИБКА: таймаут Ollama — описание чертежа невозможно]"
    except (requests.exceptions.HTTPError, json.JSONDecodeError, KeyError) as e:
        logger.error("Ошибка Ollama API для %s: %s", image_path, e)
        return f"[ОШИБКА Ollama API: {e} — описание чертежа невозможно]"


def caption_drawing_batch(
    image_paths: list[str],
    model: str = DEFAULT_MODEL,
    ollama_url: str | None = None,
) -> list[str]:
    """Генерирует описания для списка изображений."""
    captions = []
    for path in image_paths:
        logger.info("Генерация описания: %s", path)
        caption = caption_drawing(path, model=model, ollama_url=ollama_url)
        captions.append(caption)
    return captions


def is_ollama_available(ollama_url: str | None = None) -> bool:
    """Проверяет доступность Ollama (GET /api/tags)."""
    if ollama_url is None:
        ollama_url = get_ollama_generate_url()
    return is_ollama_reachable(ollama_url)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Мультимодальное описание чертежа через LLaVA")
    parser.add_argument("--image", required=True, help="Путь к PNG изображению чертежа")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Модель Ollama (по умолчанию: {DEFAULT_MODEL})")
    parser.add_argument("--url", default=None, help="URL /api/generate (по умолчанию из OLLAMA_BASE_URL)")
    args = parser.parse_args()

    gen_url = args.url or get_ollama_generate_url()
    if not is_ollama_available(gen_url):
        print(f"ПРЕДУПРЕЖДЕНИЕ: Ollama недоступна по адресу {gen_url}")
        print("Убедитесь, что Ollama запущена: ollama serve")
        print(f"И модель загружена: ollama pull {args.model}")
        return

    caption = caption_drawing(args.image, model=args.model, ollama_url=gen_url)
    print(f"\n--- Описание чертежа ({args.model}) ---\n{caption}\n---")


if __name__ == "__main__":
    main()
