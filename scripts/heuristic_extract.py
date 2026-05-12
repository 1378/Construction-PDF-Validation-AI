"""
Эвристическое извлечение параметров из текста (PDF/OCR) без LLM.

Используется, когда Ollama недоступна: по rules.json ищутся допустимые
значения резьбы, материалов, обозначений в сыром тексте, чтобы
rule_validator мог выполнить детерминированную проверку.
"""

from __future__ import annotations

import re
from typing import Any

from scripts.models import ConfidenceLevel, DrawingElement, SourceType


def extract_elements_from_text(text: str, rules: dict[str, Any]) -> list[DrawingElement]:
    """
    Находит в тексте упоминания значений из ТЗ (резьбы, материалы, обозначения).

    Не заменяет LLM: даёт низкую уверенность и возможны пропуски.
    """
    if not text or not rules:
        return []

    elements: list[DrawingElement] = []
    seen_sizes: set[str] = set()

    def add_el(el: DrawingElement) -> None:
        elements.append(el)

    bolt_rule: dict[str, Any] | None = None
    for dim in rules.get("dimensions", []):
        if dim.get("field") == "bolt_size":
            bolt_rule = dim
            break

    allowed_threads: list[str] = []
    if bolt_rule:
        allowed_threads = [str(v) for v in bolt_rule.get("allowed_values", [])]

    # 1) Явные совпадения с допустимыми размерами резьбы из ТЗ
    for v in allowed_threads:
        pattern = re.escape(v).replace(r"x", r"[x×X]")
        if re.search(pattern, text, re.IGNORECASE):
            key = v.lower()
            if key in seen_sizes:
                continue
            seen_sizes.add(key)
            add_el(
                DrawingElement(
                    item_id=f"эвристика:{v}",
                    name="Резьба (найдена в тексте)",
                    size=v,
                    note="Извлечено эвристикой по совпадению с допустимыми значениями ТЗ",
                    source=SourceType.COMBINED,
                    confidence=ConfidenceLevel.LOW,
                    raw_text_fragment=_snippet(text, v, 80),
                )
            )

    # 2) Любая метрическая резьба M… в тексте (в т.ч. вне списка — для срабатывания валидатора)
    allowed_lower = {a.lower() for a in allowed_threads}
    for m in re.finditer(r"\bM\d+(?:[x×]\d+(?:[.,]\d+)?)?\b", text, re.IGNORECASE):
        raw = m.group().replace("×", "x")
        key = raw.lower()
        if key in seen_sizes:
            continue
        seen_sizes.add(key)
        in_list = key in allowed_lower or raw in allowed_threads
        add_el(
            DrawingElement(
                item_id=f"эвристика:{raw}",
                name="Резьба (найдена в тексте)",
                size=raw,
                note=(
                    "Извлечено эвристикой по шаблону M…"
                    if in_list
                    else "Извлечено эвристикой; может не входить в допустимые по ТЗ"
                ),
                source=SourceType.COMBINED,
                confidence=ConfidenceLevel.LOW,
                raw_text_fragment=_snippet(text, raw, 80),
            )
        )

    # 3) Материалы (allowed + forbidden — чтобы зафиксировать нарушение)
    mats = rules.get("materials", {})
    seen_mat: set[str] = set()
    for mat in mats.get("allowed", []) + mats.get("forbidden", []):
        if not mat or len(mat) < 2:
            continue
        if mat not in text:
            continue
        if mat in seen_mat:
            continue
        seen_mat.add(mat)
        add_el(
            DrawingElement(
                item_id=f"эвристика:мат-{mat}",
                name="Материал (найден в тексте)",
                material=mat,
                note="Извлечено эвристикой по вхождению марки в текст",
                source=SourceType.COMBINED,
                confidence=ConfidenceLevel.LOW,
                raw_text_fragment=_snippet(text, mat, 80),
            )
        )

    # 4) Обозначения по regex из ТЗ
    des = rules.get("designations", {})
    pat = des.get("pattern")
    if pat:
        try:
            seen_des: set[str] = set()
            for m in re.finditer(pat, text):
                d = m.group(0)
                if d in seen_des:
                    continue
                seen_des.add(d)
                add_el(
                    DrawingElement(
                        item_id=f"эвристика:{d}",
                        name="Обозначение (найдено в тексте)",
                        designation=d,
                        note="Извлечено эвристикой по regex из rules.json",
                        source=SourceType.COMBINED,
                        confidence=ConfidenceLevel.LOW,
                        raw_text_fragment=_snippet(text, d, 80),
                    )
                )
        except re.error:
            pass

    return elements


def _snippet(text: str, needle: str, radius: int) -> str:
    i = text.find(needle)
    if i < 0:
        return needle[: radius * 2]
    a = max(0, i - radius)
    b = min(len(text), i + len(needle) + radius)
    return text[a:b].replace("\n", " ").strip()
