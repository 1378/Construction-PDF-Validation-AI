"""
Разрезание PDF на файлы по чертежам с учётом классификации страниц (PyMuPDF + utils.pdf_analyzer).

Запуск из корня проекта: python -m scripts.pdf_cutter
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import fitz

from utils.pdf_analyzer import (
    PageClassification,
    classify_page,
    extract_features,
    page_begins_new_project,
)

# Символы, недопустимые в имени файла Windows / общие ограничения
_INVALID_FS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_LABEL_FROM_DRAWING = re.compile(
    r"черт[её]ж\s*(?:№|#)?\s*([^\n]{1,80})",
    re.IGNORECASE | re.UNICODE,
)
_LABEL_FROM_PLAN = re.compile(
    r"(план\s+(?:этажа|осей|фасада|кровли|помещен)[^\n]{0,60})",
    re.IGNORECASE | re.UNICODE,
)
_LABEL_SHEET = re.compile(
    r"\b(\d{1,4}\s*[-–—]\s*[А-ЯA-ZЁё0-9]{1,8})\b",
    re.UNICODE,
)
# Для валидации: размерные цепочки и масштаб в тексте страницы
_DIM_OR_SCALE_HINT = re.compile(
    r"\d{2,4}\s*[xх×]\s*\d{2,4}|\d+\s*[=≈]\s*\d+|масштаб\s*1\s*[:]",
    re.IGNORECASE,
)
_SUPPORT_SUBSTR = ("примечани", "условн", "размер", "экспликац", "легенда")


def _collapse_text(raw: str) -> str:
    t = raw.replace("\xa0", " ")
    return re.sub(r"\s+", " ", t).strip()


def _sanitize_stem(s: str, max_len: int = 72) -> str:
    s = _INVALID_FS.sub("_", s)
    s = re.sub(r"\s+", "_", s.strip())
    s = s.strip("._") or "drawing"
    return s[:max_len].rstrip("._") or "drawing"


def _drawing_label_from_page(page: fitz.Page) -> str:
    """Краткая метка для имени файла по тексту страницы."""
    collapsed = _collapse_text(page.get_text("text") or "")
    if not collapsed:
        return "drawing"
    m = _LABEL_FROM_DRAWING.search(collapsed)
    if m:
        return _sanitize_stem(m.group(1))
    m = _LABEL_FROM_PLAN.search(collapsed)
    if m:
        return _sanitize_stem(m.group(1))
    m = _LABEL_SHEET.search(collapsed)
    if m:
        return _sanitize_stem(m.group(1).replace(" ", ""))
    return _sanitize_stem(collapsed[:64])


def _is_kept_content(c: PageClassification) -> bool:
    if c["is_irrelevant_photo_or_ad"]:
        return False
    return bool(
        c["is_drawing"]
        or c["is_explication"]
        or c["is_annotation"]
        or c.get("is_project_info", False)
    )


def _project_segments(n: int, new_project: list[bool]) -> list[tuple[int, int]]:
    """Диапазоны страниц одного проекта (дома). Разрез только по ``new_project[i]`` при i > 0."""
    if n <= 0:
        return []
    breaks = [i for i in range(1, n) if new_project[i]]
    out: list[tuple[int, int]] = []
    prev = 0
    for b in breaks:
        out.append((prev, b - 1))
        prev = b
    out.append((prev, n - 1))
    return out


def _glue_runs_in_non_junk_zone(
    zone: list[int],
    classifications: list[PageClassification],
    *,
    max_inner_gap: int,
    max_edge_unknown: int,
) -> list[list[int]]:
    """
    Внутри зоны без рекламных страниц: склеивает «пустые» страницы между
    чертежом, экспликацией, титулом и примечаниями (мало текста / низкая уверенность).
    """
    if not zone:
        return []

    def kept(i: int) -> bool:
        return _is_kept_content(classifications[i])

    raw: list[list[int]] = []
    cur: list[int] = []
    for i in zone:
        if kept(i):
            cur.append(i)
        else:
            if cur:
                raw.append(cur)
                cur = []
    if cur:
        raw.append(cur)
    if not raw:
        return []

    merged: list[list[int]] = [raw[0][:]]

    for nxt in raw[1:]:
        gap = list(range(merged[-1][-1] + 1, nxt[0]))
        if len(gap) <= max_inner_gap and gap:
            merged[-1].extend(gap)
            merged[-1].extend(nxt)
        elif not gap:
            merged[-1].extend(nxt)
        else:
            merged.append(nxt[:])

    head = list(range(zone[0], merged[0][0]))
    if 0 < len(head) <= max_edge_unknown and all(not kept(i) for i in head):
        merged[0] = head + merged[0]

    tail = list(range(merged[-1][-1] + 1, zone[-1] + 1))
    if 0 < len(tail) <= max_edge_unknown and all(not kept(i) for i in tail):
        merged[-1] = merged[-1] + tail

    return merged


def _collect_glued_runs_for_segment(
    classifications: list[PageClassification],
    lo: int,
    hi: int,
    *,
    max_inner_gap: int,
    max_edge_unknown: int,
) -> list[list[int]]:
    """Участки страниц внутри [lo, hi], реклама рвёт зону; внутри зоны — склейка дырок."""
    if lo > hi:
        return []
    zones: list[list[int]] = []
    buf: list[int] = []
    for i in range(lo, hi + 1):
        if classifications[i]["is_irrelevant_photo_or_ad"]:
            if buf:
                zones.append(buf)
                buf = []
            continue
        buf.append(i)
    if buf:
        zones.append(buf)

    runs: list[list[int]] = []
    for z in zones:
        runs.extend(
            _glue_runs_in_non_junk_zone(
                z,
                classifications,
                max_inner_gap=max_inner_gap,
                max_edge_unknown=max_edge_unknown,
            )
        )
    return runs


def _first_drawing_index_in_chunk(
    chunk: list[int],
    classifications: list[PageClassification],
) -> int:
    for idx in chunk:
        if classifications[idx]["is_drawing"]:
            return idx
    return chunk[0]


def _page_has_support_text(page: fitz.Page) -> bool:
    """Подписи/размеры/экспликация по тексту страницы (дополнение к флагам classify_page)."""
    t = extract_features(page).text
    if _DIM_OR_SCALE_HINT.search(t):
        return True
    return any(s in t for s in _SUPPORT_SUBSTR)


def validate_drawings(output_dir: str) -> dict:
    """
    Проверяет все ``drawing_*.pdf`` в каталоге: наличие чертежа, сопутствующего контента,
    отсутствие страниц с признаками рекламы/фото (по ``classify_page``).

    В конце печатает краткий отчёт в stdout.

    Returns:
        ``{"summary": {...}, "details": {имя_файла: {"status", "reason"}, ...}}``
    """
    out_dir = Path(output_dir).expanduser().resolve()
    if not out_dir.is_dir():
        raise NotADirectoryError(str(out_dir))

    pdf_paths = sorted(out_dir.glob("drawing_*.pdf"))
    details: dict[str, dict[str, str]] = {}
    suspicious_files: list[str] = []

    for path in pdf_paths:
        name = path.name
        try:
            with fitz.open(str(path)) as doc:
                n = len(doc)
                classifications: list[PageClassification] = [
                    classify_page(doc[i]) for i in range(n)
                ]

                reasons: list[str] = []

                if not any(c["is_drawing"] for c in classifications):
                    reasons.append("ни одна страница не распознана как чертёж")

                ad_pages = [
                    i + 1
                    for i, c in enumerate(classifications)
                    if c["is_irrelevant_photo_or_ad"]
                ]
                if ad_pages:
                    reasons.append(
                        "страницы с маркерами рекламы/фото: "
                        + ", ".join(str(p) for p in ad_pages)
                    )

                if n >= 2:
                    has_struct_support = any(
                        c["is_annotation"]
                        or c["is_explication"]
                        or c.get("is_project_info", False)
                        for c in classifications
                    )
                    has_text_support = any(
                        _page_has_support_text(doc[i]) for i in range(n)
                    )
                    if not has_struct_support and not has_text_support:
                        reasons.append(
                            "несколько страниц без экспликации/примечаний "
                            "и без явных размеров/масштаба в тексте"
                        )

                if reasons:
                    suspicious_files.append(name)
                    details[name] = {
                        "status": "suspicious",
                        "reason": "; ".join(reasons),
                    }
                else:
                    details[name] = {
                        "status": "ok",
                        "reason": "чертёж найден, реклама/фото не обнаружены, "
                        "контент выглядит согласованным",
                    }
        except Exception as exc:
            suspicious_files.append(name)
            details[name] = {
                "status": "suspicious",
                "reason": f"не удалось открыть или обработать PDF: {exc}",
            }

    total = len(pdf_paths)
    result: dict = {
        "summary": {
            "total_drawings": total,
            "suspicious_files": suspicious_files,
        },
        "details": details,
    }
    _print_validation_report(result)
    return result


def _configure_stdout_utf8() -> None:
    """Чтобы кириллица в отчёте не ломалась в cp1252 (Windows)."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _print_validation_report(result: dict) -> None:
    _configure_stdout_utf8()
    summary = result["summary"]
    total = summary["total_drawings"]
    suspicious = summary["suspicious_files"]
    details = result["details"]

    print()
    print("=== Отчёт validate_drawings ===")
    print(f"Всего файлов drawing_*.pdf: {total}")
    print(f"Подозрительных: {len(suspicious)}")
    if suspicious:
        print("Список подозрительных:")
        for fname in suspicious:
            print(f"  • {fname}")
            print(f"    {details[fname]['reason']}")
    else:
        print("Подозрительных файлов нет.")

    print()
    print("Рекомендация по фильтрации:")
    if total == 0:
        print(
            "  Каталог пуст или нет совпадений drawing_*.pdf — "
            "проверьте output_dir и что extract_drawings отработал."
        )
    elif len(suspicious) == 0:
        print(
            "  Результаты согласованы с эвристиками. "
            "Пороги в utils.pdf_analyzer менять не требуется."
        )
    elif len(suspicious) <= max(1, total // 5):
        print(
            "  Доля подозрительных невелика — просмотрите файлы вручную; "
            "глобально пороги можно не менять."
        )
    elif len(suspicious) >= max(2, int(total * 0.4)):
        print(
            "  Много отклонений: имеет смысл ужесточить классификацию "
            "(пороги/маркеры в pdf_analyzer) или проверить качество исходного альбома."
        )
    else:
        print(
            "  Средняя доля подозрительных: откройте reason в details. "
            "Если часто срабатывает проверка «несколько страниц без…», "
            "ослабьте условие в validate_drawings; если часто «реклама/фото» — "
            "ужесточьте отсев в extract_drawings / pdf_analyzer."
        )
    print()


def extract_drawings(
    input_pdf_path: str,
    output_dir: str,
    max_pages_per_drawing: int | None = 48,
    *,
    respect_project_boundaries: bool = True,
    glue_unknown_gap: int = 3,
    glue_edge_unknown: int = 2,
) -> dict[str, str]:
    """
    Разбивает PDF на файлы по чертежам.

    В одном выходном PDF остаются вместе: титул/данные проекта (``is_project_info``),
    чертёж, экспликация и примечания, если между ними только «пустые» страницы
    (мало текста / классификатор не уверен) — они подтягиваются склейкой внутри
    проекта. Реклама и фото по маркерам по-прежнему вырезаются.

    Если в исходнике несколько домов, по текстовым маркерам начала проекта
    (см. ``page_begins_new_project``) страницы делятся на сегменты; имена файлов
    получают префикс ``p01_``, ``p02_``, … чтобы не смешивать объекты.

    Args:
        max_pages_per_drawing: максимум страниц в одном файле; ``None`` — не резать
            по длине (только по рекламе и границам проекта).

    Returns:
        Словарь ``{имя_без_расширения: абсолютный_путь_к_pdf}``.
    """
    if max_pages_per_drawing is not None and max_pages_per_drawing < 1:
        raise ValueError("max_pages_per_drawing must be >= 1 or None")

    src_path = Path(input_pdf_path).expanduser().resolve()
    if not src_path.is_file():
        raise FileNotFoundError(str(src_path))

    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, str] = {}
    allocated_keys: set[str] = set()

    with fitz.open(str(src_path)) as doc:
        n = len(doc)
        classifications: list[PageClassification] = [
            classify_page(doc[i]) for i in range(n)
        ]
        new_proj = [
            page_begins_new_project(extract_features(doc[i])) for i in range(n)
        ]
        if respect_project_boundaries:
            segments = _project_segments(n, new_proj)
        else:
            segments = [(0, n - 1)] if n else []

        file_seq = 0
        for seg_i, (lo, hi) in enumerate(segments, start=1):
            seg_runs = _collect_glued_runs_for_segment(
                classifications,
                lo,
                hi,
                max_inner_gap=glue_unknown_gap,
                max_edge_unknown=glue_edge_unknown,
            )
            drawing_runs = [
                r for r in seg_runs if any(classifications[i]["is_drawing"] for i in r)
            ]

            proj_prefix = f"p{seg_i:02d}_" if respect_project_boundaries else ""

            for run in drawing_runs:
                if max_pages_per_drawing is None:
                    chunks = [run]
                    split_parts = False
                else:
                    m = max_pages_per_drawing
                    split_parts = len(run) > m
                    chunks = [run[s : s + m] for s in range(0, len(run), m)]

                for start, chunk in enumerate(chunks):
                    file_seq += 1

                    label_page_i = _first_drawing_index_in_chunk(chunk, classifications)
                    label = _drawing_label_from_page(doc[label_page_i])
                    base_stem = f"drawing_{proj_prefix}{label}"
                    if split_parts or start > 0:
                        base_stem = f"{base_stem}_p{file_seq:03d}"

                    dup = 1
                    out_path: Path
                    while True:
                        candidate = base_stem if dup == 1 else f"{base_stem}_{dup}"
                        out_path = out_dir / f"{candidate}.pdf"
                        extra = 0
                        while out_path.exists():
                            extra += 1
                            out_path = out_dir / f"{candidate}_{extra}.pdf"
                        key = out_path.stem
                        if key not in allocated_keys:
                            allocated_keys.add(key)
                            break
                        dup += 1

                    with fitz.open() as out_doc:
                        for pi in chunk:
                            out_doc.insert_pdf(doc, from_page=pi, to_page=pi)
                        out_doc.save(str(out_path))

                    result[out_path.stem] = str(out_path.resolve())

    return result


def main() -> None:
    """CLI: нарезка альбома и опционально проверка выходного каталога."""
    import argparse

    _configure_stdout_utf8()
    parser = argparse.ArgumentParser(
        description="Нарезка PDF-альбома на файлы по чертежам (с титулом и экспликацией рядом).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True, help="Исходный PDF")
    parser.add_argument("--out", "-o", required=True, help="Каталог для drawing_*.pdf")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=48,
        help="Макс. страниц в одном файле; 0 = без ограничения",
    )
    parser.add_argument(
        "--no-project-split",
        action="store_true",
        help="Не делить по маркерам начала проекта (весь PDF как один объект)",
    )
    parser.add_argument(
        "--glue-gap",
        type=int,
        default=3,
        help="Склеивать до N «пустых» страниц между чертежом и экспликацией",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="После нарезки вызвать validate_drawings для каталога --out",
    )
    args = parser.parse_args()

    max_p: int | None = None if args.max_pages == 0 else args.max_pages
    created = extract_drawings(
        args.input,
        args.out,
        max_pages_per_drawing=max_p,
        respect_project_boundaries=not args.no_project_split,
        glue_unknown_gap=args.glue_gap,
    )
    print(f"Создано файлов: {len(created)}")
    for stem, path in sorted(created.items()):
        print(f"  • {stem}.pdf → {path}")
    if args.validate:
        validate_drawings(args.out)


if __name__ == "__main__":
    main()
