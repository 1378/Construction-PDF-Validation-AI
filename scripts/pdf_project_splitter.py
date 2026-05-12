"""
Разделение большого PDF-альбома на отдельные файлы по уникальным описаниям домов
(маркеры «жилой дом» / тип этажности + размеры + материал).

Поддерживается слабый text-layer: PyMuPDF + при необходимости Tesseract (rus+eng).

Запуск из корня проекта:
    python -m scripts.pdf_project_splitter --input album.pdf --output ./data/projects/
"""

from __future__ import annotations

import argparse
import io
import logging
import re
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import fitz
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)

# Поиск описания: совпадение гибкое; извлечённый фрагмент — как в документе (сравнение регистрозависимое).
_DIM_PARENS = r"\(\s*[\d\.,]+\s*[хx×]\s*[\d\.,]+\s*\)"
_DIM_FREE = r"[\d\.,]+\s*[хx×]\s*[\d\.,]+(?:\s*м\b)?"
_MATERIAL_PARENS = r"\([^)]{1,200}\)"

PROJECT_PATTERN = re.compile(
    rf"(?is)"
    rf"(?:"
    rf"ЖИЛОЙ\s+ДОМ(?:\s*{_DIM_PARENS})[^\n]{{0,400}}"
    rf"|"
    rf"ЖИЛОЙ\s+ДОМ[^\n]{{0,120}}?{_DIM_FREE}[^\n]{{0,280}}"
    rf"|"
    rf"(?:ДВУХЭТАЖНЫЙ|ОДНОЭТАЖНЫЙ|МАНСАРДНЫЙ|Двухэтажный|Одноэтажный|Мансардный)\s+"
    rf"жилой\s+дом[^\n]{{0,400}}?(?:{_DIM_PARENS}|{_DIM_FREE})[^\n]{{0,320}}"
    rf"(?:{_MATERIAL_PARENS})?"
    rf")",
    re.UNICODE,
)

_DIM_ONLY_IN_PARENS = re.compile(
    r"^[\d\.,]+\s*[хx×]\s*[\d\.,]+$",
    re.IGNORECASE | re.UNICODE,
)

_TEXT_FLAGS = fitz.TEXT_DEHYPHENATE | fitz.TEXT_PRESERVE_WHITESPACE

_REPLACEMENT_CHAR = "\ufffd"
_MOJIBAKE_HINT = re.compile(r"\ufffd{2,}")


def _normalize_for_match(text: str) -> str:
    t = text.replace("\xa0", " ")
    t = t.replace("×", "х").replace("x", "х").replace("X", "х")
    t = re.sub(r"\s+", " ", t)
    return t


def _is_garbled(text: str) -> bool:
    if _REPLACEMENT_CHAR in text or _MOJIBAKE_HINT.search(text):
        return True
    letters = len(re.findall(r"[\w\u0400-\u04FF]", text, re.UNICODE))
    if letters < 8:
        return False
    cyr = len(re.findall(r"[\u0400-\u04FF]", text))
    return letters > 20 and (cyr / letters) < 0.15


def _cyrillic_count(text: str) -> int:
    return len(re.findall(r"[\u0400-\u04FF]", text))


def _latin_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]", text))


_TITLE_HINTS = (
    "жилой",
    "дом",
    "мансард",
    "этажн",
    "гараж",
    "арболит",
    "брус",
    "кирпич",
    "одноэтаж",
    "двухэтаж",
    "вариант",
    "стен",
)


def _text_has_title_hints(text: str) -> bool:
    low = _normalize_for_match(text).lower()
    return any(h in low for h in _TITLE_HINTS)


def needs_ocr_for_detection(raw_text: str) -> bool:
    """True — text-layer страницы подозрителен: без OCR заголовок часто не находится."""
    s = (raw_text or "").strip()
    if not s:
        return True
    if len(s) < 100:
        return True
    if _is_garbled(raw_text):
        return True
    cyr = _cyrillic_count(s)
    lat = _latin_count(s)
    if len(s) >= 50 and cyr < 5 and lat >= 25:
        return True
    if len(s) >= 80 and cyr < 12 and not _text_has_title_hints(s):
        return True
    if len(s) >= 120 and cyr < 20 and not _text_has_title_hints(s):
        return True
    return False


def extract_project_signature(text: str) -> Optional[str]:
    """
    Извлекает фрагмент описания проекта из текста страницы.
    Сравнение подписей — регистрозависимое; нормализуются только пробелы (и nbsp).
    """
    if not text or not text.strip():
        return None
    match = PROJECT_PATTERN.search(text)
    if not match:
        return None
    full_match = match.group(0).strip()
    return re.sub(r"\s+", " ", full_match.replace("\xa0", " "))


def normalize_project_name(signature: str) -> str:
    """
    Короткое имя файла из подписи. Скобки вида «(7.6х9.2)» — только размер, не материал;
    «каркас»/«тамбур» ищутся в тексте целиком, если нет отдельных скобок с материалом.
    """
    low = signature.lower()
    parts: List[str] = []

    if re.search(r"двух", low):
        parts.append("2etaj")
    elif re.search(r"одно", low):
        parts.append("1etaj")
    elif re.search(r"мансард", low):
        parts.append("mansard")
    elif re.search(r"жилой\s+дом", low):
        parts.append("zhdom")

    size_match = re.search(
        r"(\d+[\.,]?\d*)\s*[хx×]\s*(\d+[\.,]?\d*)",
        signature,
        re.IGNORECASE,
    )
    if size_match:
        a = size_match.group(1).replace(",", ".")
        b = size_match.group(2).replace(",", ".")
        parts.append(f"{a}x{b}".replace(".", "_"))

    keyword_slugs = (
        ("кирпич", "kirpich"),
        ("арболит", "arbolit"),
        ("брус", "brus"),
        ("каркас", "karkas"),
        ("тамбур", "tambur"),
        ("гараж", "garage"),
    )
    material_slug: Optional[str] = None
    for m in reversed(list(re.finditer(r"\(([^)]{1,160})\)", signature))):
        inner = m.group(1).strip()
        if _DIM_ONLY_IN_PARENS.match(inner):
            continue
        tail = inner.lower()
        for key, slug in keyword_slugs:
            if key in tail:
                material_slug = slug
                break
        if material_slug:
            break
        slug = re.sub(r"[^\w]+", "_", tail, flags=re.UNICODE)[:32].strip("_")
        if slug:
            material_slug = slug
            break

    if not material_slug:
        for key, slug in keyword_slugs:
            if key in low:
                material_slug = slug
                break

    if material_slug:
        parts.append(material_slug)

    base = "_".join(p for p in parts if p) or "project"
    safe = re.sub(r"[^\w]+", "_", base, flags=re.UNICODE).strip("_").lower()
    if not safe:
        safe = "project"
    return f"project_{safe}"[:120]


def ocr_fallback(
    page: fitz.Page, *, dpi: int = 200, tesseract_config: str = "--oem 3 --psm 3"
) -> str:
    """Растр страницы → OCR (rus+eng)."""
    pix = page.get_pixmap(alpha=False, dpi=dpi)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    cfg = tesseract_config.strip()
    kw: dict[str, Any] = {"lang": "rus+eng"}
    if cfg:
        kw["config"] = cfg
    return pytesseract.image_to_string(img, **kw) or ""


def get_text_with_ocr(
    page: fitz.Page,
    *,
    use_ocr: bool,
    ocr_dpi: int,
    ocr_config: str,
) -> str:
    """Текст страницы; при слабом слое добавляет OCR."""
    raw = (page.get_text("text", flags=_TEXT_FLAGS) or "").strip()
    if not use_ocr:
        return raw
    if needs_ocr_for_detection(raw):
        ocr = ocr_fallback(page, dpi=ocr_dpi, tesseract_config=ocr_config)
        if ocr.strip():
            logger.info(
                "Страница %s: OCR из-за слабого text-layer (слой %s симв., OCR %s симв.)",
                page.number + 1,
                len(raw),
                len(ocr),
            )
            return raw + "\n" + ocr
    return raw


def find_project_boundaries(
    doc: fitz.Document,
    use_ocr: bool = True,
    *,
    ocr_dpi: int = 200,
    ocr_config: str = "--oem 3 --psm 3",
    debug_signatures: bool = False,
) -> List[Tuple[int, int, str]]:
    """
    (start_page, end_page, project_stem), страницы с 1.
    project_stem из normalize_project_name; при коллизиях — суффикс _2, _3, …
    """
    project_signatures: List[Tuple[int, str]] = []
    debug_lines: List[str] = []

    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = get_text_with_ocr(
            page, use_ocr=use_ocr, ocr_dpi=ocr_dpi, ocr_config=ocr_config
        )
        signature = extract_project_signature(text)
        if debug_signatures:
            preview = (signature or "")[:200].replace("\n", " ")
            debug_lines.append(
                f"стр.{i + 1}: {preview!r}" if preview else f"стр.{i + 1}: <нет подписи>"
            )

        if not signature:
            raw = (page.get_text("text", flags=_TEXT_FLAGS) or "").strip()
            if use_ocr and len(raw) < 80:
                ocr_only = ocr_fallback(page, dpi=ocr_dpi, tesseract_config=ocr_config)
                signature = extract_project_signature(raw + "\n" + ocr_only)
                if debug_signatures and signature:
                    debug_lines[-1] = (
                        f"стр.{i + 1}: (повторный OCR) {signature[:200]!r}"
                    )

        if signature:
            if not project_signatures or signature != project_signatures[-1][1]:
                project_signatures.append((i + 1, signature))
                logger.info(
                    "Новый проект со стр.%s: %s",
                    i + 1,
                    signature[:120] + ("…" if len(signature) > 120 else ""),
                )
                print(
                    f"✅ стр.{i + 1}: НОВЫЙ ПРОЕКТ "
                    f"'{signature[:60]}{'…' if len(signature) > 60 else ''}'"
                )
            else:
                logger.info(
                    "Тот же проект на стр.%s: %s",
                    i + 1,
                    signature[:100] + ("…" if len(signature) > 100 else ""),
                )
                print(
                    f"🔄 стр.{i + 1}: тот же проект "
                    f"'{signature[:60]}{'…' if len(signature) > 60 else ''}'"
                )

    if debug_signatures:
        print("\n--- debug-signatures (все страницы) ---")
        for line in debug_lines:
            print(line)
        print("--- конец debug-signatures ---\n")

    if not project_signatures:
        return []

    stem_counts: dict[str, int] = {}
    boundaries: List[Tuple[int, int, str]] = []

    for idx, (start_page, signature) in enumerate(project_signatures):
        if idx == len(project_signatures) - 1:
            end_page = doc.page_count
        else:
            end_page = project_signatures[idx + 1][0] - 1

        base_stem = normalize_project_name(signature)
        n = stem_counts.get(base_stem, 0) + 1
        stem_counts[base_stem] = n
        project_stem = base_stem if n == 1 else f"{base_stem}_{n}"

        boundaries.append((start_page, end_page, project_stem))
        n_pages = end_page - start_page + 1
        logger.info(
            "Блок %s: стр. %s–%s (%s стр.), файл=%s.pdf",
            project_stem,
            start_page,
            end_page,
            n_pages,
            project_stem,
        )
        print(
            f"📄 {project_stem}: стр. {start_page}-{end_page} ({n_pages} стр.)"
        )

    return boundaries


def split_by_projects(
    input_pdf: str,
    output_dir: str,
    use_ocr: bool = True,
    *,
    ocr_dpi: int = 200,
    ocr_config: str = "--oem 3 --psm 3",
    debug_signatures: bool = False,
) -> dict[str, Any]:
    """
    Возвращает dict:
      ``files`` — имя_файла.pdf -> абсолютный путь;
      ``boundaries`` — (start_page, end_page, project_stem), страницы с 1.
    """
    inp = Path(input_pdf).resolve()
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(inp))
    try:
        boundaries = find_project_boundaries(
            doc,
            use_ocr=use_ocr,
            ocr_dpi=ocr_dpi,
            ocr_config=ocr_config,
            debug_signatures=debug_signatures,
        )
        if not boundaries:
            logger.warning("Не найдено ни одного описания проекта — проверьте PDF и OCR.")
        result: dict[str, str] = {}
        for start, end, project_stem in boundaries:
            fname = f"{project_stem}.pdf"
            out_path = out / fname
            start0 = start - 1
            end0 = end - 1
            new_doc = fitz.open()
            try:
                new_doc.insert_pdf(doc, from_page=start0, to_page=end0)
                new_doc.save(str(out_path))
            finally:
                new_doc.close()
            n_pages = end - start + 1
            result[fname] = str(out_path)
            logger.info("Записан %s (%s стр.)", fname, n_pages)
    finally:
        doc.close()
    return {"files": result, "boundaries": boundaries}


def _print_report(boundaries: List[Tuple[int, int, str]]) -> None:
    print(f"\n✅ Разделено на {len(boundaries)} проектов:")
    for start, end, stem in boundaries:
        fname = f"{stem}.pdf"
        n = end - start + 1
        print(f"  {fname} — стр. {start}–{end} ({n} стр.)")


def main(argv: Optional[List[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass
    parser = argparse.ArgumentParser(
        description="Разделение PDF-альбома по уникальным описаниям домов (текст + OCR)."
    )
    parser.add_argument("--input", required=True, help="Путь к исходному PDF")
    parser.add_argument("--output", required=True, help="Каталог для выходных PDF")
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Не вызывать OCR на страницах с коротким/подозрительным text-layer",
    )
    parser.add_argument(
        "--ocr-dpi",
        type=int,
        default=200,
        help="DPI растра для Tesseract (по умолчанию 200)",
    )
    parser.add_argument(
        "--ocr-psm",
        type=int,
        default=3,
        help="Tesseract PSM (3=авто; см. tesseract --help-extra)",
    )
    parser.add_argument(
        "--debug-signatures",
        action="store_true",
        help="Вывести извлечённые подписи по каждой странице",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG в консоль")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    ocr_cfg = f"--oem 3 --psm {args.ocr_psm}"
    manifest = split_by_projects(
        args.input,
        args.output,
        use_ocr=not args.no_ocr,
        ocr_dpi=args.ocr_dpi,
        ocr_config=ocr_cfg,
        debug_signatures=args.debug_signatures,
    )
    _print_report(manifest["boundaries"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
