"""
pdf_extractor.py — разбор PDF на текст и изображения страниц.

Использует PyMuPDF (fitz) для извлечения текста и pymupdf4llm для
структурированного Markdown-представления. Каждая страница сохраняется
как PNG (300 dpi) и как JSON с текстом.

Пример запуска:
    python -m scripts.pdf_extractor --pdf data/pdf/drawing.pdf --id 123-А
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

try:
    import pymupdf4llm
    HAS_PYMUPDF4LLM = True
except ImportError:
    HAS_PYMUPDF4LLM = False

from scripts.models import ExtractionResult, PageExtraction

logger = logging.getLogger(__name__)

# Разрешение рендеринга страниц в PNG (dpi)
RENDER_DPI = 300
# Минимальное количество символов на странице, чтобы считать её текстовой
MIN_TEXT_CHARS = 30
# Минимальная площадь изображения на странице (px²), чтобы считать её чертежом
MIN_IMAGE_AREA = 10_000


def _page_has_drawings(page: fitz.Page) -> bool:
    """Проверяет, есть ли на странице векторная графика или растровые изображения."""
    images = page.get_images(full=False)
    if images:
        for img in images:
            xref = img[0]
            try:
                img_rect = page.get_image_rects(xref)
                if img_rect:
                    r = img_rect[0]
                    if r.width * r.height >= MIN_IMAGE_AREA:
                        return True
            except Exception:
                pass

    # Проверяем наличие векторных путей (линии, дуги — признак чертежа)
    paths = page.get_drawings()
    return len(paths) > 5


def render_page_to_image(page: fitz.Page, output_path: Path, dpi: int = RENDER_DPI) -> Path:
    """Рендерит страницу PDF в PNG-файл с заданным разрешением."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pix.save(str(output_path))
    return output_path


def extract_page_text(page: fitz.Page) -> str:
    """Извлекает текст со страницы с сохранением структуры (блоки, строки)."""
    return page.get_text("text").strip()


def extract_pdf(
    pdf_path: str,
    drawing_id: str,
    images_dir: str = "data/images",
    text_dir: str = "data/extracted_text",
    render_all_pages: bool = True,
) -> ExtractionResult:
    """
    Разбирает PDF на страницы: извлекает текст и сохраняет PNG изображения.

    Args:
        pdf_path: Путь к исходному PDF-файлу.
        drawing_id: Идентификатор чертежа (используется в именах файлов).
        images_dir: Директория для сохранения PNG.
        text_dir: Директория для сохранения JSON с текстом.
        render_all_pages: Если True — рендерит все страницы, иначе только с чертежами.

    Returns:
        ExtractionResult с данными по каждой странице.
    """
    pdf_path_obj = Path(pdf_path)
    if not pdf_path_obj.exists():
        raise FileNotFoundError(f"PDF не найден: {pdf_path}")

    images_dir_obj = Path(images_dir)
    text_dir_obj = Path(text_dir)
    images_dir_obj.mkdir(parents=True, exist_ok=True)
    text_dir_obj.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path_obj))
    total_pages = len(doc)
    pages: list[PageExtraction] = []

    logger.info("Открыт PDF: %s (%d страниц)", pdf_path, total_pages)

    # Если pymupdf4llm доступен — используем его для лучшего извлечения текста
    md_texts: list[str] = []
    if HAS_PYMUPDF4LLM:
        try:
            md_texts = pymupdf4llm.to_markdown(str(pdf_path_obj), page_chunks=True)
            logger.info("pymupdf4llm: извлечён Markdown (%d страниц)", len(md_texts))
        except Exception as e:
            logger.warning("pymupdf4llm не сработал, используем fitz напрямую: %s", e)

    for page_idx in range(total_pages):
        page_num = page_idx + 1
        page = doc[page_idx]

        # Текст: сначала пробуем pymupdf4llm, затем базовый fitz
        if md_texts and page_idx < len(md_texts):
            chunk = md_texts[page_idx]
            pdf_text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
        else:
            pdf_text = extract_page_text(page)

        has_drawings = _page_has_drawings(page)
        should_render = render_all_pages or has_drawings or len(pdf_text) < MIN_TEXT_CHARS

        image_path: Optional[str] = None
        if should_render:
            img_filename = f"{drawing_id}_page_{page_num:03d}.png"
            img_path = images_dir_obj / img_filename
            render_page_to_image(page, img_path)
            image_path = str(img_path)
            logger.debug("Страница %d → %s", page_num, img_path)

        page_data = PageExtraction(
            page_number=page_num,
            pdf_text=pdf_text,
            image_path=image_path,
            has_drawings=has_drawings,
        )
        pages.append(page_data)

        # Сохраняем текст страницы в JSON
        json_path = text_dir_obj / f"{drawing_id}_page_{page_num:03d}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(page_data.model_dump(), f, ensure_ascii=False, indent=2)

    doc.close()

    result = ExtractionResult(
        pdf_path=str(pdf_path_obj.resolve()),
        drawing_id=drawing_id,
        total_pages=total_pages,
        pages=pages,
    )

    logger.info(
        "Извлечение завершено: %d страниц, %d с чертежами",
        total_pages,
        sum(1 for p in pages if p.has_drawings),
    )
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Извлечь текст и изображения из PDF")
    parser.add_argument("--pdf", required=True, help="Путь к PDF-файлу")
    parser.add_argument("--id", required=True, dest="drawing_id", help="Идентификатор чертежа")
    parser.add_argument("--images-dir", default="data/images")
    parser.add_argument("--text-dir", default="data/extracted_text")
    args = parser.parse_args()

    result = extract_pdf(
        pdf_path=args.pdf,
        drawing_id=args.drawing_id,
        images_dir=args.images_dir,
        text_dir=args.text_dir,
    )
    print(f"\nРезультат:")
    print(f"  PDF: {result.pdf_path}")
    print(f"  Страниц: {result.total_pages}")
    for p in result.pages:
        print(
            f"  Стр.{p.page_number}: текст={len(p.pdf_text)} симв., "
            f"чертёж={'да' if p.has_drawings else 'нет'}, "
            f"изображение={p.image_path or '—'}"
        )


if __name__ == "__main__":
    main()
