"""
Тесты для pdf_extractor.py.

Для запуска: pytest tests/test_extractor.py -v
"""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import pytest

from scripts.models import ExtractionResult, PageExtraction


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------


def _make_minimal_pdf(path: Path, text: str = "Drawing 123-A M16x1.5") -> None:
    """Создаёт минимальный валидный PDF с одной страницей и текстом."""
    lines = [
        "%PDF-1.4",
        "1 0 obj<</Type /Catalog /Pages 2 0 R>>endobj",
        f"2 0 obj<</Type /Pages /Kids [3 0 R] /Count 1>>endobj",
        "3 0 obj<</Type /Page /Parent 2 0 R /MediaBox [0 0 595 842]"
        " /Contents 4 0 R /Resources <</Font <</F1 5 0 R>>>>>>endobj",
    ]
    stream_content = f"BT /F1 12 Tf 50 750 Td ({text}) Tj ET"
    stream_bytes = stream_content.encode("latin-1")
    lines.append(f"4 0 obj<</Length {len(stream_bytes)}>>")
    lines.append("stream")
    content = "\n".join(lines) + "\n"
    content_bytes = content.encode("latin-1") + stream_bytes + b"\nendstream\nendobj\n"
    font_obj = (
        b"5 0 obj<</Type /Font /Subtype /Type1 /BaseFont /Helvetica>>endobj\n"
    )
    content_bytes += font_obj

    xref_offset = len(content_bytes)
    xref = (
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000266 00000 n \n"
        b"0000000350 00000 n \n"
    )
    trailer = f"trailer<</Size 6 /Root 1 0 R>>\nstartxref\n{xref_offset}\n%%EOF\n"
    path.write_bytes(content_bytes + xref + trailer.encode())


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Минимальный PDF для тестирования."""
    pdf_path = tmp_path / "test_drawing.pdf"
    _make_minimal_pdf(pdf_path)
    return pdf_path


@pytest.fixture
def output_dirs(tmp_path: Path) -> tuple[Path, Path]:
    images = tmp_path / "images"
    texts = tmp_path / "texts"
    images.mkdir()
    texts.mkdir()
    return images, texts


# ---------------------------------------------------------------------------
# Тесты
# ---------------------------------------------------------------------------


def test_extract_pdf_returns_result(sample_pdf: Path, output_dirs: tuple[Path, Path]) -> None:
    """extract_pdf возвращает ExtractionResult с корректным числом страниц."""
    from scripts.pdf_extractor import extract_pdf

    images_dir, text_dir = output_dirs
    result = extract_pdf(
        pdf_path=str(sample_pdf),
        drawing_id="TEST-01",
        images_dir=str(images_dir),
        text_dir=str(text_dir),
    )

    assert isinstance(result, ExtractionResult)
    assert result.total_pages == 1
    assert len(result.pages) == 1
    assert result.drawing_id == "TEST-01"


def test_extract_pdf_creates_image(sample_pdf: Path, output_dirs: tuple[Path, Path]) -> None:
    """extract_pdf создаёт PNG-файл для каждой страницы."""
    from scripts.pdf_extractor import extract_pdf

    images_dir, text_dir = output_dirs
    result = extract_pdf(
        pdf_path=str(sample_pdf),
        drawing_id="TEST-01",
        images_dir=str(images_dir),
        text_dir=str(text_dir),
    )

    page = result.pages[0]
    assert page.image_path is not None
    assert Path(page.image_path).exists()
    assert Path(page.image_path).suffix == ".png"


def test_extract_pdf_creates_json(sample_pdf: Path, output_dirs: tuple[Path, Path]) -> None:
    """extract_pdf сохраняет JSON с данными страницы."""
    from scripts.pdf_extractor import extract_pdf

    images_dir, text_dir = output_dirs
    extract_pdf(
        pdf_path=str(sample_pdf),
        drawing_id="TEST-01",
        images_dir=str(images_dir),
        text_dir=str(text_dir),
    )

    json_files = list(text_dir.glob("*.json"))
    assert len(json_files) == 1

    data = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert "page_number" in data
    assert data["page_number"] == 1


def test_extract_pdf_missing_file() -> None:
    """extract_pdf выбрасывает FileNotFoundError при отсутствии PDF."""
    from scripts.pdf_extractor import extract_pdf

    with pytest.raises(FileNotFoundError):
        extract_pdf(
            pdf_path="nonexistent.pdf",
            drawing_id="NONE",
            images_dir="/tmp/img",
            text_dir="/tmp/txt",
        )


def test_page_extraction_model() -> None:
    """PageExtraction корректно сериализуется в dict."""
    page = PageExtraction(
        page_number=1,
        pdf_text="Тест",
        image_path="/tmp/img.png",
        has_drawings=True,
    )
    d = page.model_dump()
    assert d["page_number"] == 1
    assert d["pdf_text"] == "Тест"
    assert d["has_drawings"] is True


def test_extract_pdf_uses_committed_minimal_fixture(tmp_path: Path) -> None:
    """Фикстура tests/fixtures/minimal.pdf (план шаг 11)."""
    from scripts.pdf_extractor import extract_pdf

    pdf_path = Path(__file__).resolve().parent / "fixtures" / "minimal.pdf"
    assert pdf_path.is_file()
    images_dir, text_dir = tmp_path / "img", tmp_path / "txt"
    images_dir.mkdir()
    text_dir.mkdir()
    result = extract_pdf(
        pdf_path=str(pdf_path),
        drawing_id="FIXTURE-PDF",
        images_dir=str(images_dir),
        text_dir=str(text_dir),
    )
    assert result.total_pages == 1
    assert result.pages[0].image_path is not None
