"""
Тесты scripts.pdf_cutter: нарезка PDF с подменой classify_page
и контроль classify_from_features для фиктивных признаков страницы.
"""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from utils.pdf_analyzer import PageClassification, PageFeatures, classify_from_features


def _cls(
    *,
    drawing: bool = False,
    explication: bool = False,
    annotation: bool = False,
    project_info: bool = False,
    junk: bool = False,
) -> PageClassification:
    return PageClassification(
        is_drawing=drawing,
        is_explication=explication,
        is_annotation=annotation,
        is_project_info=project_info,
        is_irrelevant_photo_or_ad=junk,
    )


def _make_n_page_pdf(path: Path, n: int) -> None:
    doc = fitz.open()
    for _ in range(n):
        doc.new_page()
    doc.save(str(path))
    doc.close()


def _patch_classifier(
    monkeypatch: pytest.MonkeyPatch,
    seq: list[PageClassification],
    module_name: str = "scripts.pdf_cutter",
) -> None:
    def fake_classify(page: fitz.Page) -> PageClassification:
        i = page.number
        if i < 0 or i >= len(seq):
            raise IndexError(f"classify_page: page {i} out of range for seq len {len(seq)}")
        return seq[i]

    monkeypatch.setattr(f"{module_name}.classify_page", fake_classify)


# ---------------------------------------------------------------------------
# classify_from_features — те же ключи, что ожидает нарезка
# ---------------------------------------------------------------------------


def test_classifier_shape_matches_page_classification():
    """Фиктивные PageFeatures + classify_from_features дают PageClassification."""
    features = PageFeatures(
        text="чертёж № 5-1 план этажа",
        text_upper="ЧЕРТЁЖ № 5-1 ПЛАН ЭТАЖА",
        char_count=24,
        image_area_ratio=0.0,
        image_count=0,
    )
    c = classify_from_features(features)
    assert set(c.keys()) == {
        "is_drawing",
        "is_explication",
        "is_annotation",
        "is_project_info",
        "is_irrelevant_photo_or_ad",
    }
    assert isinstance(c["is_drawing"], bool)


# ---------------------------------------------------------------------------
# extract_drawings (monkeypatch classify_page)
# ---------------------------------------------------------------------------


def test_extract_two_groups_split_by_junk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.pdf_cutter import extract_drawings

    pdf = tmp_path / "in.pdf"
    _make_n_page_pdf(pdf, 3)
    _patch_classifier(
        monkeypatch,
        [_cls(drawing=True), _cls(junk=True), _cls(drawing=True)],
    )

    out = tmp_path / "out"
    out.mkdir()
    r = extract_drawings(str(pdf), str(out))

    assert len(r) == 2
    paths = sorted(r.values())
    with fitz.open(paths[0]) as a, fitz.open(paths[1]) as b:
        assert len(a) == 1
        assert len(b) == 1


def test_extract_drawing_with_explication_same_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.pdf_cutter import extract_drawings

    pdf = tmp_path / "in.pdf"
    _make_n_page_pdf(pdf, 2)
    _patch_classifier(
        monkeypatch,
        [_cls(explication=True), _cls(drawing=True)],
    )

    out = tmp_path / "out"
    r = extract_drawings(str(pdf), str(out))

    assert len(r) == 1
    only = next(iter(r.values()))
    with fitz.open(only) as doc:
        assert len(doc) == 2


def test_extract_skips_annotation_only_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.pdf_cutter import extract_drawings

    pdf = tmp_path / "in.pdf"
    _make_n_page_pdf(pdf, 2)
    _patch_classifier(monkeypatch, [_cls(annotation=True), _cls(annotation=True)])

    out = tmp_path / "out"
    r = extract_drawings(str(pdf), str(out))

    assert r == {}


def test_extract_max_pages_splits_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.pdf_cutter import extract_drawings

    pdf = tmp_path / "in.pdf"
    _make_n_page_pdf(pdf, 12)
    seq = [_cls(drawing=True)] * 12
    _patch_classifier(monkeypatch, seq)

    out = tmp_path / "out"
    r = extract_drawings(str(pdf), str(out), max_pages_per_drawing=10)

    assert len(r) == 2
    counts: list[int] = []
    for p in r.values():
        with fitz.open(p) as d:
            counts.append(len(d))
    assert sorted(counts) == [2, 10]


def test_extract_preserves_page_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.pdf_cutter import extract_drawings

    pdf = tmp_path / "in.pdf"
    doc = fitz.open()
    for i in range(3):
        p = doc.new_page()
        p.insert_text((72, 100), f"MARK_{i}")
    doc.save(str(pdf))
    doc.close()

    _patch_classifier(monkeypatch, [_cls(drawing=True)] * 3)

    out = tmp_path / "out"
    r = extract_drawings(str(pdf), str(out))
    out_pdf = next(iter(r.values()))

    with fitz.open(out_pdf) as cut:
        texts = [cut[i].get_text() for i in range(3)]
    assert "MARK_0" in texts[0] and "MARK_1" in texts[1] and "MARK_2" in texts[2]


def test_extract_file_not_found(tmp_path: Path) -> None:
    from scripts.pdf_cutter import extract_drawings

    missing = tmp_path / "missing.pdf"
    with pytest.raises(FileNotFoundError):
        extract_drawings(str(missing), str(tmp_path / "out"))


def test_extract_invalid_max_pages() -> None:
    from scripts.pdf_cutter import extract_drawings

    with pytest.raises(ValueError, match="max_pages_per_drawing"):
        extract_drawings("x.pdf", ".", max_pages_per_drawing=0)


def test_extract_glues_unknown_between_kept_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Пустая по классификатору страница между титулом и чертежом не рвёт один PDF."""
    from scripts.pdf_cutter import extract_drawings

    pdf = tmp_path / "in.pdf"
    _make_n_page_pdf(pdf, 3)
    _patch_classifier(
        monkeypatch,
        [
            _cls(project_info=True),
            _cls(),
            _cls(drawing=True),
        ],
    )

    out = tmp_path / "out"
    r = extract_drawings(str(pdf), str(out), respect_project_boundaries=False)

    assert len(r) == 1
    with fitz.open(next(iter(r.values()))) as doc:
        assert len(doc) == 3


def test_extract_respects_project_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import pdf_cutter as pc

    pdf = tmp_path / "in.pdf"
    _make_n_page_pdf(pdf, 3)
    _patch_classifier(
        monkeypatch,
        [_cls(drawing=True), _cls(drawing=True), _cls(drawing=True)],
    )

    def fake_boundary(f):
        return f.text == "start2"

    monkeypatch.setattr(pc, "page_begins_new_project", lambda features: fake_boundary(features))

    doc = fitz.open()
    for t in ("p1", "start2", "p3"):
        p = doc.new_page()
        p.insert_text((72, 100), t)
    doc.save(str(pdf))
    doc.close()

    out = tmp_path / "out"
    r = pc.extract_drawings(str(pdf), str(out), respect_project_boundaries=True)

    assert len(r) == 2
    stems = sorted(r.keys())
    assert stems[0].startswith("drawing_p01_")
    assert stems[1].startswith("drawing_p02_")
