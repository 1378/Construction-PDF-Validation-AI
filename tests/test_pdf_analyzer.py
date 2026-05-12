"""Тесты для utils.pdf_analyzer (через classify_from_features / PageFeatures)."""

from __future__ import annotations

from utils.pdf_analyzer import (
    PageFeatures,
    classify_from_features,
)


def _f(
    text: str,
    *,
    image_area_ratio: float = 0.0,
    image_count: int = 0,
) -> PageFeatures:
    collapsed = " ".join(text.replace("\xa0", " ").split())
    return PageFeatures(
        text=collapsed.lower(),
        text_upper=collapsed.upper(),
        char_count=len(collapsed.lower()),
        image_area_ratio=image_area_ratio,
        image_count=image_count,
    )


def test_drawing_strong_title():
    r = classify_from_features(_f("Чертёж № 3-4. План этажа"))
    assert r["is_drawing"] is True
    assert r["is_explication"] is False
    assert r["is_project_info"] is False
    assert r["is_irrelevant_photo_or_ad"] is False


def test_explication_strong():
    r = classify_from_features(_f("Экспликация помещений. Площадь квартиры."))
    assert r["is_explication"] is True
    assert r["is_drawing"] is False


def test_annotation_no_drawing_title():
    r = classify_from_features(
        _f("Примечания. Масштаб 1:100. Условные обозначения. Размеры 1200 x 800")
    )
    assert r["is_annotation"] is True
    assert r["is_drawing"] is False


def test_low_confidence_all_false_except_explicit_ad():
    r = classify_from_features(_f("разный текст без маркеров enough length " * 8))
    assert r["is_drawing"] is False
    assert r["is_explication"] is False
    assert r["is_annotation"] is False
    assert r["is_project_info"] is False
    assert r["is_irrelevant_photo_or_ad"] is False


def test_project_info_title_low_confidence():
    r = classify_from_features(
        _f(
            "Заказчик ООО Ромашка. Объект капитального строительства. "
            "Наименование объекта: жилой дом. Состав проектной документации."
        )
    )
    assert r["is_drawing"] is False
    assert r["is_project_info"] is True
    assert r["is_irrelevant_photo_or_ad"] is False


def test_low_confidence_explicit_reklama():
    r0 = classify_from_features(_f("x"))
    assert r0["is_irrelevant_photo_or_ad"] is False
    # низкая уверенность, но явный маркер
    r2 = classify_from_features(_f("реклама"))
    assert r2["is_irrelevant_photo_or_ad"] is True
    assert r2["is_drawing"] is False


def test_low_confidence_explicit_avtor_caps():
    r = classify_from_features(_f("АВТОР"))
    assert r["is_irrelevant_photo_or_ad"] is True


def test_irrelevant_photo_structural():
    r = classify_from_features(
        _f(" ", image_area_ratio=0.5, image_count=2),
    )
    # мало текста, много картинок
    assert r["is_irrelevant_photo_or_ad"] is True


def test_explicit_ad_overrides_drawing_when_both_present():
    r = classify_from_features(
        _f("План этажа. реклама скидки", image_area_ratio=0.5, image_count=1),
    )
    assert r["is_irrelevant_photo_or_ad"] is True
    assert r["is_drawing"] is False


def test_drawing_hint_suppresses_annotation():
    r = classify_from_features(_f("План этажа. Примечания и масштаб 1:50"))
    assert r["is_drawing"] is True
    assert r["is_annotation"] is False
