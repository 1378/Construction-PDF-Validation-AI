"""
Тесты для rule_validator.py.

Для запуска: pytest tests/test_validator.py -v
"""

from __future__ import annotations

import pytest

from scripts.models import (
    ComplianceReport,
    ConfidenceLevel,
    DrawingElement,
    DrawingReport,
    SourceType,
)
from scripts.rule_validator import (
    _check_dimension,
    _check_material,
    _check_designation,
    _parse_numeric,
    validate_report,
    load_rules,
)


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------

SAMPLE_RULES = {
    "dimensions": [
        {
            "field": "bolt_size",
            "allowed_values": ["M12x1.75", "M14x2", "M16x2"],
            "unit": "мм",
            "standard": "ГОСТ 24705-2004",
            "tz_reference": "Раздел 3, п. 3.1",
        },
        {
            "field": "wall_thickness",
            "min": 4.0,
            "max": 12.0,
            "unit": "мм",
            "standard": "ГОСТ 8731-74",
            "tz_reference": "Раздел 4, п. 4.2.1",
        },
    ],
    "materials": {
        "allowed": ["Ст3сп", "09Г2С", "12Х18Н10Т"],
        "forbidden": ["Ст5", "Ст6"],
        "standard": "ГОСТ 380-2005",
        "tz_reference": "Раздел 2, п. 2.1",
    },
    "designations": {
        "pattern": r"^[А-ЯA-Z]{2,5}-\d{3,6}(-\d{2})?$",
        "examples": ["КД-00100", "СБ-00200-01"],
        "tz_reference": "Раздел 1, п. 1.3",
    },
    "coatings": {
        "allowed_codes": ["Хим.Окс", "Ц6", "Н12"],
        "standard": "ГОСТ 9.306-85",
        "tz_reference": "Раздел 6, п. 6.1",
    },
}


def _make_report(elements: list[DrawingElement]) -> DrawingReport:
    return DrawingReport(
        drawing_id="TEST-01",
        pdf_path="test.pdf",
        elements=elements,
        compliance=ComplianceReport(),
    )


def _make_element(**kwargs) -> DrawingElement:
    defaults = {
        "item_id": "Поз01",
        "name": "Болт",
        "source": SourceType.PDF_TEXT,
        "confidence": ConfidenceLevel.MEDIUM,
    }
    defaults.update(kwargs)
    return DrawingElement(**defaults)


# ---------------------------------------------------------------------------
# Тесты _parse_numeric
# ---------------------------------------------------------------------------


def test_parse_numeric_simple():
    assert _parse_numeric("4.5 мм") == 4.5


def test_parse_numeric_int():
    assert _parse_numeric("12") == 12.0


def test_parse_numeric_comma():
    assert _parse_numeric("3,5") == 3.5


def test_parse_numeric_none_on_text():
    # "M16x1.5" — начинается с буквы, первое число 16
    result = _parse_numeric("M16x1.5")
    assert result == 16.0  # парсим первое число


def test_parse_numeric_empty():
    assert _parse_numeric(None) is None
    assert _parse_numeric("") is None


# ---------------------------------------------------------------------------
# Тесты _check_dimension
# ---------------------------------------------------------------------------


def test_check_dimension_allowed_values_ok():
    rule = SAMPLE_RULES["dimensions"][0]  # bolt_size с allowed_values
    issue = _check_dimension("M16x2", rule, "bolt_size")
    assert issue is None


def test_check_dimension_allowed_values_fail():
    rule = SAMPLE_RULES["dimensions"][0]
    issue = _check_dimension("M22x2.5", rule, "bolt_size")
    assert issue is not None
    assert "M22x2.5" in issue.message
    assert issue.field == "bolt_size"


def test_check_dimension_range_ok():
    rule = SAMPLE_RULES["dimensions"][1]  # wall_thickness 4.0–12.0
    issue = _check_dimension("8.0 мм", rule, "wall_thickness")
    assert issue is None


def test_check_dimension_range_too_low():
    rule = SAMPLE_RULES["dimensions"][1]
    issue = _check_dimension("2.5 мм", rule, "wall_thickness")
    assert issue is not None
    assert "меньше минимально допустимого" in issue.message
    assert "4.0" in issue.expected


def test_check_dimension_range_too_high():
    rule = SAMPLE_RULES["dimensions"][1]
    issue = _check_dimension("15.0 мм", rule, "wall_thickness")
    assert issue is not None
    assert "превышает максимально допустимое" in issue.message


def test_check_dimension_no_value():
    rule = SAMPLE_RULES["dimensions"][1]
    issue = _check_dimension(None, rule, "wall_thickness")
    assert issue is None  # нет значения — нет issue (попадёт в missing_info)


# ---------------------------------------------------------------------------
# Тесты _check_material
# ---------------------------------------------------------------------------


def test_check_material_allowed():
    issue, missing = _check_material("Ст3сп", SAMPLE_RULES)
    assert issue is None
    assert missing is None


def test_check_material_not_in_allowed():
    issue, missing = _check_material("Ст4", SAMPLE_RULES)
    assert issue is not None
    assert "Ст4" in issue.message
    assert missing is None


def test_check_material_forbidden():
    issue, missing = _check_material("Ст5", SAMPLE_RULES)
    assert issue is not None
    assert "запрещённых" in issue.message


def test_check_material_missing():
    issue, missing = _check_material(None, SAMPLE_RULES)
    assert issue is None
    assert missing is not None
    assert "не указана" in missing


# ---------------------------------------------------------------------------
# Тесты _check_designation
# ---------------------------------------------------------------------------


def test_check_designation_valid():
    issue, missing = _check_designation("КД-00100", SAMPLE_RULES)
    assert issue is None


def test_check_designation_invalid():
    issue, missing = _check_designation("неверное-обозначение", SAMPLE_RULES)
    assert issue is not None
    assert "паттерн" in issue.message


def test_check_designation_none():
    issue, missing = _check_designation(None, SAMPLE_RULES)
    assert issue is None
    assert missing is not None


# ---------------------------------------------------------------------------
# Тесты validate_report (интеграционные)
# ---------------------------------------------------------------------------


def test_validate_report_all_compliant():
    elements = [
        _make_element(size="M16x2", material="Ст3сп", designation="КД-00100"),
    ]
    report = _make_report(elements)
    result = validate_report(report, SAMPLE_RULES)
    assert result.compliance.is_compliant is True
    assert len(result.compliance.issues) == 0


def test_validate_report_bad_material():
    elements = [
        _make_element(material="Ст5"),
    ]
    report = _make_report(elements)
    result = validate_report(report, SAMPLE_RULES)
    assert result.compliance.is_compliant is False
    assert any("Ст5" in i.message for i in result.compliance.issues)


def test_validate_report_bad_size():
    elements = [
        _make_element(size="M22x2.5", material="Ст3сп"),
    ]
    report = _make_report(elements)
    result = validate_report(report, SAMPLE_RULES)
    assert result.compliance.is_compliant is False
    assert len(result.compliance.issues) > 0


def test_validate_report_empty_elements():
    report = _make_report([])
    result = validate_report(report, SAMPLE_RULES)
    assert result.compliance.is_compliant is None
    assert any("не извлечены" in m for m in result.compliance.missing_info)


def test_validate_report_empty_rules():
    elements = [_make_element(material="Ст5")]  # запрещён, но rules пустые
    report = _make_report(elements)
    result = validate_report(report, {})
    assert any("пустой" in m for m in result.compliance.missing_info)
