"""
Тесты фикстур из tests/fixtures/ (план: минимальный PDF + sample JSON).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.models import DrawingReport
from scripts.rule_validator import load_rules, validate_report

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_minimal_pdf_fixture_exists_and_opens():
    pdf = FIXTURES / "minimal.pdf"
    assert pdf.is_file()
    import fitz

    doc = fitz.open(str(pdf))
    assert len(doc) == 1
    text = doc[0].get_text()
    doc.close()
    assert "Drawing" in text or "123" in text


def test_sample_drawing_report_json_roundtrip():
    path = FIXTURES / "sample_drawing_report.json"
    assert path.is_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    report = DrawingReport.model_validate(data)
    assert report.drawing_id == "FIXTURE-01"
    assert len(report.elements) == 1
    assert report.elements[0].size == "M16x2"


def test_sample_report_passes_rule_validator_with_project_rules():
    path = FIXTURES / "sample_drawing_report.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    report = DrawingReport.model_validate(data)
    rules_path = Path(__file__).resolve().parent.parent / "config" / "rules.json"
    if not rules_path.is_file():
        pytest.skip("config/rules.json отсутствует")
    rules = load_rules(str(rules_path))
    result = validate_report(report, rules)
    assert result.compliance.is_compliant is not None
