"""Тесты нормализации ComplianceReport (ответы LLM: строки и dict с field/text)."""

from __future__ import annotations

import pytest

from scripts.models import ComplianceReport


def test_missing_info_and_citations_plain_strings():
    r = ComplianceReport(
        missing_info=["a"],
        citations=["b"],
    )
    assert r.missing_info == ["a"]
    assert r.citations == ["b"]


def test_missing_info_dict_with_field_citations_dict_with_text():
    r = ComplianceReport(
        missing_info=[{"field": "данные не соответствуют ТЗ"}],
        citations=[{"text": "не указано в ТЗ"}],
    )
    assert r.missing_info == ["данные не соответствуют ТЗ"]
    assert r.citations == ["не указано в ТЗ"]


def test_fallback_str_for_unknown_dict_structure():
    r = ComplianceReport(
        missing_info=[{"unknown": 1}],
        citations=[{}],
    )
    assert "unknown" in r.missing_info[0] or r.missing_info[0].startswith("{")
    assert r.citations[0].startswith("{")


def test_mixed_list_and_string_coercion():
    r = ComplianceReport(
        missing_info=["x", {"field": "y", "text": "ignored_for_missing"}],
        citations=[123],
    )
    assert r.missing_info == ["x", "y"]
    assert r.citations == ["123"]


def test_single_string_normalized_to_one_element():
    r = ComplianceReport(missing_info="one", citations="two")
    assert r.missing_info == ["one"]
    assert r.citations == ["two"]
