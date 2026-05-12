"""
rule_validator.py — детерминированная проверка элементов чертежа по rules.json.

НЕ использует LLM. Чистая Python-логика:
  - числовые диапазоны (min/max)
  - списки допустимых значений (allowed_values, allowed_codes)
  - паттерны regex (designation)
  - запрещённые значения (forbidden)

Это финальный слой защиты от галлюцинаций: даже если LLM что-то
«подтвердил», здесь выполняется проверка по числам и кодам из rules.json.

Пример запуска:
    python -m scripts.rule_validator --report data/report.json --rules config/rules.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Optional

from scripts.models import (
    ComplianceReport,
    ConfidenceLevel,
    DrawingReport,
    ValidationIssue,
)

logger = logging.getLogger(__name__)


def load_rules(rules_path: str = "config/rules.json") -> dict:
    """Загружает rules.json. Возвращает пустой dict если файл не найден."""
    path = Path(rules_path)
    if not path.exists():
        logger.warning("rules.json не найден: %s — проверка по правилам пропущена", rules_path)
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        logger.error("Ошибка парсинга rules.json: %s", e)
        return {}


def _parse_numeric(value: Optional[str]) -> Optional[float]:
    """
    Извлекает первое числовое значение из строки.
    Примеры: "4.5 мм" → 4.5, "M16x1.5" → None (буква в начале → не число).
    """
    if not value:
        return None
    match = re.search(r"[-+]?\d+(?:[.,]\d+)?", value.replace(",", "."))
    if match:
        try:
            return float(match.group().replace(",", "."))
        except ValueError:
            return None
    return None


def _check_dimension(
    element_value: Optional[str],
    rule: dict,
    field_label: str,
) -> Optional[ValidationIssue]:
    """
    Проверяет числовое значение по диапазону или списку допустимых значений.

    Returns:
        ValidationIssue если есть расхождение, иначе None.
    """
    if not element_value:
        return None  # нет значения — проверить нельзя (missing_info отдельно)

    ref = rule.get("tz_reference", "")
    std = rule.get("standard", "")
    unit = rule.get("unit", "")

    # Проверка по списку допустимых значений
    if "allowed_values" in rule:
        allowed = [str(v).strip() for v in rule["allowed_values"]]
        val_stripped = element_value.strip()
        if val_stripped not in allowed:
            return ValidationIssue(
                field=field_label,
                actual_value=element_value,
                expected=f"одно из: {', '.join(allowed)}",
                message=(
                    f"'{element_value}' не входит в список допустимых значений "
                    f"[{', '.join(allowed)}]"
                ),
                tz_reference=f"{ref} (стандарт: {std})" if ref else std,
            )
        return None

    # Проверка по диапазону min/max
    if "min" in rule or "max" in rule:
        numeric = _parse_numeric(element_value)
        if numeric is None:
            # Не удалось распарсить — добавим в missing, но не в issues
            return None

        low = rule.get("min")
        high = rule.get("max")

        if low is not None and numeric < low:
            return ValidationIssue(
                field=field_label,
                actual_value=element_value,
                expected=f"{low}–{high} {unit}",
                message=(
                    f"значение {numeric} {unit} меньше минимально допустимого "
                    f"{low} {unit}"
                ),
                tz_reference=f"{ref} (стандарт: {std})" if ref else std,
            )

        if high is not None and numeric > high:
            return ValidationIssue(
                field=field_label,
                actual_value=element_value,
                expected=f"{low}–{high} {unit}",
                message=(
                    f"значение {numeric} {unit} превышает максимально допустимое "
                    f"{high} {unit}"
                ),
                tz_reference=f"{ref} (стандарт: {std})" if ref else std,
            )

    return None


def _check_material(
    material: Optional[str],
    rules: dict,
) -> tuple[Optional[ValidationIssue], Optional[str]]:
    """
    Проверяет марку материала по allowed/forbidden.

    Returns:
        (issue, missing_reason) — один из них None.
    """
    mat_rules = rules.get("materials", {})
    if not mat_rules:
        return None, None

    if not material:
        ref = mat_rules.get("tz_reference", "")
        return None, f"Марка материала не указана на чертеже (требование: {ref})"

    allowed = [str(v).strip() for v in mat_rules.get("allowed", [])]
    forbidden = [str(v).strip() for v in mat_rules.get("forbidden", [])]
    std = mat_rules.get("standard", "")
    ref = mat_rules.get("tz_reference", "")

    mat_stripped = material.strip()

    if forbidden and mat_stripped in forbidden:
        return ValidationIssue(
            field="material",
            actual_value=material,
            expected=f"одна из: {', '.join(allowed)}",
            message=f"Материал '{material}' входит в список запрещённых",
            tz_reference=f"{ref} (стандарт: {std})" if ref else std,
        ), None

    if allowed and mat_stripped not in allowed:
        return ValidationIssue(
            field="material",
            actual_value=material,
            expected=f"одна из: {', '.join(allowed)}",
            message=(
                f"Материал '{material}' отсутствует в списке допустимых: "
                f"[{', '.join(allowed)}]"
            ),
            tz_reference=f"{ref} (стандарт: {std})" if ref else std,
        ), None

    return None, None


def _check_designation(
    designation: Optional[str],
    rules: dict,
) -> tuple[Optional[ValidationIssue], Optional[str]]:
    """Проверяет обозначение по regex-паттерну."""
    des_rules = rules.get("designations", {})
    if not des_rules or not des_rules.get("pattern"):
        return None, None

    if not designation:
        ref = des_rules.get("tz_reference", "")
        return None, f"Конструкторское обозначение не указано на чертеже (требование: {ref})"

    pattern = des_rules["pattern"]
    ref = des_rules.get("tz_reference", "")
    examples = des_rules.get("examples", [])

    if not re.match(pattern, designation.strip()):
        return ValidationIssue(
            field="designation",
            actual_value=designation,
            expected=f"паттерн: {pattern}, примеры: {', '.join(examples)}",
            message=(
                f"Обозначение '{designation}' не соответствует паттерну '{pattern}'"
            ),
            tz_reference=ref,
        ), None

    return None, None


def _check_coating(
    note: Optional[str],
    rules: dict,
) -> Optional[ValidationIssue]:
    """Проверяет код покрытия если он упомянут в примечании."""
    coating_rules = rules.get("coatings", {})
    if not coating_rules or not note:
        return None

    allowed_codes = [str(c).strip() for c in coating_rules.get("allowed_codes", [])]
    std = coating_rules.get("standard", "")
    ref = coating_rules.get("tz_reference", "")

    for code in allowed_codes:
        if code in note:
            return None  # код найден и допустим

    # Ищем что-то похожее на код покрытия в примечании
    coating_match = re.search(r"\b(Хим\.\w+|Ц\d+|Н\d+|Кд\d+|Ан\.\w+)\b", note)
    if coating_match:
        found_code = coating_match.group()
        if found_code not in allowed_codes:
            return ValidationIssue(
                field="coating",
                actual_value=found_code,
                expected=f"одно из: {', '.join(allowed_codes)}",
                message=(
                    f"Код покрытия '{found_code}' отсутствует в списке допустимых"
                ),
                tz_reference=f"{ref} (стандарт: {std})" if ref else std,
            )

    return None


def _append_missing(report: DrawingReport, message: str) -> None:
    if message not in report.compliance.missing_info:
        report.compliance.missing_info.append(message)


def validate_report(
    report: DrawingReport,
    rules: dict,
    heuristic_fallback_used: bool = False,
) -> DrawingReport:
    """
    Детерминированно проверяет элементы DrawingReport по rules.json.

    Добавляет ValidationIssue и missing_info в report.compliance.
    Пересчитывает is_compliant на основе найденных расхождений.

    Args:
        report: Отчёт от LLM (может быть частично заполнен).
        rules: Словарь правил из rules.json.
        heuristic_fallback_used: True если элементы получены эвристикой без LLM.

    Returns:
        Обновлённый DrawingReport.
    """
    if not rules:
        _append_missing(
            report,
            "rules.json пустой — детерминированная проверка пропущена",
        )
        return report

    new_issues: list[ValidationIssue] = list(report.compliance.issues)
    new_missing: list[str] = list(report.compliance.missing_info)

    dimension_rules_by_field: dict[str, dict] = {
        r["field"]: r for r in rules.get("dimensions", [])
    }

    for element in report.elements:
        el_label = element.item_id or element.name or "неизвестный элемент"

        # --- Проверка размера (size) ---
        # Пробуем сопоставить с bolt_size если это резьба
        if element.size:
            bolt_rule = dimension_rules_by_field.get("bolt_size")
            if bolt_rule and re.match(r"M\d+", element.size or ""):
                issue = _check_dimension(element.size, bolt_rule, f"bolt_size ({el_label})")
                if issue:
                    new_issues.append(issue)
            else:
                # Числовой размер — проверяем все подходящие правила
                for field_name, rule in dimension_rules_by_field.items():
                    if field_name == "bolt_size":
                        continue
                    numeric = _parse_numeric(element.size)
                    if numeric is not None and ("min" in rule or "max" in rule):
                        issue = _check_dimension(element.size, rule, f"{field_name} ({el_label})")
                        if issue:
                            new_issues.append(issue)
                            break  # одно расхождение за раз на элемент

        # --- Проверка материала ---
        mat_issue, mat_missing = _check_material(element.material, rules)
        if mat_issue:
            new_issues.append(mat_issue)
        if mat_missing:
            new_missing.append(f"{el_label}: {mat_missing}")

        # --- Проверка обозначения ---
        des_issue, des_missing = _check_designation(element.designation, rules)
        if des_issue:
            new_issues.append(des_issue)
        if des_missing:
            new_missing.append(f"{el_label}: {des_missing}")

        # --- Проверка покрытия (из примечания) ---
        coat_issue = _check_coating(element.note, rules)
        if coat_issue:
            new_issues.append(coat_issue)

    report.compliance.issues = new_issues
    report.compliance.missing_info = new_missing

    # Пересчитываем is_compliant
    if new_issues:
        report.compliance.is_compliant = False
    elif not report.elements:
        report.compliance.is_compliant = None
        if heuristic_fallback_used:
            _append_missing(
                report,
                "В тексте страницы не найдены параметры по rules.json — сравнить с ТЗ нельзя",
            )
        else:
            _append_missing(
                report,
                "Элементы чертежа не извлечены — статус соответствия не определён",
            )
    else:
        if heuristic_fallback_used:
            if report.compliance.is_compliant is None:
                report.compliance.is_compliant = True
            _append_missing(
                report,
                "Проверка без текстовой LLM: эвристика + rules.json; возможны пропуски",
            )
        else:
            if report.compliance.is_compliant is None:
                _append_missing(
                    report,
                    "LLM не дал однозначного вывода — детерминированных расхождений не обнаружено",
                )
            # Нет расхождений по rules.json → считаем соответствие подтверждённым на этом слое
            report.compliance.is_compliant = True

    # Пересчитываем общую уверенность
    if new_issues:
        report.overall_confidence = ConfidenceLevel.MEDIUM
    elif new_missing:
        report.overall_confidence = ConfidenceLevel.MEDIUM

    logger.info(
        "rule_validator: %d расхождений, %d не определено",
        len(new_issues),
        len(new_missing),
    )
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Детерминированная проверка отчёта по rules.json")
    parser.add_argument("--report", required=True, help="Путь к JSON-файлу DrawingReport")
    parser.add_argument("--rules", default="config/rules.json")
    args = parser.parse_args()

    report_data = json.loads(Path(args.report).read_text(encoding="utf-8"))
    report = DrawingReport(**report_data)
    rules = load_rules(args.rules)

    validated = validate_report(report, rules)

    print(f"\n{validated.summary()}")
    if validated.compliance.issues:
        print("\nРасхождения:")
        for issue in validated.compliance.issues:
            print(f"  [{issue.field}] {issue.message}")
            if issue.tz_reference:
                print(f"    Ссылка: {issue.tz_reference}")
    if validated.compliance.missing_info:
        print("\nНельзя проверить:")
        for m in validated.compliance.missing_info:
            print(f"  - {m}")


if __name__ == "__main__":
    main()
