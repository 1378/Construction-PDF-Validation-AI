"""
Pydantic-модели данных для системы анализа PDF-чертежей.

Все поля необязательны по умолчанию — LLM не должен выдумывать значения.
При нехватке данных поля остаются None, а причина записывается в missing_info.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


def _normalize_llm_string_list(v: Any, prefer_keys: tuple[str, ...]) -> list[str]:
    """
    Приводит ответ LLM к list[str]: строки как есть, из dict — по prefer_keys, иначе str(item).
    """
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    if not isinstance(v, list):
        return [str(v)]
    out: list[str] = []
    for item in v:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            piece: str | None = None
            for key in prefer_keys:
                val = item.get(key)
                if val is not None and val != "":
                    piece = str(val)
                    break
            out.append(piece if piece is not None else str(item))
        else:
            out.append(str(item))
    return out


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SourceType(str, Enum):
    PDF_TEXT = "pdf_text"
    OCR = "ocr"
    LLAVA_CAPTION = "llava_caption"
    COMBINED = "combined"


class DrawingElement(BaseModel):
    """Отдельный элемент чертежа: деталь, позиция, обозначение и т.п."""

    item_id: Optional[str] = Field(None, description="Позиционный номер или код (напр. Поз01)")
    name: Optional[str] = Field(None, description="Наименование элемента (напр. Штуцер)")
    element_type: Optional[str] = Field(None, description="Тип элемента (напр. Металлический, Сварной)")
    size: Optional[str] = Field(None, description="Размер или обозначение резьбы (напр. M16x1.5)")
    material: Optional[str] = Field(None, description="Материал (напр. Ст3сп, 09Г2С)")
    designation: Optional[str] = Field(None, description="Конструкторское обозначение по стандарту")
    note: Optional[str] = Field(None, description="Примечание или дополнительная информация")
    source: SourceType = Field(SourceType.PDF_TEXT, description="Откуда извлечён элемент")
    confidence: ConfidenceLevel = Field(ConfidenceLevel.MEDIUM, description="Уверенность в корректности данных")
    raw_text_fragment: Optional[str] = Field(None, description="Исходный фрагмент текста, из которого извлечён элемент")


class ValidationIssue(BaseModel):
    """Конкретное расхождение с ТЗ."""

    field: str = Field(..., description="Проверяемое поле (напр. size, material)")
    actual_value: Optional[str] = Field(None, description="Значение из чертежа")
    expected: Optional[str] = Field(None, description="Ожидаемое значение или диапазон по ТЗ")
    message: str = Field(..., description="Человекочитаемое описание расхождения")
    tz_reference: Optional[str] = Field(None, description="Ссылка на пункт ТЗ или ГОСТ")


class ComplianceReport(BaseModel):
    """Отчёт о соответствии чертежа ТЗ."""

    standard: Optional[str] = Field(None, description="Применённый стандарт или пункт ТЗ")
    is_compliant: Optional[bool] = Field(
        None,
        description="True — соответствует, False — не соответствует, None — нельзя определить",
    )
    issues: list[ValidationIssue] = Field(default_factory=list, description="Список расхождений с ТЗ")
    missing_info: list[str] = Field(
        default_factory=list,
        description="Что невозможно проверить из-за отсутствия данных (на чертеже или в ТЗ)",
    )
    citations: list[str] = Field(
        default_factory=list,
        description="Цитаты из ТЗ, на основании которых сделан вывод",
    )

    @field_validator("missing_info", mode="before")
    @classmethod
    def _normalize_missing_info(cls, v: Any) -> list[str]:
        return _normalize_llm_string_list(v, prefer_keys=("field", "text"))

    @field_validator("citations", mode="before")
    @classmethod
    def _normalize_citations(cls, v: Any) -> list[str]:
        return _normalize_llm_string_list(v, prefer_keys=("text", "field"))


class PageExtraction(BaseModel):
    """Результат извлечения данных с одной страницы PDF."""

    page_number: int
    pdf_text: str = Field(default="", description="Текст, извлечённый PyMuPDF")
    image_path: Optional[str] = Field(None, description="Путь к PNG изображению страницы")
    has_drawings: bool = Field(False, description="Есть ли на странице графика/чертёж")


class OcrResult(BaseModel):
    """Результат OCR-распознавания изображения."""

    image_path: str
    text: str = Field(default="", description="Распознанный текст")
    engine: str = Field("tesseract", description="Движок: tesseract | easyocr")
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    char_count: int = Field(0, description="Количество значимых символов в тексте")


class ExtractionResult(BaseModel):
    """Полный результат извлечения данных из PDF."""

    pdf_path: str
    drawing_id: str
    total_pages: int
    pages: list[PageExtraction] = Field(default_factory=list)


class DrawingReport(BaseModel):
    """Финальный отчёт по анализу чертежа — основной выходной формат системы."""

    drawing_id: str = Field(..., description="Идентификатор чертежа (напр. 123-А)")
    pdf_path: str
    page_number: Optional[int] = Field(
        None,
        description="Номер страницы (1-based); None — весь документ как один чертёж",
    )
    total_pages: int = Field(1)
    llm_used: bool = Field(
        True,
        description="False — проверка без текстовой LLM (эвристика + rules.json)",
    )
    elements: list[DrawingElement] = Field(default_factory=list, description="Извлечённые элементы чертежа")
    compliance: ComplianceReport = Field(default_factory=ComplianceReport)
    raw_ocr_text: Optional[str] = Field(None, description="Сырой текст от OCR (для отладки)")
    llava_caption: Optional[str] = Field(None, description="Описание чертежа от LLaVA")
    tz_chunks_used: list[str] = Field(
        default_factory=list,
        description="Чанки ТЗ, переданные в LLM (для аудита)",
    )
    overall_confidence: ConfidenceLevel = Field(
        ConfidenceLevel.MEDIUM,
        description="Общая уверенность в результате анализа",
    )

    def summary(self) -> str:
        """Короткая сводка результата для вывода в консоль."""
        status = (
            "СООТВЕТСТВУЕТ"
            if self.compliance.is_compliant is True
            else "НЕ СООТВЕТСТВУЕТ"
            if self.compliance.is_compliant is False
            else "НЕЛЬЗЯ ОПРЕДЕЛИТЬ"
        )
        issues_count = len(self.compliance.issues)
        missing_count = len(self.compliance.missing_info)
        elements_count = len(self.elements)
        return (
            f"Чертёж {self.drawing_id}: {status} | "
            f"Элементов: {elements_count} | "
            f"Расхождений: {issues_count} | "
            f"Не определено: {missing_count}"
        )


class PdfBatchReport(BaseModel):
    """Сводный отчёт по PDF с несколькими чертежами (страницами)."""

    pdf_path: str
    base_drawing_id: str
    total_drawings: int
    ollama_reachable: bool = False
    text_llm_model: str = ""
    text_llm_available: bool = False
    drawings: list[DrawingReport] = Field(default_factory=list)
    compliant: list[str] = Field(
        default_factory=list,
        description="drawing_id страниц, прошедших проверку (is_compliant=True)",
    )
    non_compliant: list[str] = Field(
        default_factory=list,
        description="drawing_id с расхождениями по ТЗ",
    )
    undetermined: list[str] = Field(
        default_factory=list,
        description="нельзя однозначно оценить соответствие",
    )
