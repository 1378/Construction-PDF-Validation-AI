"""
Классификация страниц PDF (чертёж, экспликация, аннотации, реклама/фото).

Основная точка входа — ``classify_page``. Для юнит-тестов без файла PDF
используйте ``classify_from_features`` с заранее подготовленным текстом и метриками.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    import fitz

logger = logging.getLogger(__name__)

# --- Пороги (можно подменять в тестах через monkeypatch модуля) ---
CONFIDENCE_THRESHOLD = 0.52
MARGIN_SECOND_BEST = 0.12
LOW_TEXT_CHAR_THRESHOLD = 220
HIGH_IMAGE_AREA_RATIO = 0.38
MIN_IMAGES_FOR_PHOTO_PAGE = 2

# Сильные маркеры (дают высокий вес уверенности)
_DRAWING_STRONG = (
    r"черт[её]ж\s*№",
    r"черт[её]ж\s+",
    r"рабочий\s+черт[её]ж",
    r"план\s+(этажа|помещен|кровли|фасада|осей)",
    r"схема\s+(этажа|расположения|осей|коллектор)",
    r"план\s*[-–—]\s*",
)
_DRAWING_WEAKER = (
    r"\bплан\b",
    r"\bсхема\b",
    r"\bчерт[её]ж\b",
    r"\bразрез\b",
    r"\bфасад\b",
)

_EXPLICATION_STRONG = (
    r"экспликац(?:ия|ии)\s+помещен",
    r"таблица\s+экспликац",
)
_EXPLICATION_WEAKER = (
    r"\bэкспликац",
    r"площад[ьи]\s+помещен",
    r"наименовани[ея]\s+помещен",
)

_ANNOTATION_MARKERS = (
    r"примечани",
    r"условн(?:ые|ых)\s+обозначен",
    r"масштаб\s*1\s*[:]",
    r"\bразмеры\b",
    r"\bлегенда\b",
    r"сокращени[ея]",
)

# Титул, шифр, задание, состав ПД — держим рядом с чертежами одного дома
_PROJECT_INFO_STRONG = (
    r"заказчик",
    r"застройщик",
    r"объект\s+капитального\s+строительства",
    r"наименовани[ея]\s+объекта",
    r"площадк[аи]\s+строительства",
    r"сведени[яь]\s+о\s+застройщике",
    r"состав\s+проектной\s+документации",
    r"задани[ея]\s+на\s+проектирование",
    r"шифр\s+проекта",
    r"гип[\s\.]",
    r"главный\s+инженер\s+проекта",
    r"технико-экономическ(?:ие|их)\s+показател",
)
_PROJECT_INFO_WEAKER = (
    r"проектная\s+документаци",
    r"пояснительн\w*\s+записк",
    r"разрешени[ея]\s+на\s+строительство",
)
# Типичные размерные цепочки на чертежах / выносках
_DIM_LIKE = re.compile(
    r"\d{2,4}\s*[xх×]\s*\d{2,4}|\d+\s*[=≈]\s*\d+",
    re.IGNORECASE,
)

# Явные маркеры рекламы / «мусорных» страниц (сохраняются при низкой уверенности)
_EXPLICIT_AD_PHRASES = (
    "реклама",
    "фото здания",
    "фотография здания",
    "сайт:",
    "www.",
    "http://",
    "https://",
)
_EXPLICIT_AD_AUTHOR = re.compile(r"\bАВТОР\b", re.UNICODE)

# Дополнительные рекламные слова (повышают score, не обязательно «явные»)
_SOFT_AD_WORDS = (
    "скидк",
    "акци",
    "купить",
    "заказать",
    "телефон:",
    "объявлен",
)


class PageClassification(TypedDict):
    """Результат классификации одной страницы."""

    is_drawing: bool
    is_explication: bool
    is_annotation: bool
    is_project_info: bool
    is_irrelevant_photo_or_ad: bool


@dataclass(frozen=True)
class PageFeatures:
    """Признаки страницы, вычисленные из текста и геометрии (удобно подставлять в тестах)."""

    text: str
    text_upper: str
    char_count: int
    image_area_ratio: float
    image_count: int


def _normalize_text(raw: str) -> str:
    t = raw.lower().replace("\xa0", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _compile(patterns: tuple[str, ...]) -> list[re.Pattern[str]]:
    return [re.compile(p, re.IGNORECASE | re.UNICODE) for p in patterns]


_DRAWING_STRONG_RE = _compile(_DRAWING_STRONG)
_DRAWING_WEAK_RE = _compile(_DRAWING_WEAKER)
_EXPL_STRONG_RE = _compile(_EXPLICATION_STRONG)
_EXPL_WEAK_RE = _compile(_EXPLICATION_WEAKER)
_ANNOT_RE = _compile(_ANNOTATION_MARKERS)
_PROJECT_STRONG_RE = _compile(_PROJECT_INFO_STRONG)
_PROJECT_WEAK_RE = _compile(_PROJECT_INFO_WEAKER)


def _drawing_confidence(text: str) -> float:
    score = 0.0
    for p in _DRAWING_STRONG_RE:
        if p.search(text):
            score = max(score, 0.88)
    for p in _DRAWING_WEAK_RE:
        if p.search(text):
            score = max(score, 0.42)
    return min(1.0, score)


def _explication_confidence(text: str) -> float:
    score = 0.0
    for p in _EXPL_STRONG_RE:
        if p.search(text):
            score = max(score, 0.9)
    for p in _EXPL_WEAK_RE:
        if p.search(text):
            score = max(score, 0.48)
    return min(1.0, score)


def _project_info_confidence(text: str) -> float:
    score = 0.0
    for p in _PROJECT_STRONG_RE:
        if p.search(text):
            score = max(score, 0.82)
    for p in _PROJECT_WEAK_RE:
        if p.search(text):
            score = max(score, 0.44)
    return min(1.0, score)


def _annotation_confidence(text: str, has_drawing_hint: bool) -> float:
    if has_drawing_hint:
        return 0.0
    score = 0.0
    hits = sum(1 for p in _ANNOT_RE if p.search(text))
    if hits:
        score = 0.35 + min(0.45, 0.12 * hits)
    if _DIM_LIKE.search(text):
        score = max(score, 0.4)
    return min(1.0, score)


def _has_explicit_ad_markers(text_norm: str, text_upper: str) -> bool:
    for ph in _EXPLICIT_AD_PHRASES:
        if ph in text_norm:
            return True
    if _EXPLICIT_AD_AUTHOR.search(text_upper):
        return True
    return False


def _irrelevant_photo_ad_confidence(f: PageFeatures, text_norm: str) -> float:
    score = 0.0
    for w in _SOFT_AD_WORDS:
        if w in text_norm:
            score = max(score, 0.35)

    low_text = f.char_count < LOW_TEXT_CHAR_THRESHOLD
    busy_images = f.image_area_ratio >= HIGH_IMAGE_AREA_RATIO or f.image_count >= MIN_IMAGES_FOR_PHOTO_PAGE

    if low_text and busy_images:
        score = max(score, 0.55)
    if low_text and f.image_count >= 1 and f.char_count < 80:
        score = max(score, 0.45)

    if _has_explicit_ad_markers(text_norm, f.text_upper):
        score = max(score, 0.95)

    return min(1.0, score)


def _image_metrics(page: fitz.Page) -> tuple[float, int]:
    """Доля площади страницы, покрытая картинками, и число уникальных xref."""
    rect = page.rect
    page_area = float(rect.width * rect.height)
    if page_area <= 0:
        return 0.0, 0

    total_img_area = 0.0
    xrefs: set[int] = set()
    for info in page.get_images(full=True):
        xref = int(info[0])
        if xref in xrefs:
            continue
        try:
            rlist = page.get_image_rects(xref)
        except Exception:
            rlist = []
        if not rlist:
            continue
        xrefs.add(xref)
        for r in rlist:
            inter = r & rect
            if inter.is_empty:
                continue
            total_img_area += float(inter.get_area())

    ratio = min(1.0, total_img_area / page_area)
    return ratio, len(xrefs)


def _collapse_preserve_case(raw: str) -> str:
    t = raw.replace("\xa0", " ")
    return re.sub(r"\s+", " ", t).strip()


def page_begins_new_project(features: PageFeatures) -> bool:
    """
    Эвристика начала **другого** дома/проекта в общем альбоме (только текст).

    Используется нарезчиком: перед такой страницей обрывается «хвост» предыдущего проекта,
    чтобы не склеивать чертежи разных объектов.
    """
    t = features.text[:1800]
    if not t.strip():
        return False
    if re.search(r"лист\s*1\s*из", t, re.IGNORECASE) and (
        "пояснительн" in t
        or "проектная документаци" in t
        or "наименование объекта" in t
        or "задание на проектирование" in t
        or "состав проектной документации" in t
    ):
        return True
    if "пояснительная записка к проекту" in t:
        return True
    if "пояснительная записка к архитектурно" in t:
        return True
    return False


def extract_features(page: fitz.Page) -> PageFeatures:
    """Собирает текст и метрики из ``fitz.Page`` (удобно мокать в тестах)."""
    raw = page.get_text("text") or ""
    collapsed = _collapse_preserve_case(raw)
    norm = _normalize_text(raw)
    ratio, nimg = _image_metrics(page)
    return PageFeatures(
        text=norm,
        text_upper=collapsed.upper(),
        char_count=len(norm),
        image_area_ratio=ratio,
        image_count=nimg,
    )


def classify_from_features(features: PageFeatures) -> PageClassification:
    """
    Классифицирует страницу по уже извлечённым признакам.

    При низкой уверенности обнуляет ``is_drawing``, ``is_explication``, ``is_annotation``,
    но может выставить ``is_project_info`` по текстовым маркерам титула/ПД;
    ``is_irrelevant_photo_or_ad=True`` — только при явных рекламных маркерах.
    """
    text = features.text

    c_draw = _drawing_confidence(text)
    c_expl = _explication_confidence(text)
    c_proj = _project_info_confidence(text)
    c_ad = _irrelevant_photo_ad_confidence(features, text)

    has_draw_hint = c_draw >= 0.42
    c_annot = _annotation_confidence(text, has_drawing_hint=has_draw_hint)

    scores = {
        "drawing": c_draw,
        "explication": c_expl,
        "annotation": c_annot,
        "irrelevant": c_ad,
    }
    sorted_vals = sorted(scores.values(), reverse=True)
    best = sorted_vals[0] if sorted_vals else 0.0
    second = sorted_vals[1] if len(sorted_vals) > 1 else 0.0
    confident = best >= CONFIDENCE_THRESHOLD and (best - second) >= MARGIN_SECOND_BEST

    explicit_ad = _has_explicit_ad_markers(text, features.text_upper)

    if not confident:
        logger.info(
            "pdf_analyzer: низкая уверенность классификации (scores=%s, threshold=%s, margin=%s)",
            scores,
            CONFIDENCE_THRESHOLD,
            MARGIN_SECOND_BEST,
        )
        is_proj = bool(c_proj >= 0.5 and not explicit_ad)
        return PageClassification(
            is_drawing=False,
            is_explication=False,
            is_annotation=False,
            is_project_info=is_proj,
            is_irrelevant_photo_or_ad=explicit_ad,
        )

    # Победитель среди содержательных классов; реклама может сосуществовать только если явно доминирует
    labels = [
        ("drawing", c_draw),
        ("explication", c_expl),
        ("annotation", c_annot),
        ("irrelevant", c_ad),
    ]
    labels.sort(key=lambda x: x[1], reverse=True)
    winner, wscore = labels[0]

    irrelevant = winner == "irrelevant" and wscore >= CONFIDENCE_THRESHOLD

    is_drawing = winner == "drawing" and not irrelevant
    is_explication = winner == "explication" and not irrelevant
    is_annotation = winner == "annotation" and not irrelevant
    is_project_info = (
        not irrelevant
        and not is_drawing
        and not is_explication
        and c_proj >= 0.48
        and (is_annotation or c_proj >= 0.55)
    )

    # Явная реклама перебивает «сомнительный» чертёж по тексту
    if explicit_ad and c_ad >= 0.5:
        is_drawing = is_explication = is_annotation = False
        is_project_info = False
        irrelevant = True

    return PageClassification(
        is_drawing=is_drawing,
        is_explication=is_explication,
        is_annotation=is_annotation,
        is_project_info=is_project_info,
        is_irrelevant_photo_or_ad=irrelevant or (explicit_ad and c_ad > c_draw and c_ad > c_expl),
    )


def classify_page(page: fitz.Page) -> PageClassification:
    """
    Возвращает словарь с признаками страницы::

        {
            "is_drawing": bool,  # страница относится к чертежу
            "is_explication": bool,  # экспликация помещений
            "is_annotation": bool,  # подписи/примечания к чертежу (без заголовка чертежа)
            "is_project_info": bool,  # титул, шифр, заказчик, состав ПД и т.п.
            "is_irrelevant_photo_or_ad": bool,  # фото/реклама, не относящаяся к чертежу
        }

    При низкой уверенности в чертеж/экспликацию/примечания флаги сбрасываются, но
    ``is_project_info`` может остаться по маркерам ПД. Реклама — только явные фразы
    в ``_EXPLICIT_AD_PHRASES`` / ``_EXPLICIT_AD_AUTHOR``.
    """
    return classify_from_features(extract_features(page))


__all__ = [
    "PageClassification",
    "PageFeatures",
    "CONFIDENCE_THRESHOLD",
    "MARGIN_SECOND_BEST",
    "classify_page",
    "classify_from_features",
    "extract_features",
    "page_begins_new_project",
]
