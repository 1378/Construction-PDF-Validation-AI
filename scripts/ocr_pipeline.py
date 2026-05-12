"""
ocr_pipeline.py — OCR-распознавание текста на изображениях чертежей.

Основной движок: pytesseract (Tesseract).
Fallback: easyocr — используется если Tesseract даёт < MIN_MEANINGFUL_CHARS
значимых символов (не пробелы, не мусор).

Пример запуска:
    python -m scripts.ocr_pipeline --image data/images/123-А_page_001.png
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

from scripts.models import ConfidenceLevel, OcrResult

logger = logging.getLogger(__name__)

# Порог значимых символов — ниже него переключаемся на easyocr
MIN_MEANINGFUL_CHARS = 50
# Языки по умолчанию для Tesseract
TESSERACT_LANG = "rus+eng"


def _count_meaningful_chars(text: str) -> int:
    """Считает буквы и цифры (кириллица, латиница, числа) в строке."""
    return len(re.findall(r"[а-яёА-ЯЁa-zA-Z0-9]", text))


def _assess_confidence(text: str, char_count: int) -> ConfidenceLevel:
    """Оценивает качество OCR по количеству значимых символов."""
    if char_count >= 200:
        return ConfidenceLevel.HIGH
    if char_count >= MIN_MEANINGFUL_CHARS:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


def run_tesseract(image_path: str, lang: str = TESSERACT_LANG) -> tuple[str, bool]:
    """
    Запускает Tesseract OCR на изображении.

    Returns:
        (text, success) — распознанный текст и флаг успеха.
    """
    try:
        import pytesseract
        from PIL import Image

        img = Image.open(image_path)
        # Конфигурация: сохраняем порядок строк, ориентируемся на печатный текст
        custom_config = r"--oem 3 --psm 6"
        text = pytesseract.image_to_string(img, lang=lang, config=custom_config)
        return text.strip(), True
    except ImportError:
        logger.warning("pytesseract не установлен")
        return "", False
    except Exception as e:
        logger.warning("Tesseract ошибка для %s: %s", image_path, e)
        return "", False


def run_easyocr(image_path: str) -> tuple[str, bool]:
    """
    Запускает EasyOCR на изображении (поддерживает русский и английский).

    Returns:
        (text, success) — распознанный текст и флаг успеха.
    """
    try:
        import easyocr

        reader = easyocr.Reader(["ru", "en"], gpu=False, verbose=False)
        results = reader.readtext(image_path, detail=0, paragraph=True)
        text = "\n".join(results)
        return text.strip(), True
    except ImportError:
        logger.warning("easyocr не установлен")
        return "", False
    except Exception as e:
        logger.warning("EasyOCR ошибка для %s: %s", image_path, e)
        return "", False


def run_ocr(
    image_path: str,
    lang: str = TESSERACT_LANG,
    force_easyocr: bool = False,
) -> OcrResult:
    """
    Выполняет OCR на изображении с автоматическим выбором движка.

    Алгоритм:
    1. Пробуем Tesseract.
    2. Если результат < MIN_MEANINGFUL_CHARS символов — пробуем EasyOCR.
    3. Берём лучший результат (больше значимых символов).

    Args:
        image_path: Путь к PNG/JPG изображению.
        lang: Языки для Tesseract (напр. "rus+eng").
        force_easyocr: Если True — сразу используем EasyOCR без Tesseract.

    Returns:
        OcrResult с текстом, движком и уровнем уверенности.
    """
    if not Path(image_path).exists():
        logger.error("Изображение не найдено: %s", image_path)
        return OcrResult(
            image_path=image_path,
            text="",
            engine="none",
            confidence=ConfidenceLevel.LOW,
            char_count=0,
        )

    best_text = ""
    best_engine = "none"
    best_count = 0

    if not force_easyocr:
        tess_text, tess_ok = run_tesseract(image_path, lang)
        if tess_ok:
            count = _count_meaningful_chars(tess_text)
            logger.debug("Tesseract: %d значимых символов", count)
            if count > best_count:
                best_text = tess_text
                best_engine = "tesseract"
                best_count = count

    # Fallback или принудительный запуск EasyOCR
    if force_easyocr or best_count < MIN_MEANINGFUL_CHARS:
        easy_text, easy_ok = run_easyocr(image_path)
        if easy_ok:
            count = _count_meaningful_chars(easy_text)
            logger.debug("EasyOCR: %d значимых символов", count)
            if count > best_count:
                best_text = easy_text
                best_engine = "easyocr"
                best_count = count

    confidence = _assess_confidence(best_text, best_count)

    if best_count < MIN_MEANINGFUL_CHARS:
        logger.warning(
            "OCR дал мало символов (%d) для %s — confidence=low", best_count, image_path
        )

    return OcrResult(
        image_path=image_path,
        text=best_text,
        engine=best_engine,
        confidence=confidence,
        char_count=best_count,
    )


def run_ocr_batch(
    image_paths: list[str],
    lang: str = TESSERACT_LANG,
) -> list[OcrResult]:
    """Выполняет OCR для списка изображений."""
    results = []
    for path in image_paths:
        logger.info("OCR: %s", path)
        result = run_ocr(path, lang=lang)
        results.append(result)
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="OCR-распознавание изображения чертежа")
    parser.add_argument("--image", required=True, help="Путь к PNG/JPG изображению")
    parser.add_argument("--lang", default=TESSERACT_LANG, help="Языки Tesseract (напр. rus+eng)")
    parser.add_argument("--easyocr", action="store_true", help="Принудительно использовать EasyOCR")
    args = parser.parse_args()

    result = run_ocr(args.image, lang=args.lang, force_easyocr=args.easyocr)
    print(f"\nРезультат OCR:")
    print(f"  Движок: {result.engine}")
    print(f"  Уверенность: {result.confidence.value}")
    print(f"  Значимых символов: {result.char_count}")
    print(f"\n--- Распознанный текст ---\n{result.text}\n---")


if __name__ == "__main__":
    main()
