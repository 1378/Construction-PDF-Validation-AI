"""
main.py — точка входа. Анализирует PDF-чертеж(и) и выдаёт отчёт о соответствии ТЗ.

Режимы:
  • По умолчанию: весь PDF как один чертёж (все страницы объединяются в один текст).
  • --all-pages: каждая страница — отдельный чертёж; в JSON — сводка: подходят / не подходят / не определено.

Запуск:
    python main.py --pdf ./data/pdf/drawing_123.pdf --drawing-id 123-А
    python main.py --pdf ./data/pdf/album.pdf --drawing-id КД-100 --all-pages
    python main.py --pipeline-album --pdf ./data/pdf/album.pdf --projects-dir ./data/projects_run

Опции см. в argparse ниже.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from scripts.heuristic_extract import extract_elements_from_text
from scripts.llm_checker import check_drawing
from scripts.models import ComplianceReport, ConfidenceLevel, DrawingReport, PdfBatchReport
from scripts.multimodal_caption import caption_drawing, is_ollama_available
from scripts.ocr_pipeline import run_ocr
from scripts.ollama_util import (
    diagnose_ollama,
    get_ollama_generate_url,
    is_ollama_reachable,
    ollama_model_loaded,
)
from scripts.pdf_extractor import extract_pdf
from scripts.rule_validator import load_rules, validate_report

logger = logging.getLogger(__name__)

DEFAULT_IMAGES_DIR = "data/images"
DEFAULT_TEXT_DIR = "data/extracted_text"
DEFAULT_INDEX_DIR = "data/faiss_index"
DEFAULT_RULES_PATH = "config/rules.json"
DEFAULT_SPEC_PATH = "docs/tech_spec.md"
DEFAULT_PROJECTS_PIPELINE_DIR = "data/projects_pipeline"


def _ensure_tz_index(index_dir: str, spec_path: str, rules_path: str) -> None:
    from scripts.index_rag import build_tz_index

    tz_index = Path(index_dir) / "tz_index.faiss"
    if not tz_index.exists():
        logger.info("ТЗ-индекс не найден — строим...")
        build_tz_index(spec_path=spec_path, rules_path=rules_path, index_dir=index_dir)


def _rebuild_tz_index(index_dir: str, spec_path: str, rules_path: str) -> None:
    from scripts.index_rag import build_tz_index

    logger.info("Перестройка ТЗ-индекса...")
    build_tz_index(spec_path=spec_path, rules_path=rules_path, index_dir=index_dir)


def _text_llm_ready(model: str) -> bool:
    """Ollama запущена и в списке есть текстовая модель."""
    gen = get_ollama_generate_url()
    if not is_ollama_reachable(gen):
        return False
    return ollama_model_loaded(model, gen)


def _finalize_report(
    report: DrawingReport,
    full_text: str,
    rules: dict,
    *,
    pdf_path: str,
    total_pages: int,
    page_number: int | None,
    raw_ocr: str | None,
    llava_caption: str | None,
) -> DrawingReport:
    """Эвристика при пустых элементах + rule_validator."""
    elements_llm = len(report.elements)
    if not full_text.strip():
        report.pdf_path = str(Path(pdf_path).resolve())
        report.total_pages = total_pages
        report.page_number = page_number
        report.raw_ocr_text = raw_ocr
        report.llava_caption = llava_caption
        return validate_report(report, rules, heuristic_fallback_used=not report.llm_used)

    if elements_llm == 0:
        heur = extract_elements_from_text(full_text, rules)
        if heur:
            report.elements = heur

    heuristic_fb = (not report.llm_used) or (elements_llm == 0 and len(report.elements) > 0)

    report.pdf_path = str(Path(pdf_path).resolve())
    report.total_pages = total_pages
    report.page_number = page_number
    report.raw_ocr_text = raw_ocr
    report.llava_caption = llava_caption

    return validate_report(report, rules, heuristic_fallback_used=heuristic_fb)


def _run_llm_check(
    drawing_id: str,
    full_text: str,
    tz_chunks: list[dict],
    model: str,
    text_llm_ok: bool,
    ollama_generate_url: str | None = None,
) -> DrawingReport:
    gen = ollama_generate_url or get_ollama_generate_url()
    if text_llm_ok:
        return check_drawing(drawing_id, full_text, tz_chunks, model=model, ollama_url=gen)
    return DrawingReport(
        drawing_id=drawing_id,
        pdf_path="",
        llm_used=False,
        tz_chunks_used=[c.get("text", "") for c in tz_chunks],
        compliance=ComplianceReport(
            is_compliant=None,
            missing_info=[
                "Текстовая LLM не вызывалась: Ollama недоступна или модель не установлена "
                f"(ожидалась «{model}»). Будет эвристика + rules.json."
            ],
        ),
        overall_confidence=ConfidenceLevel.LOW,
    )


def _process_one_sheet(
    *,
    drawing_id: str,
    page_number: int | None,
    total_pages: int,
    pdf_path: str,
    full_text: str,
    raw_ocr: str | None,
    llava_caption: str | None,
    model: str,
    top_k: int,
    index_dir: str,
    rules: dict,
    text_llm_ok: bool,
    ollama_generate_url: str | None = None,
) -> DrawingReport:
    from scripts.index_rag import add_drawing_to_index, search_tz

    if full_text.strip():
        add_drawing_to_index(
            drawing_id=drawing_id,
            text=full_text,
            page_number=page_number or 1,
            index_dir=index_dir,
        )
    query = full_text[:800] if full_text.strip() else "техническое задание требования материалы"
    tz_chunks = search_tz(query, top_k=top_k, index_dir=index_dir)

    report = _run_llm_check(
        drawing_id, full_text, tz_chunks, model, text_llm_ok, ollama_generate_url=ollama_generate_url
    )
    report.tz_chunks_used = [c.get("text", "") for c in tz_chunks]

    return _finalize_report(
        report,
        full_text,
        rules,
        pdf_path=pdf_path,
        total_pages=total_pages,
        page_number=page_number,
        raw_ocr=raw_ocr,
        llava_caption=llava_caption,
    )


def analyze_pdf_all_pages(
    pdf_path: str,
    base_drawing_id: str,
    output_path: str | None = None,
    model: str = "mistral",
    vision_model: str = "llava",
    top_k: int = 5,
    skip_ocr: bool = False,
    skip_caption: bool = False,
    images_dir: str = DEFAULT_IMAGES_DIR,
    text_dir: str = DEFAULT_TEXT_DIR,
    index_dir: str = DEFAULT_INDEX_DIR,
    rules_path: str = DEFAULT_RULES_PATH,
    spec_path: str = DEFAULT_SPEC_PATH,
) -> dict:
    """
    Каждая страница PDF — отдельный чертёж с id «{base}-стр.N».
    Возвращает PdfBatchReport как dict со списками compliant / non_compliant / undetermined.
    """
    _ensure_tz_index(index_dir, spec_path, rules_path)
    rules = load_rules(rules_path)
    text_llm_ok = _text_llm_ready(model)
    gen = get_ollama_generate_url()
    ollama_ok = is_ollama_available(gen)

    if not text_llm_ok:
        logger.warning(
            "Текстовая LLM недоступна (запустите Ollama и: ollama pull %s). "
            "Используется эвристика + rules.json. Диагностика: python main.py --check-ollama",
            model,
        )

    logger.info("[batch] Извлечение PDF: %s", pdf_path)
    extraction = extract_pdf(
        pdf_path=pdf_path,
        drawing_id=base_drawing_id,
        images_dir=images_dir,
        text_dir=text_dir,
    )

    drawings_out: list[DrawingReport] = []
    compliant: list[str] = []
    non_compliant: list[str] = []
    undetermined: list[str] = []

    for page in extraction.pages:
        sub_id = f"{base_drawing_id}-стр.{page.page_number}"
        texts: list[str] = []
        ocr_parts: list[str] = []
        cap_parts: list[str] = []

        if page.pdf_text:
            texts.append(f"[PDF стр.{page.page_number}]\n{page.pdf_text}")

        if not skip_ocr and page.image_path:
            logger.info("[batch] OCR %s", sub_id)
            ocr_result = run_ocr(page.image_path)
            if ocr_result.text:
                ocr_parts.append(
                    f"[OCR стр.{page.page_number} ({ocr_result.engine})]\n{ocr_result.text}"
                )

        if not skip_caption and ollama_ok and page.image_path:
            logger.info("[batch] LLaVA %s", sub_id)
            cap = caption_drawing(page.image_path, model=vision_model, ollama_url=gen)
            if cap and not cap.startswith("[ОШИБКА"):
                cap_parts.append(f"[LLaVA стр.{page.page_number}]\n{cap}")

        full_text = "\n\n".join(texts + ocr_parts + cap_parts)
        raw_ocr = "\n\n".join(ocr_parts) if ocr_parts else None
        llava_c = "\n\n".join(cap_parts) if cap_parts else None

        report = _process_one_sheet(
            drawing_id=sub_id,
            page_number=page.page_number,
            total_pages=extraction.total_pages,
            pdf_path=pdf_path,
            full_text=full_text,
            raw_ocr=raw_ocr,
            llava_caption=llava_c,
            model=model,
            top_k=top_k,
            index_dir=index_dir,
            rules=rules,
            text_llm_ok=text_llm_ok,
            ollama_generate_url=gen,
        )
        drawings_out.append(report)

        if report.compliance.is_compliant is True:
            compliant.append(sub_id)
        elif report.compliance.is_compliant is False:
            non_compliant.append(sub_id)
        else:
            undetermined.append(sub_id)

    batch = PdfBatchReport(
        pdf_path=str(Path(pdf_path).resolve()),
        base_drawing_id=base_drawing_id,
        total_drawings=len(drawings_out),
        ollama_reachable=ollama_ok,
        text_llm_model=model,
        text_llm_available=text_llm_ok,
        drawings=drawings_out,
        compliant=compliant,
        non_compliant=non_compliant,
        undetermined=undetermined,
    )
    result = batch.model_dump()

    if output_path is None:
        stem = Path(pdf_path).stem
        output_path = str(Path(pdf_path).parent / f"{stem}_{base_drawing_id}_batch_report.json")

    Path(output_path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Сводный отчёт сохранён: %s", output_path)

    return result


def analyze_drawing(
    pdf_path: str,
    drawing_id: str,
    output_path: str | None = None,
    model: str = "mistral",
    vision_model: str = "llava",
    top_k: int = 5,
    skip_ocr: bool = False,
    skip_caption: bool = False,
    images_dir: str = DEFAULT_IMAGES_DIR,
    text_dir: str = DEFAULT_TEXT_DIR,
    index_dir: str = DEFAULT_INDEX_DIR,
    rules_path: str = DEFAULT_RULES_PATH,
    spec_path: str = DEFAULT_SPEC_PATH,
) -> dict:
    """
    Весь PDF как один чертёж: текст и OCR со всех страниц объединяются.
    """
    _ensure_tz_index(index_dir, spec_path, rules_path)
    rules = load_rules(rules_path)
    text_llm_ok = _text_llm_ready(model)
    gen = get_ollama_generate_url()
    ollama_ok = is_ollama_available(gen)
    if not text_llm_ok:
        logger.warning(
            "Текстовая LLM недоступна (ollama pull %s). Эвристика + rules.json. "
            "Проверка: python main.py --check-ollama",
            model,
        )

    logger.info("Извлечение PDF: %s", pdf_path)
    extraction = extract_pdf(
        pdf_path=pdf_path,
        drawing_id=drawing_id,
        images_dir=images_dir,
        text_dir=text_dir,
    )

    all_texts: list[str] = []
    all_ocr: list[str] = []
    all_cap: list[str] = []

    if not ollama_ok and not skip_caption:
        logger.warning(
            "Ollama недоступна — LLaVA пропущена. ollama pull %s. Проверка: python main.py --check-ollama",
            vision_model,
        )

    for page in extraction.pages:
        if page.pdf_text:
            all_texts.append(f"[PDF стр.{page.page_number}]\n{page.pdf_text}")
        if not skip_ocr and page.image_path:
            ocr_result = run_ocr(page.image_path)
            if ocr_result.text:
                all_ocr.append(
                    f"[OCR стр.{page.page_number} ({ocr_result.engine})]\n{ocr_result.text}"
                )
        if not skip_caption and ollama_ok and page.image_path:
            cap = caption_drawing(page.image_path, model=vision_model, ollama_url=gen)
            if cap and not cap.startswith("[ОШИБКА"):
                all_cap.append(f"[LLaVA стр.{page.page_number}]\n{cap}")

    full_text = "\n\n".join(all_texts + all_ocr + all_cap)
    combined_ocr = "\n\n".join(all_ocr) if all_ocr else None
    combined_cap = "\n\n".join(all_cap) if all_cap else None

    report = _process_one_sheet(
        drawing_id=drawing_id,
        page_number=None,
        total_pages=extraction.total_pages,
        pdf_path=pdf_path,
        full_text=full_text,
        raw_ocr=combined_ocr,
        llava_caption=combined_cap,
        model=model,
        top_k=top_k,
        index_dir=index_dir,
        rules=rules,
        text_llm_ok=text_llm_ok,
        ollama_generate_url=gen,
    )

    result_dict = report.model_dump()

    if output_path is None:
        stem = Path(pdf_path).stem
        output_path = str(Path(pdf_path).parent / f"{stem}_{drawing_id}_report.json")

    Path(output_path).write_text(json.dumps(result_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Отчёт сохранён: %s", output_path)

    return result_dict


def _print_project_pipeline_block(
    report_dict: dict, drawing_id: str, *, summary_line: str = ""
) -> None:
    """Печать результата проверки ТЗ по одному проекту после нарезки."""
    comp = report_dict.get("compliance") or {}
    iso = comp.get("is_compliant")
    status = "OK" if iso is True else "НЕТ" if iso is False else "?"
    print(f"\n{'=' * 60}")
    print(f"Проект: {drawing_id}  [{status}]")
    print(f"{'=' * 60}")
    if summary_line:
        print(summary_line)
    issues = comp.get("issues") or []
    if issues:
        print(f"Расхождения с ТЗ ({len(issues)}):")
        for issue in issues:
            field = issue.get("field", "?")
            msg = issue.get("message", "")
            exp = issue.get("expected")
            ref = issue.get("tz_reference")
            extra = []
            if exp:
                extra.append(f"ожидалось: {exp}")
            if ref:
                extra.append(f"ТЗ: {ref}")
            tail = f" ({'; '.join(extra)})" if extra else ""
            print(f"  • [{field}] {msg}{tail}")
    else:
        print("Расхождений по структурированным критериям нет.")
    missing = comp.get("missing_info") or []
    if missing:
        print("Нельзя проверить / не хватает данных:")
        for m in missing[:12]:
            print(f"  - {m}")
        if len(missing) > 12:
            print(f"  … ещё {len(missing) - 12}")


def _print_pipeline_final_summary(data: dict) -> None:
    print(f"\n{'#' * 60}")
    print("# ИТОГ по всем проектам альбома")
    print(f"{'#' * 60}")
    print(f"Альбом: {data.get('source_album', '')}")
    print(f"Каталог нарезки: {data.get('projects_dir', '')}")
    print(f"Всего проектов: {len(data.get('projects', []))}")
    print(f"Соответствуют ТЗ ({len(data.get('compliant', []))}): {', '.join(data['compliant']) or '—'}")
    print(f"Не соответствуют ({len(data.get('non_compliant', []))}): {', '.join(data['non_compliant']) or '—'}")
    print(f"Нельзя определить ({len(data.get('undetermined', []))}): {', '.join(data['undetermined']) or '—'}")
    rep_path = data.get("pipeline_report_path")
    if rep_path:
        print(f"Сводный JSON: {rep_path}")


def run_album_pipeline(
    album_pdf: str,
    projects_dir: str,
    pipeline_report_path: str | None = None,
    *,
    model: str = "mistral",
    vision_model: str = "llava",
    top_k: int = 5,
    skip_ocr: bool = False,
    skip_caption: bool = False,
    split_use_ocr: bool = True,
    split_debug_signatures: bool = False,
    rebuild_tz: bool = False,
    rules_path: str = DEFAULT_RULES_PATH,
    spec_path: str = DEFAULT_SPEC_PATH,
    images_dir: str = DEFAULT_IMAGES_DIR,
    text_dir: str = DEFAULT_TEXT_DIR,
    index_dir: str = DEFAULT_INDEX_DIR,
    ocr_psm: int = 3,
) -> dict:
    """
    Нарезка альбома (pdf_project_splitter) → по очереди analyze_drawing для каждого PDF.
    Возвращает агрегат для JSON; сохраняет сводный отчёт на диск.
    """
    from scripts.pdf_project_splitter import split_by_projects

    album_path = Path(album_pdf).resolve()
    out_dir = Path(projects_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if rebuild_tz:
        _rebuild_tz_index(index_dir, spec_path, rules_path)

    ocr_cfg = f"--oem 3 --psm {ocr_psm}"
    print(f"\n=== Шаг 1: нарезка альбома по проектам → {out_dir} ===")
    manifest = split_by_projects(
        str(album_path),
        str(out_dir),
        use_ocr=split_use_ocr,
        ocr_dpi=200,
        ocr_config=ocr_cfg,
        debug_signatures=split_debug_signatures,
    )
    boundaries = manifest.get("boundaries") or []

    compliant: list[str] = []
    non_compliant: list[str] = []
    undetermined: list[str] = []
    projects: list[dict] = []

    if not boundaries:
        logger.warning("Нарезка не дала ни одного проекта — анализ ТЗ пропущен.")
    else:
        print(f"\n=== Шаг 2–3: извлечение данных и проверка ТЗ ({len(boundaries)} проектов) ===")

    for start_page, end_page, stem in boundaries:
        pdf_path = out_dir / f"{stem}.pdf"
        if not pdf_path.is_file():
            logger.error("После нарезки не найден файл: %s", pdf_path)
            continue

        logger.info("Анализ проекта %s (%s)", stem, pdf_path)
        print(f"\n--- Извлечение и ТЗ: {stem} (стр. альбома {start_page}–{end_page}) ---")

        report_dict = analyze_drawing(
            pdf_path=str(pdf_path),
            drawing_id=stem,
            output_path=None,
            model=model,
            vision_model=vision_model,
            top_k=top_k,
            skip_ocr=skip_ocr,
            skip_caption=skip_caption,
            images_dir=images_dir,
            text_dir=text_dir,
            index_dir=index_dir,
            rules_path=rules_path,
            spec_path=spec_path,
        )

        try:
            rep = DrawingReport.model_validate(report_dict)
            summary_line = rep.summary()
        except Exception:
            summary_line = ""

        comp = report_dict.get("compliance") or {}
        iso = comp.get("is_compliant")
        if iso is True:
            compliant.append(stem)
        elif iso is False:
            non_compliant.append(stem)
        else:
            undetermined.append(stem)

        projects.append(
            {
                "drawing_id": stem,
                "pdf_path": str(pdf_path.resolve()),
                "album_pages": {"start": start_page, "end": end_page},
                "report": report_dict,
            }
        )
        _print_project_pipeline_block(report_dict, stem, summary_line=summary_line)

    report_file = pipeline_report_path
    if not report_file:
        report_file = str(out_dir / f"{album_path.stem}_pipeline_report.json")
    else:
        report_file = str(Path(report_file).resolve())
        Path(report_file).parent.mkdir(parents=True, exist_ok=True)

    aggregate: dict = {
        "source_album": str(album_path),
        "projects_dir": str(out_dir),
        "split_boundaries": [
            {"start_page": a, "end_page": b, "stem": s} for a, b, s in boundaries
        ],
        "compliant": compliant,
        "non_compliant": non_compliant,
        "undetermined": undetermined,
        "projects": projects,
        "pipeline_report_path": report_file,
    }
    Path(report_file).write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Сводный отчёт конвейера: %s", report_file)

    _print_pipeline_final_summary(aggregate)
    return aggregate


def _print_batch_summary(data: dict) -> None:
    print(f"\n=== Сводка по PDF ({data['total_drawings']} чертежей-страниц) ===")
    print(f"Ollama: {'доступна' if data.get('ollama_reachable') else 'недоступна'}")
    print(f"Текстовая LLM ({data.get('text_llm_model')}): ", end="")
    print("да" if data.get("text_llm_available") else "нет (эвристика + rules.json)")
    print(f"\nПодходят по ТЗ ({len(data['compliant'])}): {', '.join(data['compliant']) or '—'}")
    print(f"Не подходят ({len(data['non_compliant'])}): {', '.join(data['non_compliant']) or '—'}")
    print(f"Нельзя определить ({len(data['undetermined'])}): {', '.join(data['undetermined']) or '—'}")

    for d in data.get("drawings", []):
        did = d.get("drawing_id", "?")
        comp = d.get("compliance", {})
        status = comp.get("is_compliant")
        label = "OK" if status is True else "НЕТ" if status is False else "?"
        print(f"\n--- {did} [{label}] ---")
        for issue in comp.get("issues", [])[:8]:
            print(f"  • [{issue.get('field')}] {issue.get('message')}")
        if len(comp.get("issues", [])) > 8:
            print(f"  … ещё {len(comp['issues']) - 8} расхождений")
        for m in comp.get("missing_info", [])[:4]:
            print(f"  (инфо) {m}")


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

    parser = argparse.ArgumentParser(
        description="Анализ PDF-чертежа(ей) с проверкой соответствия ТЗ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pdf", default=None, help="Путь к PDF-файлу")
    parser.add_argument(
        "--drawing-id",
        default=None,
        dest="drawing_id",
        help="Идентификатор: для одного файла — номер чертежа; с --all-pages — префикс для страниц",
    )
    parser.add_argument(
        "--check-ollama",
        action="store_true",
        help="Проверить связь с Ollama и наличие моделей (без анализа PDF)",
    )
    parser.add_argument(
        "--ollama-base-url",
        default=None,
        metavar="URL",
        help="Базовый URL Ollama, например http://127.0.0.1:11434 (или задайте OLLAMA_BASE_URL)",
    )
    parser.add_argument(
        "--all-pages",
        action="store_true",
        help="Каждая страница — отдельный чертёж (id: <drawing-id>-стр.N); сводный JSON",
    )
    parser.add_argument("--output", default=None, help="Путь для сохранения JSON")
    parser.add_argument("--model", default="mistral", help="Текстовая LLM Ollama")
    parser.add_argument("--vision-model", default="llava", help="Мультимодальная модель Ollama")
    parser.add_argument("--top-k", type=int, default=5, help="Чанков ТЗ для RAG")
    parser.add_argument("--skip-ocr", action="store_true")
    parser.add_argument("--skip-caption", action="store_true")
    parser.add_argument("--rebuild-tz", action="store_true")
    parser.add_argument("--rules", default=DEFAULT_RULES_PATH)
    parser.add_argument("--spec", default=DEFAULT_SPEC_PATH)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--split-output-dir",
        default=None,
        metavar="DIR",
        help="Только нарезать PDF-альбом в каталог (чертёж + титул/экспликация вместе); анализ ТЗ не запускается",
    )
    parser.add_argument(
        "--validate-split",
        action="store_true",
        help="После --split-output-dir прогнать validate_drawings по созданным PDF",
    )
    parser.add_argument(
        "--pipeline-album",
        action="store_true",
        help=(
            "Нарезать альбом по проектам (pdf_project_splitter), затем по очереди "
            "полный анализ каждого PDF и проверка ТЗ; сводный JSON (--pipeline-report)"
        ),
    )
    parser.add_argument(
        "--projects-dir",
        default=DEFAULT_PROJECTS_PIPELINE_DIR,
        metavar="DIR",
        help="Каталог для PDF проектов при --pipeline-album",
    )
    parser.add_argument(
        "--pipeline-report",
        default=None,
        metavar="PATH",
        help="Путь к сводному JSON (по умолчанию: <projects-dir>/<имя_альбома>_pipeline_report.json)",
    )
    parser.add_argument(
        "--split-no-ocr",
        action="store_true",
        help="При --pipeline-album — без OCR на этапе нарезки (только text-layer)",
    )
    parser.add_argument(
        "--debug-split-signatures",
        action="store_true",
        help="При --pipeline-album — вывести подписи проектов по страницам (см. pdf_project_splitter)",
    )
    args = parser.parse_args()

    if args.ollama_base_url:
        os.environ["OLLAMA_BASE_URL"] = args.ollama_base_url.strip().rstrip("/")

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s [%(module)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.check_ollama:
        print(diagnose_ollama(text_model=args.model, vision_model=args.vision_model))
        return

    if args.pipeline_album:
        if not args.pdf:
            parser.error("С --pipeline-album укажите --pdf")
        try:
            run_album_pipeline(
                album_pdf=args.pdf,
                projects_dir=args.projects_dir,
                pipeline_report_path=args.pipeline_report,
                model=args.model,
                vision_model=args.vision_model,
                top_k=args.top_k,
                skip_ocr=args.skip_ocr,
                skip_caption=args.skip_caption,
                split_use_ocr=not args.split_no_ocr,
                split_debug_signatures=args.debug_split_signatures,
                rebuild_tz=args.rebuild_tz,
                rules_path=args.rules,
                spec_path=args.spec,
            )
        except FileNotFoundError as e:
            print(f"ОШИБКА: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            logger.exception("Ошибка конвейера альбома")
            print(f"ОШИБКА: {e}", file=sys.stderr)
            sys.exit(1)
        print("\nКонвейер завершён.")
        return

    if args.split_output_dir:
        if not args.pdf:
            parser.error("Для нарезки укажите --pdf")
        from scripts.pdf_cutter import extract_drawings, validate_drawings

        print(f"\nНарезка альбома → {args.split_output_dir}")
        print("-" * 60)
        try:
            created = extract_drawings(str(Path(args.pdf).resolve()), args.split_output_dir)
            print(f"Создано файлов: {len(created)}")
            for stem, p in sorted(created.items()):
                print(f"  • {stem}.pdf")
            if args.validate_split:
                validate_drawings(args.split_output_dir)
        except FileNotFoundError as e:
            print(f"ОШИБКА: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            logger.exception("Ошибка нарезки")
            print(f"ОШИБКА: {e}", file=sys.stderr)
            sys.exit(1)
        return

    if not args.pdf or not args.drawing_id:
        parser.error(
            "Укажите --pdf и --drawing-id; или --check-ollama; или --split-output-dir с --pdf; "
            "или --pipeline-album с --pdf"
        )

    if args.rebuild_tz:
        _rebuild_tz_index(DEFAULT_INDEX_DIR, args.spec, args.rules)

    print(f"\nPDF: {args.pdf}")
    print(f"Режим: {'все страницы отдельно' if args.all_pages else 'весь PDF одним чертежом'}")
    print("-" * 60)

    try:
        if args.all_pages:
            result = analyze_pdf_all_pages(
                pdf_path=args.pdf,
                base_drawing_id=args.drawing_id,
                output_path=args.output,
                model=args.model,
                vision_model=args.vision_model,
                top_k=args.top_k,
                skip_ocr=args.skip_ocr,
                skip_caption=args.skip_caption,
                rules_path=args.rules,
                spec_path=args.spec,
            )
            _print_batch_summary(result)
        else:
            result = analyze_drawing(
                pdf_path=args.pdf,
                drawing_id=args.drawing_id,
                output_path=args.output,
                model=args.model,
                vision_model=args.vision_model,
                top_k=args.top_k,
                skip_ocr=args.skip_ocr,
                skip_caption=args.skip_caption,
                rules_path=args.rules,
                spec_path=args.spec,
            )
            report = DrawingReport(**result)
            print(f"\n{report.summary()}")
            print(f"Уверенность: {report.overall_confidence.value}")
            if report.compliance.issues:
                print(f"\nРАСХОЖДЕНИЯ ({len(report.compliance.issues)}):")
                for issue in report.compliance.issues:
                    print(f"  [{issue.field}] {issue.message}")
            if report.compliance.missing_info:
                print(f"\nНЕ ОПРЕДЕЛЕНО:")
                for m in report.compliance.missing_info:
                    print(f"  - {m}")
    except FileNotFoundError as e:
        print(f"ОШИБКА: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.exception("Ошибка анализа")
        print(f"ОШИБКА: {e}", file=sys.stderr)
        sys.exit(1)

    print("\nJSON сохранён.")


if __name__ == "__main__":
    main()
