"""
index_rag.py — индексация ТЗ и описаний чертежей в FAISS для RAG-поиска.

Строит два индекса:
  - tz_index: чанки из docs/tech_spec.md + config/rules.json (источник истины)
  - drawings_index: тексты OCR + LLaVA-описания (для поиска похожих чертежей)

Модель эмбеддингов: paraphrase-multilingual-MiniLM-L12-v2
(поддерживает кириллицу, работает локально без интернета после скачивания).

Пример запуска:
    python -m scripts.index_rag --build-tz
    python -m scripts.index_rag --search "толщина стенки"
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
INDEX_DIR = Path("data/faiss_index")
TZ_INDEX_FILE = INDEX_DIR / "tz_index.faiss"
TZ_META_FILE = INDEX_DIR / "tz_metadata.json"
DRAWINGS_INDEX_FILE = INDEX_DIR / "drawings_index.faiss"
DRAWINGS_META_FILE = INDEX_DIR / "drawings_metadata.json"

# Размер чанка (символы) и перекрытие
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


# ---------------------------------------------------------------------------
# Чанкинг текста
# ---------------------------------------------------------------------------


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Разбивает текст на перекрывающиеся чанки по границам абзацев/предложений.

    Приоритет разбиения: абзацы → предложения → символы.
    """
    if not text.strip():
        return []

    # Разбиваем по абзацам (двойной перенос)
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current) + len(para) <= chunk_size:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
                # Перекрытие: берём хвост предыдущего чанка
                overlap_text = current[-overlap:] if len(current) > overlap else current
                current = (overlap_text + "\n\n" + para).strip()
            else:
                # Абзац сам по себе слишком длинный — разбиваем по предложениям
                sentences = re.split(r"(?<=[.!?])\s+", para)
                for sent in sentences:
                    if len(current) + len(sent) <= chunk_size:
                        current = (current + " " + sent).strip()
                    else:
                        if current:
                            chunks.append(current)
                        current = sent

    if current:
        chunks.append(current)

    return chunks


def rules_json_to_text(rules: dict) -> str:
    """Конвертирует rules.json в читаемый текст для индексации."""
    lines = ["=== Машиночитаемые правила ТЗ ===\n"]

    for dim in rules.get("dimensions", []):
        ref = dim.get("tz_reference", "")
        std = dim.get("standard", "")
        if "allowed_values" in dim:
            vals = ", ".join(dim["allowed_values"])
            lines.append(
                f"Поле '{dim['field']}' ({dim.get('description', '')}): "
                f"допустимые значения: {vals}. "
                f"Единица: {dim.get('unit', '')}. Стандарт: {std}. Ссылка: {ref}."
            )
        elif "min" in dim and "max" in dim:
            lines.append(
                f"Поле '{dim['field']}' ({dim.get('description', '')}): "
                f"диапазон {dim['min']}–{dim['max']} {dim.get('unit', '')}. "
                f"Стандарт: {std}. Ссылка: {ref}."
            )

    mats = rules.get("materials", {})
    if mats:
        allowed = ", ".join(mats.get("allowed", []))
        forbidden = ", ".join(mats.get("forbidden", []))
        lines.append(
            f"Материалы: допустимые: {allowed}. "
            f"Запрещённые: {forbidden}. "
            f"Стандарт: {mats.get('standard', '')}. "
            f"Ссылка: {mats.get('tz_reference', '')}."
        )

    des = rules.get("designations", {})
    if des:
        lines.append(
            f"Обозначения: паттерн '{des.get('pattern', '')}'. "
            f"Примеры: {', '.join(des.get('examples', []))}. "
            f"Ссылка: {des.get('tz_reference', '')}."
        )

    coatings = rules.get("coatings", {})
    if coatings:
        codes = ", ".join(coatings.get("allowed_codes", []))
        lines.append(
            f"Покрытия: допустимые коды: {codes}. "
            f"Стандарт: {coatings.get('standard', '')}. "
            f"Ссылка: {coatings.get('tz_reference', '')}."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Загрузка модели эмбеддингов
# ---------------------------------------------------------------------------


def _load_embedder():
    """Загружает sentence-transformer модель (однократно)."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBED_MODEL_NAME)
        logger.info("Модель эмбеддингов загружена: %s", EMBED_MODEL_NAME)
        return model
    except ImportError:
        raise RuntimeError("sentence-transformers не установлен: pip install sentence-transformers")


def _embed_texts(texts: list[str], embedder) -> np.ndarray:
    """Возвращает матрицу эмбеддингов для списка текстов."""
    embeddings = embedder.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    return np.array(embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# Построение индексов
# ---------------------------------------------------------------------------


def build_tz_index(
    spec_path: str = "docs/tech_spec.md",
    rules_path: str = "config/rules.json",
    index_dir: str = str(INDEX_DIR),
) -> None:
    """
    Строит FAISS-индекс из ТЗ (tech_spec.md + rules.json).

    Сохраняет индекс и метаданные на диск.
    """
    import faiss

    index_dir_obj = Path(index_dir)
    index_dir_obj.mkdir(parents=True, exist_ok=True)

    texts: list[str] = []
    meta: list[dict] = []

    # Читаем tech_spec.md
    spec_file = Path(spec_path)
    if spec_file.exists():
        spec_text = spec_file.read_text(encoding="utf-8")
        chunks = chunk_text(spec_text)
        for i, chunk in enumerate(chunks):
            texts.append(chunk)
            meta.append({"source": "tech_spec.md", "chunk_id": i, "text": chunk})
        logger.info("tech_spec.md: %d чанков", len(chunks))
    else:
        logger.warning("tech_spec.md не найден: %s", spec_path)

    # Читаем rules.json
    rules_file = Path(rules_path)
    if rules_file.exists():
        rules = json.loads(rules_file.read_text(encoding="utf-8"))
        rules_text = rules_json_to_text(rules)
        chunks = chunk_text(rules_text)
        for i, chunk in enumerate(chunks):
            texts.append(chunk)
            meta.append({"source": "rules.json", "chunk_id": i, "text": chunk})
        logger.info("rules.json: %d чанков", len(chunks))
    else:
        logger.warning("rules.json не найден: %s", rules_path)

    if not texts:
        raise ValueError("Нет данных для индексации ТЗ — проверьте пути к файлам")

    embedder = _load_embedder()
    embeddings = _embed_texts(texts, embedder)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner Product для нормализованных векторов = cosine similarity
    index.add(embeddings)

    tz_index_path = index_dir_obj / "tz_index.faiss"
    tz_meta_path = index_dir_obj / "tz_metadata.json"

    faiss.write_index(index, str(tz_index_path))
    tz_meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("ТЗ-индекс сохранён: %s (%d записей)", tz_index_path, len(texts))


def add_drawing_to_index(
    drawing_id: str,
    text: str,
    page_number: int = 1,
    index_dir: str = str(INDEX_DIR),
) -> None:
    """
    Добавляет текст чертежа (OCR + LLaVA caption) в drawings-индекс.

    Если индекс не существует — создаёт новый.
    """
    import faiss

    index_dir_obj = Path(index_dir)
    index_dir_obj.mkdir(parents=True, exist_ok=True)

    drawings_index_path = index_dir_obj / "drawings_index.faiss"
    drawings_meta_path = index_dir_obj / "drawings_metadata.json"

    embedder = _load_embedder()
    chunks = chunk_text(text)
    if not chunks:
        logger.warning("Нет текста для индексации чертежа %s", drawing_id)
        return

    embeddings = _embed_texts(chunks, embedder)
    dim = embeddings.shape[1]

    # Загружаем существующий индекс или создаём новый
    if drawings_index_path.exists():
        index = faiss.read_index(str(drawings_index_path))
        meta = json.loads(drawings_meta_path.read_text(encoding="utf-8"))
    else:
        index = faiss.IndexFlatIP(dim)
        meta = []

    index.add(embeddings)
    for i, chunk in enumerate(chunks):
        meta.append({
            "drawing_id": drawing_id,
            "page_number": page_number,
            "chunk_id": i,
            "text": chunk,
        })

    faiss.write_index(index, str(drawings_index_path))
    drawings_meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Чертёж %s добавлен в индекс: %d чанков", drawing_id, len(chunks))


# ---------------------------------------------------------------------------
# Поиск по индексу
# ---------------------------------------------------------------------------


def search_tz(
    query: str,
    top_k: int = 5,
    index_dir: str = str(INDEX_DIR),
) -> list[dict]:
    """
    Семантический поиск по ТЗ-индексу.

    Args:
        query: Поисковый запрос (текст элемента чертежа).
        top_k: Количество возвращаемых чанков.
        index_dir: Директория с индексом.

    Returns:
        Список словарей с полями: text, source, score.
    """
    import faiss

    tz_index_path = Path(index_dir) / "tz_index.faiss"
    tz_meta_path = Path(index_dir) / "tz_metadata.json"

    if not tz_index_path.exists():
        logger.error("ТЗ-индекс не найден. Запустите: python -m scripts.index_rag --build-tz")
        return []

    embedder = _load_embedder()
    index = faiss.read_index(str(tz_index_path))
    meta = json.loads(tz_meta_path.read_text(encoding="utf-8"))

    query_vec = embedder.encode([query], normalize_embeddings=True)
    query_vec = np.array(query_vec, dtype=np.float32)

    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(meta):
            continue
        entry = meta[idx].copy()
        entry["score"] = float(score)
        results.append(entry)

    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="FAISS-индексация ТЗ и чертежей")
    parser.add_argument("--build-tz", action="store_true", help="Построить индекс ТЗ")
    parser.add_argument("--search", type=str, help="Поисковый запрос по ТЗ")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--spec", default="docs/tech_spec.md")
    parser.add_argument("--rules", default="config/rules.json")
    parser.add_argument("--index-dir", default=str(INDEX_DIR))
    args = parser.parse_args()

    if args.build_tz:
        build_tz_index(spec_path=args.spec, rules_path=args.rules, index_dir=args.index_dir)
        print("ТЗ-индекс успешно построен.")

    if args.search:
        results = search_tz(args.search, top_k=args.top_k, index_dir=args.index_dir)
        print(f"\nРезультаты поиска '{args.search}' (top-{args.top_k}):\n")
        for i, r in enumerate(results, 1):
            print(f"[{i}] score={r['score']:.3f} | source={r['source']}")
            print(f"    {r['text'][:200]}...\n")


if __name__ == "__main__":
    main()
