"""Утилиты для проверки доступности Ollama и наличия моделей."""

from __future__ import annotations

import logging
import os
from typing import Optional
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

# По умолчанию 127.0.0.1: на Windows «localhost» иногда уходит в IPv6 (::1),
# а Ollama слушает только IPv4 — запросы «в никуда».
_FALLBACK_BASE = "http://127.0.0.1:11434"


def get_ollama_base_url() -> str:
    """Базовый URL без завершающего слэша (переменная окружения OLLAMA_BASE_URL)."""
    u = os.environ.get("OLLAMA_BASE_URL", "").strip().rstrip("/")
    return u or _FALLBACK_BASE


def get_ollama_generate_url() -> str:
    """URL эндпоинта /api/generate."""
    return f"{get_ollama_base_url()}/api/generate"


def generate_url_to_base(generate_url: str) -> str:
    """http://host:11434/api/generate -> http://host:11434"""
    p = urlparse(generate_url)
    return f"{p.scheme}://{p.netloc}"


def _candidate_bases() -> list[str]:
    """Порядок попыток подключения (основной URL + fallback localhost <-> 127.0.0.1)."""
    primary = get_ollama_base_url()
    out = [primary]
    if "127.0.0.1" in primary:
        alt = primary.replace("127.0.0.1", "localhost", 1)
        if alt not in out:
            out.append(alt)
    elif "localhost" in primary.lower():
        alt = primary.replace("localhost", "127.0.0.1", 1).replace("Localhost", "127.0.0.1", 1)
        if alt not in out:
            out.append(alt)
    return out


def is_ollama_reachable(generate_url: Optional[str] = None, timeout: float = 5.0) -> bool:
    """Проверяет, отвечает ли Ollama (GET /api/tags)."""
    urls: list[str] = []
    if generate_url:
        urls.append(generate_url_to_base(generate_url))
    else:
        urls = _candidate_bases()

    for base in urls:
        try:
            r = requests.get(f"{base}/api/tags", timeout=timeout)
            if r.status_code == 200:
                if base != get_ollama_base_url():
                    logger.info("Ollama ответила через %s (fallback)", base)
                return True
        except Exception as e:
            logger.debug("Ollama %s: %s", base, e)
    return False


def ollama_model_loaded(
    model_name: str,
    generate_url: Optional[str] = None,
    timeout: float = 5.0,
) -> bool:
    """
    Проверяет, есть ли модель в списке локальных моделей Ollama.

    model_name: короткое имя (mistral) или с тегом (mistral:latest).
    """
    bases = [generate_url_to_base(generate_url)] if generate_url else _candidate_bases()
    short = model_name.split(":", 1)[0]

    for base in bases:
        try:
            r = requests.get(f"{base}/api/tags", timeout=timeout)
            r.raise_for_status()
            data = r.json()
            names: list[str] = []
            for m in data.get("models", []):
                n = m.get("name", "")
                if n:
                    names.append(n)
                    names.append(n.split(":", 1)[0])
            if model_name in names or short in names:
                return True
        except Exception as e:
            logger.debug("tags %s: %s", base, e)
    return False


def diagnose_ollama(text_model: str = "mistral", vision_model: str = "llava") -> str:
    """
    Текстовая сводка для пользователя: почему LLM может быть «недоступна» и что сделать.
    """
    lines: list[str] = []
    gen = get_ollama_generate_url()
    base = get_ollama_base_url()
    lines.append(f"Базовый URL: {base} (переменная OLLAMA_BASE_URL, если нужен другой хост)")
    lines.append("")

    reachable = False
    for b in _candidate_bases():
        try:
            r = requests.get(f"{b}/api/tags", timeout=5)
            reachable = r.status_code == 200
            if reachable:
                lines.append(f"Связь с Ollama: OK ({b})")
                models = r.json().get("models", [])
                names = [m.get("name", "") for m in models if m.get("name")]
                lines.append(f"  Установленные модели: {', '.join(names) if names else '(пусто)'}")
                break
        except Exception as e:
            lines.append(f"  Попытка {b}: ошибка — {e}")

    if not reachable:
        lines.append("")
        lines.append("Связь с Ollama: НЕТ.")
        lines.append("  • Установите Ollama: https://ollama.com/download")
        lines.append("  • Запустите приложение Ollama (в трее Windows) или в терминале: ollama serve")
        lines.append("  • Если сервер на другом ПК: set OLLAMA_BASE_URL=http://IP:11434")
        return "\n".join(lines)

    if not ollama_model_loaded(text_model, gen):
        lines.append("")
        lines.append(f"Текстовая модель «{text_model}» не найдена среди загруженных.")
        lines.append(f"  Выполните: ollama pull {text_model}")
    else:
        lines.append(f"Текстовая модель «{text_model}»: OK")

    if not ollama_model_loaded(vision_model, gen):
        lines.append("")
        lines.append(f"Мультимодальная модель «{vision_model}» не найдена (описание чертежей).")
        lines.append(f"  Выполните: ollama pull {vision_model}")
    else:
        lines.append(f"Мультимодальная модель «{vision_model}»: OK")

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Диагностика Ollama для PDF Drawing Analyzer")
    p.add_argument("--model", default="mistral", help="Имя текстовой модели")
    p.add_argument("--vision-model", default="llava", help="Имя multimodal модели")
    p.add_argument("--ollama-base-url", default=None, help="Базовый URL (или OLLAMA_BASE_URL)")
    a = p.parse_args()
    if a.ollama_base_url:
        os.environ["OLLAMA_BASE_URL"] = a.ollama_base_url.strip().rstrip("/")
    print(diagnose_ollama(text_model=a.model, vision_model=a.vision_model))
