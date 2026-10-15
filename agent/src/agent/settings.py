"""Пути, переменные окружения и параметры подключения к GigaChat.

Все настройки собраны в одном месте, чтобы прогон можно было воспроизвести:
карточка эксперимента сохраняет их вместе с результатами.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

AGENT_DIR = Path(__file__).resolve().parents[2]
REPO_DIR = AGENT_DIR.parent

# Источник истины для данных проекта - папка raw_data/ в корне репозитория.
DATA_DIR = REPO_DIR / "raw_data"
CORPUS_DIR = DATA_DIR / "SPIEF_txt"
GOLDEN_XLSX = DATA_DIR / "Золотой набор.xlsx"

# Всё вычисляемое живёт внутри agent/.
AGENT_DATA_DIR = AGENT_DIR / "data"
INDEX_DIR = AGENT_DATA_DIR / "index"
CACHE_DIR = AGENT_DATA_DIR / "cache"
RESULTS_DIR = AGENT_DIR / "results"
RUNS_DIR = RESULTS_DIR / "runs"
FIGURES_DIR = RESULTS_DIR / "figures"

# Ключ ищем сначала в agent/.env, затем в корне репозитория. Значение никогда не печатаем.
for _env_path in (AGENT_DIR / ".env", REPO_DIR / ".env"):
    if _env_path.exists():
        load_dotenv(_env_path, override=False)


def _env_flag(name: str, default: bool) -> bool:
    """Читает логический флаг из окружения: 1/true/yes считаются истиной."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class GigaChatSettings:
    """Параметры подключения к GigaChat.

    ``counter_base_url`` вынесен отдельно: метод ``/tokens/count`` отвечает
    не на всех хостах API, поэтому счётчик поднимается
    отдельным клиентом.
    """

    credentials: str = field(repr=False, default_factory=lambda: os.getenv("GIGACHAT_CREDENTIALS", "").strip())
    scope: str = field(default_factory=lambda: os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_B2B"))
    base_url: str = field(default_factory=lambda: os.getenv("GIGACHAT_BASE_URL", "https://api.giga.chat/v1"))
    counter_base_url: str = field(
        default_factory=lambda: os.getenv(
            "GIGACHAT_COUNTER_BASE_URL", "https://gigachat.devices.sberbank.ru/api/v1"
        )
    )
    # Учебный обход цепочки сертификатов российских УЦ. В production проверку TLS
    # не отключают, а ставят корневой сертификат.
    verify_ssl_certs: bool = field(default_factory=lambda: _env_flag("GIGACHAT_VERIFY_SSL", False))
    model: str = field(default_factory=lambda: os.getenv("GIGACHAT_MODEL", "GigaChat-2-Max"))
    embedding_model: str = field(
        default_factory=lambda: os.getenv("GIGACHAT_EMBEDDING_MODEL", "EmbeddingsGigaR")
    )
    timeout_s: float = 120.0

    @property
    def has_credentials(self) -> bool:
        """Есть ли ключ в окружении (сам ключ не раскрывается)."""
        return bool(self.credentials)


def ensure_dirs() -> None:
    """Создаёт каталоги для вычисляемых артефактов."""
    for path in (AGENT_DATA_DIR, INDEX_DIR, CACHE_DIR, RESULTS_DIR, RUNS_DIR, FIGURES_DIR):
        path.mkdir(parents=True, exist_ok=True)
