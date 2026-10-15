"""Кеши: готовый результат по строгому ключу и семантический кеш.

Три механизма, которые нельзя путать:

1. **кеш входа GigaChat** (``X-Session-ID``) - переиспользует совпавший префикс,
   ответ генерируется всегда; живёт в llm.py, меряется в эксперименте E7;
2. **кеш готового результата** - пропускает весь вызов; ключ включает всё, что
   меняет ответ: нормализованный вопрос, конфигурацию, хеши промптов и схем,
   версию корпуса. Метка «v2» вместо хеша пережила бы незаметную правку;
3. **семантический кеш** - отдаёт ответ на *похожий* вопрос; его риск - ложные
   попадания на ловушках («IT-завтрак 2014» против «2016»). Порог эту проблему
   не решает, решает гибридный ключ: точная часть (годы, числа, имена) + смысл.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np

from .config import PipelineConfig
from .prompts import PROMPT_CARDS
from .settings import CACHE_DIR, INDEX_DIR
from .text import normalize


def stable_hash(value) -> str:
    """Детерминированный хеш: сортировка ключей обязательна (иначе ломается кеш префикса)."""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def corpus_version(index_dir: Path = INDEX_DIR) -> str:
    """Версия корпуса = хеш манифеста индекса: обновили базу - старые ответы недействительны."""
    manifest = index_dir / "manifest.json"
    if not manifest.exists():
        return "no-index"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    return stable_hash({k: data.get(k) for k in ("chunks", "chunk_max_chars", "chunk_overlap_chars", "embedding_model", "built_at")})[:16]


class ResultCache:
    """Кеш готовых ответов на диске со строгим ключом и временем жизни."""

    def __init__(self, path: Path = CACHE_DIR / "result_cache.json", ttl_seconds: int = 7 * 24 * 3600) -> None:
        """Загружает кеш с диска; ``ttl_seconds`` - срок жизни записи."""
        self.path = path
        self.ttl = ttl_seconds
        self.data: dict = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    @staticmethod
    def key(question: str, config: PipelineConfig) -> str:
        """Ключ: всё, от чего зависит ответ."""
        return stable_hash({
            "question": normalize(question),
            "config": config.to_dict(),
            "prompts": {name: card.sha256 for name, card in PROMPT_CARDS.items()},
            "corpus": corpus_version(),
        })

    def get(self, question: str, config: PipelineConfig) -> dict | None:
        """Ответ из кеша или None (промах / истёк срок)."""
        entry = self.data.get(self.key(question, config))
        if entry and time.time() - entry["created_at"] <= self.ttl:
            return entry["value"]
        return None

    def put(self, question: str, config: PipelineConfig, value: dict) -> None:
        """Кешируются только ответы, прошедшие все проверки: брак в кеш не кладём."""
        self.data[self.key(question, config)] = {"value": value, "created_at": time.time()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")


# ------------------------------------------------------------------ семантический кеш
_TOPIC_WORDS = [
    "it", "завтрак", "кибер", "нефт", "газ", "атом", "ядер", "реактор", "кино", "музык", "арктик",
    "почт", "биржа", "финтех", "лес", "авиа", "образован", "стартап", "hyperloop", "торгов",
]
_NEGATIONS = ["не ", "без ", "нельзя", "кроме"]


def discriminators(question: str) -> frozenset[str]:
    """Точная часть ключа: годы, числа, имена собственные, тема, отрицания.

    Список специфичен для домена ПМЭФ: универсального не бывает.
    """
    text = normalize(question)
    found = set(re.findall(r"\d+", text))
    raw_words = re.findall(r"[A-ZА-ЯЁ][a-zа-яё]+", question)
    found.update(w.lower()[:5] for w in raw_words[1:])  # имена и организации (первое слово - начало фразы)
    found.update(t for t in _TOPIC_WORDS if t in text)
    found.update(n.strip() for n in _NEGATIONS if n in f" {text} ")
    for ordinal, value in (("перв", "1"), ("втор", "2"), ("трет", "3"), ("четверт", "4"), ("пят", "5"), ("шест", "6")):
        if re.search(rf"\b{ordinal}\w*", text):
            found.add(f"ord{value}")
    return frozenset(found)


def evaluate_semantic_cache(probe: list[dict], vectors: dict[str, np.ndarray], thresholds: list[float],
                            *, hybrid: bool) -> list[dict]:
    """Доля и точность попаданий семантического кеша на наборе база / перефразы / ловушки."""
    base = [p for p in probe if p["group"] == "base"]
    rows = []
    for threshold in thresholds:
        hits = false_hits = misses = 0
        for item in probe:
            if item["group"] == "base":
                continue
            sims = [(float(vectors[item["id"]] @ vectors[b["id"]]), b) for b in base]
            hit = False
            for sim, b in sorted(sims, key=lambda x: -x[0]):
                if sim < threshold:
                    break
                if hybrid and discriminators(item["question"]) != discriminators(b["question"]):
                    continue
                hit = True
                break
            if item["expect_hit"] and hit:
                hits += 1
            elif item["expect_hit"]:
                misses += 1
            elif hit:
                false_hits += 1
        total_hits = hits + false_hits
        rows.append({
            "ключ": "гибридный" if hybrid else "только смысл",
            "порог": threshold,
            "попаданий": total_hits,
            "ложных": false_hits,
            "доля попаданий на перефразах": round(hits / max(hits + misses, 1), 2),
            "точность попаданий": round(hits / total_hits, 2) if total_hits else 1.0,
        })
    return rows
