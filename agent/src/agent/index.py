"""Офлайн-контур: построение и загрузка поискового индекса (один раз).

Индекс оплачивается однократно, поэтому строится тщательно: с проверками,
с возобновлением после сбоя сети (векторы пишутся на диск порциями) и с
манифестом, в котором зафиксированы параметры нарезки, модель и стоимость.
Повторный запуск ничего не оплачивает, если индекс уже собран.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .corpus import Chunk, Corpus, chunk_corpus, load_corpus, session_catalog_rows
from .settings import CORPUS_DIR, INDEX_DIR
from .tariffs import rub, tokens_cost_rub

CHUNK_MAX_CHARS = 1000
CHUNK_OVERLAP_CHARS = 150


@dataclass
class IndexFiles:
    """Пути файлов индекса."""

    root: Path

    @property
    def chunks(self) -> Path:
        """Файл с фрагментами корпуса."""
        return self.root / "chunks.jsonl"

    @property
    def embeddings(self) -> Path:
        """Матрица векторов фрагментов."""
        return self.root / "embeddings.npy"

    @property
    def partial(self) -> Path:
        """Промежуточная матрица векторов для продолжения прерванной индексации."""
        return self.root / "embeddings.partial.npy"

    @property
    def manifest(self) -> Path:
        """Манифест индекса: параметры, стоимость, дата сборки."""
        return self.root / "manifest.json"

    @property
    def catalog(self) -> Path:
        """Каталог уникальных сессий."""
        return self.root / "sessions.json"

    @property
    def duplicates(self) -> Path:
        """Найденные дубли и их канонические файлы."""
        return self.root / "duplicates.json"


def prepare_chunks(corpus_dir: Path = CORPUS_DIR) -> tuple[Corpus, list[Chunk]]:
    """Загружает корпус, исключает дубли и режет на фрагменты (без вызовов API)."""
    corpus = load_corpus(corpus_dir)
    chunks = chunk_corpus(corpus, max_chars=CHUNK_MAX_CHARS, overlap_chars=CHUNK_OVERLAP_CHARS)
    ids = [chunk.chunk_id for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("chunk_id не уникальны")
    if any(not chunk.text.strip() for chunk in chunks):
        raise ValueError("Есть пустые фрагменты")
    return corpus, chunks


def estimate_index_cost(chunks: list[Chunk], chars_per_token: float = 3.6) -> dict:
    """Оценка стоимости индексации до запуска (оценка до вызова)."""
    chars = sum(len(chunk.embedding_text()) for chunk in chunks)
    tokens = round(chars / chars_per_token)
    return {
        "chunks": len(chunks),
        "characters": chars,
        "estimated_tokens": tokens,
        "estimated_cost_rub": rub(tokens_cost_rub(tokens, "EmbeddingsGigaR"), 2),
    }


def build_index(llm, *, index_dir: Path = INDEX_DIR, batch_size: int = 32) -> dict:
    """Строит индекс: фрагменты, эмбеддинги, каталог сессий, манифест.

    Векторы сохраняются порциями в ``embeddings.partial.npy``: оборвалась сеть -
    повторный запуск продолжит с места остановки и не оплатит готовое повторно.
    """
    files = IndexFiles(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    corpus, chunks = prepare_chunks()
    if files.manifest.exists() and files.embeddings.exists():
        manifest = json.loads(files.manifest.read_text(encoding="utf-8"))
        if manifest.get("chunks") == len(chunks):
            print("Индекс уже построен, повторная индексация не нужна.")
            return manifest

    with files.chunks.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    files.catalog.write_text(
        json.dumps(session_catalog_rows(corpus), ensure_ascii=False, indent=1), encoding="utf-8"
    )
    files.duplicates.write_text(json.dumps(corpus.duplicates, ensure_ascii=False, indent=1), encoding="utf-8")

    estimate = estimate_index_cost(chunks)
    print(
        f"Индексация: {estimate['chunks']} фрагментов, ≈{estimate['estimated_tokens']:,} токенов, "
        f"≈{estimate['estimated_cost_rub']} ₽ (один раз)".replace(",", " ")
    )

    done: list[np.ndarray] = []
    if files.partial.exists():
        done = [np.load(files.partial, allow_pickle=False)]
    start = int(done[0].shape[0]) if done else 0
    if start:
        print(f"Продолжаю с фрагмента {start}")
    started = time.perf_counter()
    spent_tokens = 0
    texts = [chunk.embedding_text() for chunk in chunks]
    step = batch_size * 10
    for begin in range(start, len(texts), step):
        vectors, tokens = llm.embed(texts[begin : begin + step], tag="index_embeddings", batch_size=batch_size)
        spent_tokens += tokens
        block = np.asarray(vectors, dtype=np.float32)
        block /= np.linalg.norm(block, axis=1, keepdims=True)
        done.append(block.astype(np.float16))
        np.save(files.partial, np.vstack(done), allow_pickle=False)
        print(f"  {min(begin + step, len(texts))}/{len(texts)} фрагментов, токенов {spent_tokens:,}".replace(",", " "))

    matrix = np.vstack(done)
    if matrix.shape[0] != len(chunks):
        raise RuntimeError("Число векторов не совпало с числом фрагментов")
    np.save(files.embeddings, matrix, allow_pickle=False)
    if files.partial.exists():
        files.partial.unlink()

    manifest = {
        "corpus_dir": str(CORPUS_DIR.name),
        "files_total": len(corpus.sessions),
        "sessions_unique": len(corpus.canonical_ids),
        "duplicates_removed": len(corpus.duplicates),
        "chunks": len(chunks),
        "chunk_max_chars": CHUNK_MAX_CHARS,
        "chunk_overlap_chars": CHUNK_OVERLAP_CHARS,
        "embedding_model": llm.settings.embedding_model,
        "vector_dim": int(matrix.shape[1]),
        "embedding_tokens_this_run": spent_tokens,
        "cost_rub_this_run": rub(tokens_cost_rub(spent_tokens, llm.settings.embedding_model), 2),
        "build_seconds": round(time.perf_counter() - started, 1),
        "built_at": time.strftime("%Y-%m-%d %H:%M"),
    }
    files.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return manifest


def load_index(index_dir: Path = INDEX_DIR) -> tuple[list[Chunk], np.ndarray, list[dict], dict[str, str]]:
    """Загружает фрагменты, нормализованную матрицу векторов, каталог сессий и карту дублей."""
    files = IndexFiles(index_dir)
    if not files.embeddings.exists():
        raise FileNotFoundError("Индекс не построен. Запустите ./run.sh index")
    chunks = [Chunk(**json.loads(line)) for line in files.chunks.read_text(encoding="utf-8").splitlines()]
    matrix = np.load(files.embeddings).astype(np.float32)
    if matrix.shape[0] != len(chunks):
        raise ValueError(f"Фрагментов {len(chunks)}, векторов {matrix.shape[0]} - индекс повреждён")
    catalog = json.loads(files.catalog.read_text(encoding="utf-8"))
    duplicates = json.loads(files.duplicates.read_text(encoding="utf-8"))
    return chunks, matrix, catalog, duplicates
