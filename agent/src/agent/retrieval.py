"""Поиск по корпусу: векторный, лексический (BM25), гибридный и отбор контекста.

Онлайн-контур: вопрос -> эмбеддинг запроса -> ранжирование ->
отбор контекста. Оплачивается эмбеддинг запроса и **каждый токен контекста**,
поэтому отбор сделан пошаговым и проверяемым:

1. фильтр по метаданным (год, тип сессии) - в тексте фрагментов года нет;
2. порог сходства, подобранный на калибровочной части золотого набора;
3. склейка соседних фрагментов одной сессии в одно окно без дублей;
4. ограничение числа окон.

Полная выдача с оценками и причинами отбора пишется в retrieval trace и
модели не показывается (trace ≠ citations).
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .corpus import Chunk
from .settings import CACHE_DIR, INDEX_DIR
from .text import lexical_tokens

# Инструкция добавляется только к запросу, к документам - нет: это асимметрия
# обучения EmbeddingsGigaR.
RETRIEVAL_INSTRUCTION = "Дан вопрос, необходимо найти абзац, наиболее релевантный по смыслу\n"

YEAR_RE = re.compile(r"(?:ПМЭФ|SPIEF|форум\w*)[\s\-–-]*(20(?:1[1-6]))|\b(20(?:1[1-6]))\s*(?:г\.|год)", re.I)


def detect_year(question: str) -> int | None:
    """Год форума из вопроса («ПМЭФ-2016», «в 2014 году»). Нужен для фильтра по метаданным."""
    years = {int(a or b) for a, b in YEAR_RE.findall(question)}
    return years.pop() if len(years) == 1 else None


@dataclass
class Hit:
    """Найденный фрагмент с оценками по способам поиска."""

    chunk: Chunk
    score: float
    dense_score: float | None = None
    bm25_score: float | None = None
    rank: int = 0


@dataclass
class ContextBlock:
    """Окно контекста для генератора: один или несколько соседних фрагментов одной сессии."""

    chunk_ids: list[str]
    source_id: str
    session_title: str
    time_start: str
    time_end: str
    speakers: list[str]
    text: str
    score: float

    @property
    def block_id(self) -> str:
        """Идентификатор окна - первый фрагмент; на него ссылается модель."""
        return self.chunk_ids[0]


@dataclass
class RetrievalTrace:
    """Полная история поиска для разработчика и оценки (не для модели)."""

    query: str
    mode: str
    year_filter: int | None
    candidates: list[dict] = field(default_factory=list)
    selected_block_ids: list[str] = field(default_factory=list)
    embedding_tokens: int = 0


class BM25:
    """Лексический поиск Okapi BM25 без внешних зависимостей.

    Ловит то, на чём эмбеддинги «плывут»: фамилии, организации, числа - такая ошибка
    поиска лечится добавлением лексического поиска. К словам запроса
    на кириллице добавляется транслитерация: «Амано» находит «Amano».
    """

    def __init__(self, documents: list[list[str]], k1: float = 1.4, b: float = 0.75) -> None:
        """Строит инвертированный индекс по токенизированным документам."""
        self.k1, self.b = k1, b
        self.doc_len = np.array([len(doc) for doc in documents], dtype=np.float32)
        self.avg_len = float(self.doc_len.mean()) if len(documents) else 0.0
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for idx, doc in enumerate(documents):
            for term, tf in Counter(doc).items():
                postings[term].append((idx, tf))
        self.postings = dict(postings)
        n = len(documents)
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}
        self.n_docs = n

    def scores(self, query: str) -> np.ndarray:
        """Оценки BM25 всех документов для запроса."""
        result = np.zeros(self.n_docs, dtype=np.float32)
        for term in set(lexical_tokens(query, add_translit=True)):
            if term not in self.postings:
                continue
            idf = self.idf[term]
            for idx, tf in self.postings[term]:
                norm = tf + self.k1 * (1 - self.b + self.b * self.doc_len[idx] / self.avg_len)
                result[idx] += idf * tf * (self.k1 + 1) / norm
        return result


class Retriever:
    """Поиск по индексу проекта и сборка контекста."""

    def __init__(
        self,
        chunks: list[Chunk],
        matrix: np.ndarray,
        catalog: list[dict],
        duplicates: dict[str, str],
        llm=None,
        *,
        index_dir: Path = INDEX_DIR,
    ) -> None:
        """Поиск над фрагментами, матрицей векторов, каталогом сессий и картой дублей."""
        self.chunks = chunks
        self.matrix = matrix
        self.catalog = {row["source_id"]: row for row in catalog}
        self.duplicates = duplicates
        self.llm = llm
        self.by_id = {chunk.chunk_id: i for i, chunk in enumerate(chunks)}
        self.years = np.array([chunk.year for chunk in chunks])
        self.sources = np.array([chunk.source_id for chunk in chunks])
        self.bm25 = self._load_bm25(index_dir)
        self._query_cache_path = CACHE_DIR / "query_embeddings.json"
        self._query_cache = self._load_query_cache()

    # ---------------------------------------------------------------- служебное
    def _load_bm25(self, index_dir: Path) -> BM25:
        """BM25 из файла, если он построен по тому же числу фрагментов, иначе - заново."""
        path = index_dir / "bm25.pkl"
        if path.exists():
            with path.open("rb") as handle:
                bm25 = pickle.load(handle)
            if bm25.n_docs == len(self.chunks):
                return bm25
        docs = [lexical_tokens(f"{c.session_title} {' '.join(c.speakers)} {c.text}") for c in self.chunks]
        bm25 = BM25(docs)
        with path.open("wb") as handle:
            pickle.dump(bm25, handle)
        return bm25

    def _load_query_cache(self) -> dict[str, list[float]]:
        """Кеш векторов вопросов с диска."""
        if self._query_cache_path.exists():
            return json.loads(self._query_cache_path.read_text(encoding="utf-8"))
        return {}

    def chunk(self, chunk_id: str) -> Chunk:
        """Фрагмент по идентификатору."""
        return self.chunks[self.by_id[chunk_id]]

    def canonical_source(self, source_id: str) -> str:
        """Канонический файл для дубля."""
        return self.duplicates.get(source_id, source_id)

    # ---------------------------------------------------------------- эмбеддинг запроса
    def embed_query(self, query: str) -> tuple[np.ndarray, int]:
        """Вектор запроса с инструкцией. Повторные запросы берутся из точного кеша.

        Кеш эмбеддингов безопасен: ключ - точный текст запроса и модель,
        значение не зависит ни от промпта, ни от прав доступа.
        """
        text = f"{RETRIEVAL_INSTRUCTION}вопрос: {' '.join(query.split())}"
        key = hashlib.sha256(text.encode()).hexdigest()
        if key in self._query_cache:
            return np.asarray(self._query_cache[key], dtype=np.float32), 0
        if self.llm is None:
            raise RuntimeError("Для векторного поиска нужен клиент LLM")
        vectors, tokens = self.llm.embed([text], tag="query_embedding")
        vector = np.asarray(vectors[0], dtype=np.float32)
        vector /= np.linalg.norm(vector)
        self._query_cache[key] = [round(float(x), 6) for x in vector]
        self._query_cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._query_cache_path.write_text(json.dumps(self._query_cache), encoding="utf-8")
        return vector, tokens

    # ---------------------------------------------------------------- поиск
    def _mask(self, year: int | None, source_ids: list[str] | None) -> np.ndarray:
        """Маска фрагментов по году и списку сессий."""
        mask = np.ones(len(self.chunks), dtype=bool)
        if year is not None:
            mask &= self.years == year
        if source_ids:
            mask &= np.isin(self.sources, [self.canonical_source(s) for s in source_ids])
        return mask

    def question_scores(self, question: str, chunk_ids: list[str]) -> tuple[dict[str, float], int]:
        """Косинус с исходным вопросом для любых фрагментов - единая шкала для порога и ранжирования.

        Фрагменты в состоянии агента приходят из разных инструментов: косинус
        (search_transcripts), BM25 (keyword_search), явный запрос (get_fragment).
        Эти оценки несравнимы, поэтому перед отбором всё переоценивается
        относительно вопроса пользователя (вектор вопроса берётся из кеша).
        """
        if not chunk_ids:
            return {}, 0
        vector, tokens = self.embed_query(question)
        idx = [self.by_id[cid] for cid in chunk_ids]
        scores = self.matrix[idx].astype(np.float32) @ vector
        return {cid: float(score) for cid, score in zip(chunk_ids, scores)}, tokens

    def dense(self, query: str, *, top_k: int = 8, year: int | None = None,
              source_ids: list[str] | None = None) -> tuple[list[Hit], int]:
        """Векторный поиск по косинусному сходству."""
        vector, tokens = self.embed_query(query)
        scores = self.matrix @ vector
        scores = np.where(self._mask(year, source_ids), scores, -1.0)
        order = np.argsort(-scores)[:top_k]
        return [Hit(self.chunks[i], float(scores[i]), dense_score=float(scores[i]), rank=r)
                for r, i in enumerate(order, 1) if scores[i] > -1], tokens

    def lexical(self, query: str, *, top_k: int = 8, year: int | None = None,
                source_ids: list[str] | None = None) -> list[Hit]:
        """Лексический поиск BM25."""
        scores = self.bm25.scores(query)
        scores = np.where(self._mask(year, source_ids), scores, -1.0)
        order = np.argsort(-scores)[:top_k]
        return [Hit(self.chunks[i], float(scores[i]), bm25_score=float(scores[i]), rank=r)
                for r, i in enumerate(order, 1) if scores[i] > 0]

    def hybrid(self, query: str, *, top_k: int = 8, year: int | None = None,
               source_ids: list[str] | None = None, pool: int = 30, rrf_k: int = 60) -> tuple[list[Hit], int]:
        """Гибрид: Reciprocal Rank Fusion векторной и лексической выдачи."""
        dense_hits, tokens = self.dense(query, top_k=pool, year=year, source_ids=source_ids)
        lexical_hits = self.lexical(query, top_k=pool, year=year, source_ids=source_ids)
        fused: dict[str, Hit] = {}
        for hits, attr in ((dense_hits, "dense_score"), (lexical_hits, "bm25_score")):
            for hit in hits:
                item = fused.setdefault(hit.chunk.chunk_id, Hit(hit.chunk, 0.0))
                item.score += 1.0 / (rrf_k + hit.rank)
                setattr(item, attr, getattr(hit, attr))
        ranked = sorted(fused.values(), key=lambda h: -h.score)[:top_k]
        for rank, hit in enumerate(ranked, 1):
            hit.rank = rank
        return ranked, tokens

    def search(self, query: str, *, mode: str = "dense", top_k: int = 8, year: int | None = None,
               source_ids: list[str] | None = None) -> tuple[list[Hit], int]:
        """Единая точка поиска: ``dense`` | ``lexical`` | ``hybrid``."""
        if mode == "dense":
            return self.dense(query, top_k=top_k, year=year, source_ids=source_ids)
        if mode == "lexical":
            return self.lexical(query, top_k=top_k, year=year, source_ids=source_ids), 0
        if mode == "hybrid":
            return self.hybrid(query, top_k=top_k, year=year, source_ids=source_ids)
        raise ValueError(f"Неизвестный режим поиска: {mode}")

    # ---------------------------------------------------------------- отбор контекста
    def neighbors(self, chunk_id: str, window: int = 1) -> list[Chunk]:
        """Фрагмент и его соседи в той же сессии."""
        center = self.by_id[chunk_id]
        source = self.chunks[center].source_id
        result = []
        for idx in range(center - window, center + window + 1):
            if 0 <= idx < len(self.chunks) and self.chunks[idx].source_id == source:
                result.append(self.chunks[idx])
        return result

    def build_blocks(self, hits: list[Hit], *, min_score: float | None = None, max_blocks: int = 4,
                     merge_neighbors: bool = True, use_dense_threshold: bool = True,
                     trace: RetrievalTrace | None = None, pinned: set[str] | None = None) -> list[ContextBlock]:
        """Порог -> склейка соседних фрагментов -> ограничение числа окон.

        Порог применяется к косинусной оценке (для гибрида - к её векторной части):
        RRF-оценка не сравнима между запросами. Закреплённые фрагменты (``pinned``:
        агент явно запросил их через get_fragment) порогом не отсекаются.
        """
        pinned = pinned or set()
        kept: list[Hit] = []
        for hit in hits:
            value = hit.dense_score if use_dense_threshold and hit.dense_score is not None else hit.score
            is_pinned = hit.chunk.chunk_id in pinned
            passed = is_pinned or min_score is None or value is None or value >= min_score
            if trace is not None:
                trace.candidates.append({
                    "chunk_id": hit.chunk.chunk_id, "rank": hit.rank, "score": round(hit.score, 4),
                    "dense": None if hit.dense_score is None else round(hit.dense_score, 4),
                    "bm25": None if hit.bm25_score is None else round(hit.bm25_score, 3),
                    "decision": ("kept_pinned" if is_pinned else "kept") if passed else "below_threshold",
                })
            if passed:
                kept.append(hit)

        blocks: list[ContextBlock] = []
        used: set[str] = set()
        for hit in kept:
            if hit.chunk.chunk_id in used:
                continue
            members = [hit.chunk]
            if merge_neighbors:
                # Соседний фрагмент склеивается, только если он тоже есть в выдаче:
                # так убираются дубли, а контекст не раздувается.
                kept_ids = {h.chunk.chunk_id for h in kept}
                idx = self.by_id[hit.chunk.chunk_id]
                for step in (-1, 1):
                    j = idx + step
                    while 0 <= j < len(self.chunks) and self.chunks[j].chunk_id in kept_ids \
                            and self.chunks[j].source_id == hit.chunk.source_id \
                            and self.chunks[j].chunk_id not in used:
                        members.append(self.chunks[j])
                        j += step
            members.sort(key=lambda c: c.position)
            for member in members:
                used.add(member.chunk_id)
            speakers: dict[str, None] = {}
            for member in members:
                for speaker in member.speakers:
                    speakers.setdefault(speaker, None)
            blocks.append(
                ContextBlock(
                    chunk_ids=[m.chunk_id for m in members],
                    source_id=hit.chunk.source_id,
                    session_title=hit.chunk.session_title,
                    time_start=members[0].time_start,
                    time_end=members[-1].time_end,
                    speakers=list(speakers),
                    text="\n".join(m.text for m in members),
                    score=hit.score,
                )
            )
            if len(blocks) >= max_blocks:
                break
        if trace is not None:
            trace.selected_block_ids = [block.block_id for block in blocks]
            selected = {cid for block in blocks for cid in block.chunk_ids}
            for row in trace.candidates:
                if row["decision"] == "kept" and row["chunk_id"] not in selected:
                    row["decision"] = "limit_or_merged"
        return blocks


def render_context(blocks: list[ContextBlock], *, fmt: str = "compact") -> str:
    """Контекст для генератора.

    ``verbose`` - «как пишут обычно»: XML со всеми служебными полями и оценкой.
    ``compact`` - только то, что нужно для ответа и ссылки: id, сессия, время, спикеры, текст.
    Оценка релевантности модели не нужна: ранжирование уже сделал код.
    """
    parts = []
    for block in blocks:
        if fmt == "verbose":
            parts.append(
                f'<document source_id="{block.source_id}" chunk_id="{block.block_id}" '
                f'score="{block.score:.4f}" time_start="{block.time_start}" time_end="{block.time_end}" '
                f'chunks="{",".join(block.chunk_ids)}">\n'
                f"<session>{block.session_title}</session>\n"
                f"<speakers>{', '.join(block.speakers)}</speakers>\n"
                f"<text>{block.text}</text>\n</document>"
            )
        else:
            # Диапазон времени, а не только начало: без него модель не может проверить «в конце сессии».
            parts.append(f"[{block.block_id} | {block.session_title} | {block.time_start}–{block.time_end}]\n{block.text}")
    return "\n\n".join(parts)


def load_retriever(llm=None) -> Retriever:
    """Загружает индекс с диска и создаёт ``Retriever``."""
    from .index import load_index

    chunks, matrix, catalog, duplicates = load_index()
    return Retriever(chunks, matrix, catalog, duplicates, llm)
