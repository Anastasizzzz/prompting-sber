"""Инструменты агента для function calling (критерий 2).

Два поисковых инструмента, которых требует задание, и два вспомогательных:

* ``search_transcripts`` - поиск №1, смысловой (эмбеддинги EmbeddingsGigaR);
* ``keyword_search`` - поиск №2, лексический BM25 (имена, организации, числа);
* ``list_sessions`` - каталог сессий: выбрать нужную сессию и год;
* ``get_fragment`` - продолжение фрагмента по ``chunk_id`` (данные по ссылке,
  «just-in-time» контекст).

Правила: у функции два лица - Python-код для приложения и
описание для модели; функции проверяются кодом до подключения модели; имя
проверяется по белому списку, аргументы - Pydantic-моделью; все инструменты
только читают данные (принцип минимальных полномочий, OWASP LLM06).
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .retrieval import Hit, Retriever
from .text import normalize, stem, translit_ru_to_lat, words

YEARS = [2011, 2012, 2013, 2014, 2015, 2016]


# ---------------------------------------------------------------- аргументы (проверка до исполнения)
class SearchArgs(BaseModel):
    """Аргументы поисковых функций."""

    model_config = ConfigDict(extra="ignore")
    query: str = Field(min_length=2, max_length=500)
    year: int | None = Field(default=None, ge=2011, le=2016)
    top_k: int = Field(default=4, ge=1, le=8)


class ListSessionsArgs(BaseModel):
    """Аргументы каталога сессий."""

    model_config = ConfigDict(extra="ignore")
    year: int | None = Field(default=None, ge=2011, le=2016)
    query: str | None = Field(default=None, max_length=200)


class FragmentArgs(BaseModel):
    """Аргументы догрузки фрагмента."""

    model_config = ConfigDict(extra="ignore")
    chunk_id: str = Field(min_length=5)
    window: int = Field(default=1, ge=0, le=2)


# ---------------------------------------------------------------- описания для модели
def tool_specs(variant: str = "detailed") -> list[dict]:
    """Спецификации функций для GigaChat.

    ``detailed`` - когда вызывать, когда не вызывать, что значат поля результата.
    ``minimal`` - только суть (эксперимент E9: описания сокращают последними и
    с проверкой траектории).
    """
    year_prop = {"type": "integer", "description": "Год форума 2011–2016. Передавай, если год есть в вопросе."}
    if variant == "minimal":
        year_prop = {"type": "integer", "description": "Год форума."}
    descriptions = {
        "detailed": {
            "search_transcripts": (
                "Смысловой поиск по расшифровкам сессий ПМЭФ 2011–2016. Используй для вопросов о том, "
                "что обсуждали, кто что сказал, о позициях и оценках. Возвращает фрагменты: chunk_id, "
                "сессия, время, спикеры, текст. Работает и для английских фрагментов по русскому запросу."
            ),
            "keyword_search": (
                "Поиск по точным словам: фамилии, названия компаний и организаций, числа, термины. "
                "Используй, когда смысловой поиск не нашёл нужное имя или число. Фамилии на кириллице "
                "находит и в латинском написании. Возвращает те же поля, что search_transcripts."
            ),
            "list_sessions": (
                "Каталог сессий форума: source_id, название (год, дата, тип, тема), язык, спикеры и "
                "opening_chunk_id - начало сессии. Используй, чтобы понять, какая сессия имеется в виду, "
                "особенно когда тема повторялась в разные годы. Текстов сессий не возвращает."
            ),
            "get_fragment": (
                "Возвращает фрагмент по chunk_id вместе с соседними фрагментами той же сессии. "
                "Используй, если найденная реплика обрывается и нужен её контекст, а также чтобы прочитать "
                "начало сессии (opening_chunk_id): там модератор объявляет название, организаторов и участников."
            ),
        },
        "minimal": {
            "search_transcripts": "Поиск по расшифровкам.",
            "keyword_search": "Поиск по словам.",
            "list_sessions": "Список сессий.",
            "get_fragment": "Фрагмент по id.",
        },
    }[variant]
    query_desc = "Поисковый запрос с ключевыми сущностями вопроса." if variant == "detailed" else "Запрос."
    return [
        {
            "name": "search_transcripts",
            "description": descriptions["search_transcripts"],
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": query_desc},
                    "year": year_prop,
                    "top_k": {"type": "integer", "description": "Сколько фрагментов вернуть, 1–8, обычно 4."},
                },
                "required": ["query"],
            },
        },
        {
            "name": "keyword_search",
            "description": descriptions["keyword_search"],
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Точные слова: фамилия, организация, число." if variant == "detailed" else "Слова."},
                    "year": year_prop,
                    "top_k": {"type": "integer", "description": "Сколько фрагментов вернуть, 1–8."},
                },
                "required": ["query"],
            },
        },
        {
            "name": "list_sessions",
            "description": descriptions["list_sessions"],
            "parameters": {
                "type": "object",
                "properties": {
                    "year": year_prop,
                    "query": {"type": "string", "description": "Слово из темы сессии, например it, nuclear, film." if variant == "detailed" else "Тема."},
                },
                "required": [],
            },
        },
        {
            "name": "get_fragment",
            "description": descriptions["get_fragment"],
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string", "description": "chunk_id из результата поиска."},
                    "window": {"type": "integer", "description": "Сколько соседних фрагментов с каждой стороны, 0–2."},
                },
                "required": ["chunk_id"],
            },
        },
    ]


# ---------------------------------------------------------------- исполнение
@dataclass
class ToolContext:
    """Состояние, которое инструменты пополняют: найденные фрагменты и трасса.

    Это и есть «явное состояние вместо истории»: для финального
    ответа нужен не весь диалог с функциями, а собранные доказательства.
    """

    retriever: Retriever
    item_chars: int = 1000
    seen_chunks: dict[str, float] = field(default_factory=dict)
    pinned: set[str] = field(default_factory=set)  # явно запрошены через get_fragment
    embedding_tokens: int = 0
    calls: list[dict] = field(default_factory=list)

    def remember(self, hits: list[Hit]) -> None:
        """Запоминает найденные фрагменты с лучшей оценкой."""
        for hit in hits:
            previous = self.seen_chunks.get(hit.chunk.chunk_id)
            score = hit.dense_score if hit.dense_score is not None else hit.score
            if previous is None or score > previous:
                self.seen_chunks[hit.chunk.chunk_id] = score


def _items(hits: list[Hit], item_chars: int) -> list[dict]:
    """Результаты поиска в виде, который возвращается модели."""
    rows = []
    for hit in hits:
        chunk = hit.chunk
        text = chunk.text if len(chunk.text) <= item_chars else chunk.text[:item_chars] + "…"
        rows.append(
            {
                "chunk_id": chunk.chunk_id,
                "session": chunk.session_title,
                "time": chunk.time_start,
                "speakers": chunk.speakers,
                "text": text,
            }
        )
    return rows


def search_transcripts(ctx: ToolContext, args: SearchArgs) -> dict:
    """Смысловой поиск с фильтром по году."""
    hits, tokens = ctx.retriever.dense(args.query, top_k=args.top_k, year=args.year)
    ctx.embedding_tokens += tokens
    ctx.remember(hits)
    return {"query": args.query, "year": args.year, "count": len(hits), "items": _items(hits, ctx.item_chars)}


def keyword_search(ctx: ToolContext, args: SearchArgs) -> dict:
    """Лексический поиск BM25 с фильтром по году."""
    hits = ctx.retriever.lexical(args.query, top_k=args.top_k, year=args.year)
    ctx.remember(hits)
    return {"query": args.query, "year": args.year, "count": len(hits), "items": _items(hits, ctx.item_chars)}


def _catalog_terms(text: str) -> set[str]:
    """Слова для сопоставления с каталогом: без стоп-слов (тема ``it`` - не местоимение), со стеммингом и транслитом."""
    terms = set()
    for word in words(text):
        if len(word) >= 2:
            terms.update({stem(word), stem(translit_ru_to_lat(word))})
    return terms


def list_sessions(ctx: ToolContext, args: ListSessionsArgs) -> dict:
    """Каталог сессий с фильтрами по году и словам в теме/спикерах.

    Совпадение ищется по отдельным словам запроса (с транслитерацией): тема в
    каталоге взята из имени файла латиницей (``it``, ``nuclear``), а модель пишет
    «IT-завтрак» или «атомная энергетика». Пустой результат сопровождается
    подсказкой: каталог не содержит текстов, ответ ищется поиском по содержанию.
    """
    rows = list(ctx.retriever.catalog.values())
    if args.year is not None:
        rows = [row for row in rows if row["year"] == args.year]
    if args.query:
        needles = _catalog_terms(args.query)

        def matches(row: dict) -> int:
            """Число общих слов запроса и строки каталога."""
            haystack = _catalog_terms(row["title"] + " " + " ".join(row["speakers"]))
            return len(needles & haystack)

        scored = [(matches(row), row) for row in rows]
        rows = [row for score, row in sorted(scored, key=lambda pair: -pair[0]) if score > 0]
    total = len(rows)
    rows = rows[:25]
    result = {
        "count": total,
        "truncated": total > len(rows),
        "sessions": [
            {"source_id": r["source_id"], "title": r["title"], "language": r["language"],
             "speakers": r["speakers"][:5], "opening_chunk_id": f"{r['source_id']}#0000"}
            for r in rows
        ],
    }
    if total == 0:
        result["hint"] = "В названиях сессий совпадений нет. Ищи по содержанию: search_transcripts или keyword_search."
    return result


def get_fragment(ctx: ToolContext, args: FragmentArgs) -> dict:
    """Фрагмент и соседи по ``chunk_id``; неизвестный id - явная ошибка, а не пустота."""
    if args.chunk_id not in ctx.retriever.by_id:
        return {"error": f"chunk_id {args.chunk_id} не найден"}
    chunks = ctx.retriever.neighbors(args.chunk_id, args.window)
    hits = [Hit(chunk, ctx.seen_chunks.get(args.chunk_id, 0.0)) for chunk in chunks]
    ctx.remember(hits)
    ctx.pinned.update(chunk.chunk_id for chunk in chunks)
    return {"count": len(hits), "items": _items(hits, ctx.item_chars)}


HANDLERS: dict[str, tuple[type[BaseModel], Callable[[ToolContext, Any], dict]]] = {
    "search_transcripts": (SearchArgs, search_transcripts),
    "keyword_search": (SearchArgs, keyword_search),
    "list_sessions": (ListSessionsArgs, list_sessions),
    "get_fragment": (FragmentArgs, get_fragment),
}


def dispatch(ctx: ToolContext, name: str, arguments: dict | str | None) -> dict:
    """Исполняет вызов функции, запрошенный моделью.

    Модель не может вызвать произвольный код: имя сверяется с белым списком,
    аргументы проходят проверку Pydantic. Ошибка возвращается модели как
    результат - по ней она поймёт, что путь не сработал.
    """
    if name not in HANDLERS:
        result = {"error": f"Неизвестная функция {name}. Доступны: {', '.join(HANDLERS)}"}
        ctx.calls.append({"name": name, "arguments": arguments, "error": result["error"]})
        return result
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    args_model, handler = HANDLERS[name]
    try:
        args = args_model.model_validate(arguments or {})
    except ValidationError as error:
        result = {"error": f"Неверные аргументы: {error.errors(include_url=False)[0]['msg']}"}
        ctx.calls.append({"name": name, "arguments": arguments, "error": result["error"]})
        return result
    result = handler(ctx, args)
    ctx.calls.append({
        "name": name,
        "arguments": args.model_dump(exclude_none=True),
        "result_ids": [item.get("chunk_id") for item in result.get("items", [])],
        "count": result.get("count"),
    })
    return result


def tool_specs_copy(variant: str = "detailed") -> list[dict]:
    """Копия спецификаций, которую безопасно менять в экспериментах."""
    return copy.deepcopy(tool_specs(variant))
