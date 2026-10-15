"""Фиксированный RAG (workflow): поиск -> отбор -> один вызов генерации.

Порядок шагов задаёт код, а не модель. Используется как точка сравнения с
агентом (эксперимент E1) и как «нижняя граница стоимости».
Опционально - планировщик: декомпозиция вопроса на подвопросы.
"""

from __future__ import annotations

import re

from gigachat.models import Messages
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from ..config import PipelineConfig
from ..guardrails import check_input
from ..llm import LLM
from ..retrieval import Hit, Retriever, RetrievalTrace, detect_year
from .common import RunResult, finalize, local_answer, records_since, start_timer

PLANNER_SYSTEM = """
Ты планировщик поиска по расшифровкам сессий ПМЭФ. Не отвечай на вопрос.
Разбей вопрос на 1–3 независимых поисковых подвопроса, сохранив имена, год и сессию.
Если вопрос атомарный, верни его как единственный подвопрос.
""".strip()

PLAN_SCHEMA = {
    "type": "object",
    "properties": {"subquestions": {"type": "array", "items": {"type": "string"}}},
    "required": ["subquestions"],
    "additionalProperties": False,
}


class QueryPlan(BaseModel):
    """План поиска; нормализует мусор оформления, который модель копирует из промпта."""

    model_config = ConfigDict(extra="forbid")
    subquestions: list[str]

    @model_validator(mode="after")
    def clean(self) -> "QueryPlan":
        """Убирает нумерацию, кавычки и повторы из подвопросов плана."""
        cleaned = []
        for item in self.subquestions:
            text = re.sub(r"^[-*\d.)\s]+", "", " ".join(item.split()).strip("{}[]<>\"'`«» ")).strip()
            if text and text not in cleaned:
                cleaned.append(text)
        if not cleaned:
            raise ValueError("План пуст")
        self.subquestions = cleaned[:3]
        return self


def plan(llm: LLM, question: str, config: PipelineConfig, session_id: str | None) -> list[str]:
    """Подвопросы для поиска; при сбое - исходный вопрос (явная деградация)."""
    try:
        result = llm.chat(
            [Messages(role="system", content=PLANNER_SYSTEM), Messages(role="user", content=question)],
            tag="planner", model=config.model, temperature=0.01, max_tokens=200,
            response_format={"type": "json_schema", "schema": PLAN_SCHEMA, "strict": True},
            session_id=session_id,
        )
        return QueryPlan.model_validate_json(result.content).subquestions
    except (ValidationError, ValueError, RuntimeError):
        return [question]


def run_rag(question: str, config: PipelineConfig, llm: LLM, retriever: Retriever) -> RunResult:
    """Прогон одного вопроса через фиксированный RAG."""
    started = start_timer()
    log_start = len(llm.log.records)
    result = RunResult(question=question, config_id=config.config_id, final=None)
    result.guard = check_input(question)
    if not result.guard.allowed:
        result.final = local_answer("refused", f"guardrail:{result.guard.reason}")
        return result

    session_id = llm.new_session_id("rag")
    try:
        year = detect_year(question) if config.auto_year_filter else None
        queries = plan(llm, question, config, session_id) if config.planner else [question]
        merged: dict[str, Hit] = {}
        for query in queries:
            hits, _ = retriever.search(query, mode=config.retrieval_mode, top_k=config.top_k, year=year)
            for hit in hits:
                if hit.chunk.chunk_id not in merged or hit.score > merged[hit.chunk.chunk_id].score:
                    merged[hit.chunk.chunk_id] = hit
        hits = sorted(merged.values(), key=lambda h: -h.score)
        trace = RetrievalTrace(query=" | ".join(queries), mode=config.retrieval_mode, year_filter=year)
        result.blocks = retriever.build_blocks(
            hits, min_score=config.min_score, max_blocks=config.max_blocks,
            merge_neighbors=config.merge_neighbors, trace=trace,
        )
        result.retrieval_traces.append(trace)
        result.seen_chunk_ids = [h.chunk.chunk_id for h in hits]
        result.steps = len(queries)
        finalize(llm, question, result.blocks, config, session_id=session_id, result=result)
    except Exception as error:  # noqa: BLE001 - на внешней границе фиксируем любой сбой
        result.error = f"{type(error).__name__}: {str(error)[:300]}"
    finally:
        result.records = records_since(llm, log_start)
        result.latency_s = start_timer() - started
    return result
