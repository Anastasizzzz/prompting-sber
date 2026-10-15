"""Конвейеры ответа на вопрос: фиксированный RAG, ReAct-агент, SGR-агент."""

from __future__ import annotations

from ..config import PipelineConfig
from ..llm import LLM
from ..retrieval import Retriever
from .agent import run_agent
from .common import RunResult
from .rag import run_rag
from .sgr_agent import run_sgr_agent

RUNNERS = {"rag": run_rag, "agent": run_agent, "sgr_agent": run_sgr_agent}


def answer_question(question: str, config: PipelineConfig, llm: LLM, retriever: Retriever) -> RunResult:
    """Единая точка входа: выбирает конвейер по ``config.architecture``."""
    return RUNNERS[config.architecture](question, config, llm, retriever)


__all__ = ["RunResult", "answer_question", "run_agent", "run_rag", "run_sgr_agent"]
