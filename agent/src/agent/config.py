"""Конфигурация конвейера и цепочка конфигураций оптимизации.

``PipelineConfig`` - замороженный dataclass: старую конфигурацию нельзя тихо
поменять, можно только создать новую (карточка промпта,
карточка пайплайна). Цепочка A→… меняет ровно один слой за шаг,
и это проверяется функцией ``changed_fields``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace


@dataclass(frozen=True)
class PipelineConfig:
    """Все параметры, от которых зависит ответ и его стоимость."""

    config_id: str
    name: str
    architecture: str = "agent"  # rag | agent | sgr_agent
    model: str = "GigaChat-2-Max"
    # --- поиск и отбор контекста (для rag и для финального шага агента)
    retrieval_mode: str = "dense"  # dense | hybrid | lexical
    top_k: int = 8
    max_blocks: int = 8
    min_score: float | None = None
    merge_neighbors: bool = False
    auto_year_filter: bool = False
    planner: bool = False
    # --- генерация ответа
    context_format: str = "verbose"  # verbose | compact
    prompt: str = "verbose"  # verbose | compact
    examples: int = 3
    schema: str = "verbose"  # verbose | compact
    max_tokens: int = 1024
    temperature: float = 0.01
    max_attempts: int = 2  # вторая попытка - с обратной связью от проверок; одинаково во всей цепочке
    sc_samples: int = 1  # >1 включает self-consistency по статусу
    # --- агент
    agent_prompt: str = "verbose"
    tool_variant: str = "detailed"  # detailed | minimal
    max_steps: int = 6
    item_chars: int = 1000
    agent_max_tokens: int = 300
    finalize_from: str = "history"  # history | state
    compaction: str | None = None  # None | truncate | offload
    session_mode: str = "fresh"  # fresh | shared (кеш префикса внутри вопроса)
    timestamp_in_prefix: bool = False  # антипример: ломает кеш префикса

    def to_dict(self) -> dict:
        """Параметры для карточки эксперимента."""
        return asdict(self)


def changed_fields(a: PipelineConfig, b: PipelineConfig) -> list[str]:
    """Какие параметры различаются (кроме имени): защита честного сравнения."""
    skip = {"config_id", "name"}
    return [f.name for f in fields(PipelineConfig) if f.name not in skip and getattr(a, f.name) != getattr(b, f.name)]


# ------------------------------------------------------------------ цепочка оптимизации
# A - агент «как пишут обычно». Каждый следующий шаг меняет один слой.
A = PipelineConfig(
    "A", "A: baseline - агент, подробные промпты, финал по всей истории",
    architecture="agent", finalize_from="history", context_format="verbose", prompt="verbose",
    examples=3, schema="verbose", max_tokens=1024, agent_prompt="verbose", item_chars=1000,
    max_blocks=12, merge_neighbors=False,
)
B = replace(A, config_id="B", name="B: финальный ответ по состоянию (собранные фрагменты), а не по истории",
            finalize_from="state")
C = replace(B, config_id="C", name="C: очистка контекста: склейка соседей, порог, не больше 4 окон, компактный формат",
            merge_neighbors=True, min_score=None, max_blocks=4, context_format="compact")
D = replace(C, config_id="D", name="D: компактные инструкции генератора и агента", prompt="compact", agent_prompt="compact")
E = replace(D, config_id="E", name="E: без примеров (3 -> 0)", examples=0)
F = replace(E, config_id="F", name="F: компактная схема ответа", schema="compact")
G = replace(F, config_id="G", name="G: max_tokens по фактическому распределению выхода", max_tokens=400)
H = replace(G, config_id="H", name="H: агент получает фрагменты, обрезанные до 600 символов", item_chars=600)
I = replace(G, config_id="I", name="I: общая сессия внутри вопроса (кеш префикса)", session_mode="shared")

OPTIMIZATION_CHAIN = [A, B, C, D, E, F, G, H, I]

# Фиксированный RAG без агента (workflow) - для сравнения архитектур (E1).
RAG_BASELINE = replace(
    A, config_id="RAG-A", name="RAG-A: фиксированный RAG, top-8, подробные промпты",
    architecture="rag", retrieval_mode="dense", top_k=8, max_blocks=8,
)
RAG_OPT = replace(
    G, config_id="RAG-G", name="RAG-G: фиксированный RAG с оптимизациями G",
    architecture="rag", retrieval_mode="dense", top_k=8, max_blocks=4, auto_year_filter=True,
)


def by_id(config_id: str) -> PipelineConfig:
    """Конфигурация по идентификатору (включая экспериментальные)."""
    from .experiments import ALL_CONFIGS

    for config in ALL_CONFIGS:
        if config.config_id == config_id:
            return config
    raise KeyError(f"Нет конфигурации {config_id}. Доступны: {[c.config_id for c in ALL_CONFIGS]}")
