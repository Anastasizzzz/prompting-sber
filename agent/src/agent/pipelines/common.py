"""Общие части конвейеров: генерация структурированного ответа и результат прогона.

Финальный ответ - всегда отдельный вызов со строгой схемой (``response_format``,
``strict=true``) и перечислением ``chunk_id`` текущего контекста.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime

from gigachat.models import Messages

from ..config import PipelineConfig
from ..contracts import FinalAnswer, ValidationReport, build_answer_schema, parse_and_validate
from ..guardrails import InputCheck, check_fragment
from ..llm import LLM, CallRecord, GenerationResultError
from ..prompts import (
    COMPACT_GENERATOR_SYSTEM,
    GENERATOR_USER_TEMPLATE,
    VERBOSE_GENERATOR_SYSTEM,
    render_examples,
)
from ..retrieval import ContextBlock, RetrievalTrace, render_context


@dataclass
class RunResult:
    """Всё о прогоне одного вопроса: ответ, контекст, трасса, расход."""

    question: str
    config_id: str
    final: FinalAnswer | None
    blocks: list[ContextBlock] = field(default_factory=list)
    retrieval_traces: list[RetrievalTrace] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    seen_chunk_ids: list[str] = field(default_factory=list)
    records: list[CallRecord] = field(default_factory=list)
    steps: int = 0
    latency_s: float = 0.0
    guard: InputCheck | None = None
    error: str | None = None
    sc_votes: dict | None = None

    # ---- расход по журналу вызовов
    def _sum(self, attr: str, kind: str | None = "chat", prefix: str = "") -> int:
        """Сумма поля по записям журнала нужного вида и этапа."""
        return sum(
            getattr(r, attr) for r in self.records
            if (kind is None or r.kind == kind) and r.tag.startswith(prefix)
        )

    @property
    def cost_rub(self) -> float:
        """Стоимость всех вызовов вопроса: генерация, агент и эмбеддинги."""
        return round(sum(r.cost_rub for r in self.records), 6)

    @property
    def tokens(self) -> dict:
        """Сводка токенов: оплаченные, полный вход, кеш, выход, эмбеддинги."""
        return {
            "total_tokens": self._sum("total_tokens"),
            "prompt_tokens": self._sum("prompt_tokens"),
            "precached_prompt_tokens": self._sum("precached_prompt_tokens"),
            "raw_input_tokens": self._sum("prompt_tokens") + self._sum("precached_prompt_tokens"),
            "completion_tokens": self._sum("completion_tokens"),
            "agent_total_tokens": self._sum("total_tokens", prefix="agent"),
            "final_total_tokens": self._sum("total_tokens", prefix="final"),
            "final_completion_tokens": self._sum("completion_tokens", prefix="final"),
            "embedding_tokens": self._sum("total_tokens", kind="embeddings"),
            "chat_calls": sum(1 for r in self.records if r.kind == "chat"),
        }

    def to_public(self) -> dict:
        """Структурированный ответ для пользователя + служебная мета (критерий 3)."""
        payload = self.final.to_public() if self.final else {"status": "error", "answer": None, "citations": []}
        payload["meta"] = {
            "config_id": self.config_id,
            "steps": self.steps,
            "tool_calls": [c["name"] for c in self.tool_calls],
            "tokens": self.tokens,
            "cost_rub": self.cost_rub,
            "latency_s": round(self.latency_s, 2),
            "error": self.error,
            "answered_at": datetime.now().isoformat(timespec="seconds"),
        }
        return payload


FILTER_RETRIES = 4


def local_answer(status: str, reason: str) -> FinalAnswer:
    """Ответ без вызова модели: пустой контекст или сработал guardrail (политика сбоев).

    ``invalid`` - генерация не удалась (обрыв, фильтр): такой ответ не проходит проверки.
    """
    report = ValidationReport(schema_valid=status != "invalid")
    report.error = reason
    return FinalAnswer(status=status, answer=None, citations=[], evidence_sufficient=False, validation=report)


def system_prompt(config: PipelineConfig) -> str:
    """Системная инструкция генератора с примерами (стабильная часть - в начале промпта)."""
    base = VERBOSE_GENERATOR_SYSTEM if config.prompt == "verbose" else COMPACT_GENERATOR_SYSTEM
    examples = render_examples(config.examples, verbose=config.schema == "verbose")
    text = f"{base}\n\n{examples}" if examples else base
    if config.timestamp_in_prefix:
        # Антипример: изменяемая строка в самом начале ломает кеш префикса.
        text = f"Текущее время: {datetime.now().isoformat(timespec='microseconds')}\n{text}"
    return text


def generate_answer(
    llm: LLM,
    question: str,
    blocks: list[ContextBlock],
    config: PipelineConfig,
    *,
    session_id: str | None,
    tag: str = "final",
    temperature: float | None = None,
) -> FinalAnswer:
    """Структурированный ответ по контексту со всеми проверками.

    Если ответ не прошёл проверки, модель переспрашивается (``max_attempts``) с
    конкретной обратной связью: какая цитата не найдена дословно, какие числа
    не подтверждены (цикл «проверка -> исправление»). Сообщение с
    ошибкой добавляется в конец, поэтому префикс промпта остаётся в кеше.
    Каждая попытка оплачена и остаётся в журнале: брак - часть стоимости
    принятого ответа.

    Политика сбоя «тематический фильтр» (``finish_reason=blacklist``): фильтр
    срабатывает на входе на отдельном фрагменте и почти ничего не стоит
    (``prompt_tokens=1``). Контекст делится пополам, половины пробуются по
    очереди (бинарный поиск проходящего набора), не больше
    ``FILTER_RETRIES`` раз и без расхода попыток на исправление ответа.
    """
    def build_messages(selected: list[ContextBlock]) -> list[Messages]:
        """Сообщения генератора: стабильная инструкция в начале, контекст и вопрос в конце."""
        context = render_context(selected, fmt=config.context_format)
        return [
            Messages(role="system", content=system_prompt(config)),
            Messages(role="user", content=GENERATOR_USER_TEMPLATE.format(context=context, question=question)),
        ]

    # Слой «данные»: фрагмент, который обращается к ассистенту как к исполнителю инструкций, в контекст
    # не попадает (на корпусе ПМЭФ 0 срабатываний из 14 481 - проверено; срабатывание = атака).
    quarantined = [b.block_id for b in blocks if check_fragment(b.text)]
    if quarantined:
        blocks = [b for b in blocks if b.block_id not in quarantined]
        if not blocks:
            answer = local_answer("not_found", "all_fragments_quarantined")
            answer.quarantined = quarantined
            return answer
    pending: list[list[ContextBlock]] = []
    messages = build_messages(blocks)
    schema = build_answer_schema(config.schema, [b.block_id for b in blocks])
    last: FinalAnswer | None = None
    attempt, filter_retries, calls = 0, 0, 0
    while attempt < config.max_attempts:
        calls += 1
        try:
            result = llm.chat(
                messages,
                tag=f"{tag}_generation" if calls == 1 else f"{tag}_generation_retry{calls}",
                model=config.model,
                temperature=config.temperature if temperature is None else temperature,
                max_tokens=config.max_tokens,
                response_format={"type": "json_schema", "schema": schema, "strict": True},
                session_id=session_id,
            )
        except GenerationResultError as error:
            last = local_answer("invalid", f"generation_{error.reason}")
            if error.reason == "blacklist" and filter_retries < FILTER_RETRIES:
                filter_retries += 1
                if len(blocks) > 1:
                    half = len(blocks) // 2
                    pending = [blocks[:half], blocks[half:]] + pending
                if not pending:
                    break
                blocks = pending.pop(0)
                messages = build_messages(blocks)
                schema = build_answer_schema(config.schema, [b.block_id for b in blocks])
                continue
            attempt += 1
            continue
        attempt += 1
        last = parse_and_validate(result.content, config.schema, blocks)
        last.attempts = attempt
        last.quarantined = quarantined
        if last.validation.passed:
            return last
        feedback = repair_feedback(last.validation)
        if feedback is None:
            continue
        messages = messages[:2] + [
            Messages(role="assistant", content=result.content),
            Messages(role="user", content=feedback),
        ]
    return last if last is not None else local_answer("invalid", "no_attempts")


def repair_feedback(report) -> str | None:
    """Текст обратной связи для повторной попытки; ``None`` - если исправлять по сообщению нечего."""
    problems = []
    for quote in report.ungrounded_quotes:
        problems.append(f"Цитата не найдена во фрагменте дословно: «{quote[:200]}». Скопируй слова из текста фрагмента без изменений.")
    if report.fabricated_numbers:
        problems.append(f"Числа {', '.join(report.fabricated_numbers)} есть в ответе, но не в цитатах. Убери их или процитируй источник.")
    if report.unknown_ids:
        problems.append(f"Неизвестные chunk_id: {', '.join(report.unknown_ids)}.")
    problems.extend(report.business_errors)
    if not problems and report.schema_valid:
        return None
    if not report.schema_valid:
        problems.append("Ответ не соответствует схеме.")
    return "Ответ не прошёл проверку:\n- " + "\n- ".join(problems) + "\nВерни исправленный ответ целиком."


def self_consistency(
    llm: LLM, question: str, blocks: list[ContextBlock], config: PipelineConfig, *, session_id: str | None
) -> tuple[FinalAnswer, dict]:
    """Несколько независимых ответов и голосование по статусу.

    Голосует только дискретная метка ``status``; текст ответа в голосовании не
    участвует. Адаптивная остановка: если лидера уже не догнать, новые выборки
    не запрашиваются. Представитель победившей метки - валидный ответ с
    наибольшим числом цитат.
    """
    candidates: list[FinalAnswer] = []
    for index in range(config.sc_samples):
        candidate = generate_answer(
            llm, question, blocks, config, session_id=session_id, tag=f"final_sc{index + 1}", temperature=0.7
        )
        candidates.append(candidate)
        counts = Counter(c.status for c in candidates)
        ordered = counts.most_common()
        leader = ordered[0][1]
        runner_up = ordered[1][1] if len(ordered) > 1 else 0
        if leader > runner_up + (config.sc_samples - len(candidates)):
            break
    counts = Counter(c.status for c in candidates)
    winner = counts.most_common(1)[0][0]
    pool = [c for c in candidates if c.status == winner]
    best = max(pool, key=lambda c: (c.validation.passed, len(c.citations)))
    votes = {
        "statuses": [c.status for c in candidates],
        "dominant_share": counts[winner] / len(candidates),
        "samples_used": len(candidates),
    }
    return best, votes


def finalize(llm: LLM, question: str, blocks: list[ContextBlock], config: PipelineConfig,
             *, session_id: str | None, result: RunResult) -> None:
    """Финальный шаг конвейера: детерминированный отказ при пустом контексте или генерация."""
    if not blocks:
        result.final = local_answer("not_found", "empty_context")
        return
    if config.sc_samples > 1:
        result.final, result.sc_votes = self_consistency(llm, question, blocks, config, session_id=session_id)
    else:
        result.final = generate_answer(llm, question, blocks, config, session_id=session_id)
    if result.final is not None and result.final.status == "invalid" and result.error is None:
        result.error = result.final.validation.error  # причина отказа генерации видна в мете ответа


def start_timer() -> float:
    """Момент начала прогона вопроса."""
    return time.perf_counter()


def records_since(llm: LLM, start_index: int) -> list[CallRecord]:
    """Записи журнала, относящиеся к текущему вопросу."""
    return list(llm.log.since(start_index))


def as_dict(obj) -> dict:
    """dataclass -> dict для сохранения."""
    return asdict(obj)
