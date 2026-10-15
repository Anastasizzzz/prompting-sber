"""Метрики одного прогона и сводки по конфигурациям.

Принцип: **общего балла нет**. Отдельно видны решение (статус),
форма (валидность), поиск (дошёл ли нужный фрагмент), цитаты, полнота фактов,
запрещённые факты, достоверность, токены и деньги. Иначе одна цифра прячет
место поломки.

Сначала - дешёвые детерминированные проверки кодом, судья - только там,
где код уже бессилен (смысл ответа).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from ..llm import GenerationResultError
from ..pipelines.common import RunResult
from ..retrieval import Retriever
from .golden import GoldCase
from .judge import Judge


def expand_neighbors(chunk_ids: list[str], retriever: Retriever, window: int = 1) -> set[str]:
    """Эталонные фрагменты плюс соседи: цитата может лежать на границе или в перекрытии."""
    expanded: set[str] = set()
    for cid in chunk_ids:
        if cid in retriever.by_id:
            expanded.update(c.chunk_id for c in retriever.neighbors(cid, window))
    return expanded


def evaluate_run(case: GoldCase, run: RunResult, retriever: Retriever, judge: Judge | None) -> dict:
    """Все проверки одного ответа. Возвращает плоскую строку для таблиц."""
    final = run.final
    status = final.status if final else "error"
    row: dict = {
        "case_id": case.case_id,
        "set": case.set_name,
        "split": case.split,
        "slice": case.slice,
        "config_id": run.config_id,
        "expected_status": case.expected_status,
        "status": status,
        "status_correct": status == case.expected_status,
        "valid": bool(final and final.validation.passed),
        "schema_valid": bool(final and final.validation.schema_valid),
        "ungrounded_quotes": len(final.validation.ungrounded_quotes) if final else None,
        "fabricated_numbers": ",".join(final.validation.fabricated_numbers) if final else None,
        "answer": final.answer if final else None,
        "error": run.error,
        "steps": run.steps,
        "tool_calls": len(run.tool_calls),
        "tools_used": ",".join(dict.fromkeys(c["name"] for c in run.tool_calls)),
        "context_blocks": len(run.blocks),
        "cost_rub": run.cost_rub,
        "latency_s": round(run.latency_s, 2),
        **run.tokens,
    }

    # --- поиск и цитаты (только там, где у вопроса есть источник)
    if case.relevant_chunk_ids:
        per_source_ok = []
        cited_ids = {c.chunk_id for c in (final.citations if final else [])}
        cited_chunks = {cid for block in run.blocks if block.block_id in cited_ids for cid in block.chunk_ids}
        context_chunks = {cid for block in run.blocks for cid in block.chunk_ids}
        seen = set(run.seen_chunk_ids)
        for source in case.relevant_source_ids:
            source_relevant = [cid for cid in case.relevant_chunk_ids if cid.startswith(retriever.canonical_source(source))]
            relevant = expand_neighbors(source_relevant, retriever)
            per_source_ok.append(
                {
                    "retrieved": bool(relevant & seen),
                    "in_context": bool(relevant & context_chunks),
                    "cited": bool(relevant & cited_chunks),
                }
            )
        n = len(per_source_ok)
        row["retrieval_recall"] = sum(s["retrieved"] for s in per_source_ok) / n
        row["context_recall"] = sum(s["in_context"] for s in per_source_ok) / n
        row["citation_recall"] = sum(s["cited"] for s in per_source_ok) / n
        all_relevant = expand_neighbors(case.relevant_chunk_ids, retriever)
        cited_blocks = [b for b in run.blocks if b.block_id in cited_ids]
        row["citation_precision"] = (
            sum(1 for b in cited_blocks if set(b.chunk_ids) & all_relevant) / len(cited_blocks) if cited_blocks else None
        )
    else:
        for key in ("retrieval_recall", "context_recall", "citation_recall", "citation_precision"):
            row[key] = None

    # --- судья: полнота, запрещённые факты, достоверность
    row["required_fact_recall"] = None
    row["forbidden_fact_rate"] = None
    row["faithfulness"] = None
    row["judge_error"] = None
    if judge is not None and final is not None and final.status == "answered" and final.answer:
        try:
            if case.required_facts:
                hits = [judge.fact_present(f, final.answer) for f in case.required_facts]
                row["required_fact_recall"] = sum(hits) / len(hits)
                row["missing_facts"] = [f for f, h in zip(case.required_facts, hits) if not h]
            if case.forbidden_facts:
                bad = [judge.fact_present(f, final.answer, tag="judge_forbidden") for f in case.forbidden_facts]
                row["forbidden_fact_rate"] = sum(bad) / len(bad)
            cited_ids = {c.chunk_id for c in final.citations}
            # Судья видит то же, что генератор: сессию (год, тема), время, спикеров и текст.
            sources_text = "\n\n".join(
                f"[{b.session_title} | {b.time_start} | {', '.join(b.speakers)}]\n{b.text}"
                for b in run.blocks if b.block_id in cited_ids
            )
            if sources_text:
                try:
                    row["faithfulness"] = judge.faithfulness(final.answer, sources_text[:6000])
                except GenerationResultError as error:
                    if error.reason != "blacklist":
                        raise
                    # Тематический фильтр сработал на полном тексте фрагментов: судим по цитатам с заголовками.
                    quotes_text = "\n".join(f"[{c.session} | {c.time_start} | {', '.join(c.speakers)}] {c.quote}"
                                             for c in final.citations)
                    row["faithfulness"] = judge.faithfulness(final.answer, quotes_text, tag="judge_faithfulness_quotes")
                    row["judge_fallback"] = "quotes_only"
        except Exception as error:  # noqa: BLE001 - сбой судьи не равен нулю качества
            row["judge_error"] = f"{type(error).__name__}: {str(error)[:150]}"
    elif case.required_facts:
        # Ответа нет или формат нарушен: сквозная полнота - ноль (нарушение принадлежит системе).
        row["required_fact_recall"] = 0.0

    row["answer_contract_pass"] = contract_pass(case, row)
    return row


def contract_pass(case: GoldCase, row: dict) -> bool | None:
    """Принят ли ответ целиком: решение, форма, факты, цитаты, достоверность."""
    if not row["status_correct"]:
        return False
    if case.expected_status != "answered":
        return True
    if row.get("judge_error"):
        return None
    return bool(
        row["valid"]
        and (row["required_fact_recall"] is None or row["required_fact_recall"] == 1.0)
        and not row.get("forbidden_fact_rate")
        and (row["citation_recall"] is None or row["citation_recall"] > 0)
        and row.get("faithfulness") in (None, "supported")
    )


# ------------------------------------------------------------------ статистика
def mean(values: list) -> float | None:
    """Среднее без None/NaN."""
    clean = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(clean) / len(clean) if clean else None


def bootstrap_ci(values: list[float], repeats: int = 2000, seed: int = 7) -> tuple[float | None, float | None]:
    """95% интервал среднего бутстрэпом по кейсам (неопределённость набора задач)."""
    clean = [float(v) for v in values if v is not None]
    if len(clean) < 2:
        return None, None
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(clean) for _ in clean) / len(clean) for _ in range(repeats))
    return means[int(0.025 * repeats)], means[int(0.975 * repeats) - 1]


@dataclass
class Summary:
    """Сводка одной конфигурации на выборке."""

    config_id: str
    cases: int
    rows: int
    status_accuracy: float | None
    contract_pass_rate: float | None
    contract_pass_ci: tuple
    valid_rate: float | None
    required_fact_recall: float | None
    forbidden_fact_rate: float | None
    faithfulness_supported: float | None
    retrieval_recall: float | None
    context_recall: float | None
    citation_recall: float | None
    mean_total_tokens: float | None
    mean_raw_input_tokens: float | None
    mean_completion_tokens: float | None
    mean_precached_tokens: float | None
    mean_chat_calls: float | None
    mean_steps: float | None
    p50_latency_s: float | None
    p95_latency_s: float | None
    cost_per_request_rub: float | None
    accepted: int
    cost_per_accepted_rub: float | None
    total_cost_rub: float


def percentile(values: list[float], q: float) -> float | None:
    """Перцентиль без внешних зависимостей (p50/p95)."""
    clean = sorted(v for v in values if v is not None)
    if not clean:
        return None
    k = (len(clean) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return clean[lo] if lo == hi else clean[lo] + (clean[hi] - clean[lo]) * (k - lo)


def summarize(rows: list[dict], config_id: str) -> Summary:
    """Сводка по строкам одной конфигурации (все повторы вместе)."""
    passes = [1.0 if r["answer_contract_pass"] else 0.0 for r in rows if r["answer_contract_pass"] is not None]
    accepted = sum(1 for r in rows if r["answer_contract_pass"])
    total_cost = sum(r["cost_rub"] for r in rows)
    faith = [r["faithfulness"] for r in rows if r.get("faithfulness")]
    return Summary(
        config_id=config_id,
        cases=len({r["case_id"] for r in rows}),
        rows=len(rows),
        status_accuracy=mean([1.0 if r["status_correct"] else 0.0 for r in rows]),
        contract_pass_rate=mean(passes),
        contract_pass_ci=bootstrap_ci(passes),
        valid_rate=mean([1.0 if r["valid"] else 0.0 for r in rows if r["expected_status"] == "answered"]),
        required_fact_recall=mean([r["required_fact_recall"] for r in rows]),
        forbidden_fact_rate=mean([r["forbidden_fact_rate"] for r in rows]),
        faithfulness_supported=(sum(1 for f in faith if f == "supported") / len(faith)) if faith else None,
        retrieval_recall=mean([r["retrieval_recall"] for r in rows]),
        context_recall=mean([r["context_recall"] for r in rows]),
        citation_recall=mean([r["citation_recall"] for r in rows]),
        mean_total_tokens=mean([r["total_tokens"] for r in rows]),
        mean_raw_input_tokens=mean([r["raw_input_tokens"] for r in rows]),
        mean_completion_tokens=mean([r["completion_tokens"] for r in rows]),
        mean_precached_tokens=mean([r["precached_prompt_tokens"] for r in rows]),
        mean_chat_calls=mean([r["chat_calls"] for r in rows]),
        mean_steps=mean([r["steps"] for r in rows]),
        p50_latency_s=percentile([r["latency_s"] for r in rows], 0.5),
        p95_latency_s=percentile([r["latency_s"] for r in rows], 0.95),
        cost_per_request_rub=mean([r["cost_rub"] for r in rows]),
        accepted=accepted,
        cost_per_accepted_rub=(total_cost / accepted) if accepted else None,
        total_cost_rub=round(total_cost, 4),
    )


def paired_regressions(rows_a: list[dict], rows_b: list[dict]) -> list[dict]:
    """Что починилось и что сломалось по кейсам между двумя конфигурациями."""
    def by_case(rows: list[dict]) -> dict[str, float]:
        """Доля принятых ответов по каждому кейсу (усреднение по повторам)."""
        grouped: dict[str, list[float]] = {}
        for r in rows:
            if r["answer_contract_pass"] is not None:
                grouped.setdefault(r["case_id"], []).append(1.0 if r["answer_contract_pass"] else 0.0)
        return {k: sum(v) / len(v) for k, v in grouped.items()}

    a, b = by_case(rows_a), by_case(rows_b)
    changes = []
    for case_id in sorted(set(a) & set(b)):
        if a[case_id] != b[case_id]:
            changes.append({
                "case_id": case_id, "было": round(a[case_id], 2), "стало": round(b[case_id], 2),
                "направление": "починилось" if b[case_id] > a[case_id] else "сломалось",
            })
    return changes
