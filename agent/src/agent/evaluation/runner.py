"""Прогон конфигураций на золотом наборе: воспроизводимо, с журналом и продолжением.

* каждая пара «конфигурация × вопрос × повтор» сохраняется строкой в
  ``results/runs/<experiment>/<config>.jsonl`` вместе с трассой: прервался прогон -
  повторный запуск пропустит готовое и не оплатит его второй раз;
* порядок вопросов чередуется между конфигурациями (AB / BA);
* у каждого вызова новая сессия, чтобы кеш префикса не искажал сравнение
  (кроме конфигураций, которые как раз меряют кеш);
* перед запуском печатается одна строка с ожидаемой стоимостью.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from ..config import PipelineConfig
from ..llm import LLM
from ..pipelines import answer_question
from ..retrieval import Retriever
from ..settings import RUNS_DIR
from .golden import GoldCase
from .judge import Judge
from .metrics import evaluate_run

# Грубая оценка стоимости одного вопроса (₽) по архитектуре и объёму контекста.
# Уточняется фактом: после первого прогона берётся средняя стоимость из файлов.
# Откалибровано по живому baseline A (≈3,8 ₽ за вопрос вместе с судьёй); остальные - пропорционально.
COST_GUESS_RUB = {"agent_verbose": 3.8, "agent_compact": 1.8, "rag_verbose": 2.5, "rag_compact": 1.0, "sgr_agent": 3.0}


def guess_cost(config: PipelineConfig) -> float:
    """Априорная оценка стоимости вопроса для строки «ожидается ≈ N ₽»."""
    if config.architecture == "sgr_agent":
        base = COST_GUESS_RUB["sgr_agent"]
    else:
        heavy = config.finalize_from == "history" or config.context_format == "verbose"
        base = COST_GUESS_RUB[f"{config.architecture}_{'verbose' if heavy else 'compact'}"]
    return base * max(config.sc_samples, 1)


def run_dir(experiment: str) -> Path:
    """Каталог прогона эксперимента."""
    path = RUNS_DIR / experiment
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_rows(experiment: str, config_id: str) -> list[dict]:
    """Строки уже выполненного прогона конфигурации."""
    path = run_dir(experiment) / f"{config_id}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line)["row"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_config(
    config: PipelineConfig,
    cases: list[GoldCase],
    *,
    llm: LLM,
    retriever: Retriever,
    judge: Judge | None,
    experiment: str,
    repeats: int = 1,
    reverse_order: bool = False,
    quiet: bool = False,
) -> list[dict]:
    """Прогоняет конфигурацию на кейсах и возвращает строки метрик (включая уже готовые).

    Продолжение прерванного прогона безопасно, только если на диске результаты **той же**
    конфигурации: в цепочке шаг с тем же именем может строиться на другой базе. Поэтому
    конфигурация записывается в ``configs.json`` до первого вызова, а файл с другим
    (или неизвестным) отпечатком откладывается в ``*.stale-*.jsonl`` и прогон начинается заново.
    """
    path = run_dir(experiment) / f"{config.config_id}.jsonl"
    config_path = run_dir(experiment) / "configs.json"
    configs = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    stored = configs.get(config.config_id)
    if path.exists() and (stored is None or _fingerprint(stored) != _fingerprint(config.to_dict())):
        stale = path.with_name(f"{config.config_id}.stale-{datetime.now():%Y%m%d-%H%M%S}.jsonl")
        path.rename(stale)
        if not quiet:
            print(f"[{experiment}] {config.config_id}: на диске результаты другой конфигурации -> {stale.name}")
    configs[config.config_id] = config.to_dict()
    config_path.write_text(json.dumps(configs, ensure_ascii=False, indent=1), encoding="utf-8")
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                done.add((record["row"]["case_id"], record["row"]["repeat"]))
    ordered = list(reversed(cases)) if reverse_order else list(cases)
    todo = [(case, rep) for rep in range(1, repeats + 1) for case in ordered if (case.case_id, rep) not in done]
    if todo and not quiet:
        print(
            f"[{experiment}] {config.config_id}: {len(todo)} прогонов, ожидается ≈ "
            f"{guess_cost(config) * len(todo):.0f} ₽ (оценка), уже готово {len(done)}"
        )
    for index, (case, rep) in enumerate(todo, 1):
        result = answer_question(case.question, config, llm, retriever)
        row = evaluate_run(case, result, retriever, judge)
        row["repeat"] = rep
        record = {
            "row": row,
            "public": result.to_public(),
            "tool_calls": result.tool_calls,
            "blocks": [asdict(b) for b in result.blocks],
            "retrieval_traces": [asdict(t) for t in result.retrieval_traces],
            "records": [asdict(r) for r in result.records],
            "sc_votes": result.sc_votes,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        if not quiet:
            mark = "✓" if row["answer_contract_pass"] else ("?" if row["answer_contract_pass"] is None else "✗")
            print(
                f"  {mark} {config.config_id} {case.case_id} r{rep} [{index}/{len(todo)}] status={row['status']} "
                f"tokens={row['total_tokens']} cost={row['cost_rub']:.3f}₽ {('ERR ' + row['error']) if row['error'] else ''}"
            )
    return load_rows(experiment, config.config_id)


def _fingerprint(config: dict) -> str:
    """Отпечаток конфигурации без человекочитаемого названия."""
    return json.dumps({k: v for k, v in config.items() if k != "name"}, sort_keys=True, ensure_ascii=False)
