"""Командная строка проекта. Запуск через ``./run.sh <команда>``.

Команды:
  check        проверка ключа, доступа к моделям, счётчика токенов
  data         обзор данных: корпус, дубли, золотой набор (без вызовов модели)
  index        построение индекса по raw_data/SPIEF_txt (один раз)
  ask          вопрос агенту: вызовы функций + структурированный ответ + токены и ₽
  eval         прогон конфигурации на золотом наборе
  experiments  цепочка оптимизации и дополнительные эксперименты
  report       results/REPORT.md из сохранённых прогонов
  demo         демо-сценарий для скринкаста
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from collections import Counter

from .settings import ensure_dirs


def _line(char: str = "─", width: int = 100) -> None:
    """Горизонтальный разделитель в выводе терминала."""
    print(char * width)


def print_run(result, *, show_trace: bool = True) -> None:
    """Печатает вызовы функций поиска и структурированный ответ."""
    if show_trace and result.tool_calls:
        print("Вызовы функций поиска (function calling):")
        for i, call in enumerate(result.tool_calls, 1):
            args = json.dumps(call.get("arguments"), ensure_ascii=False)
            outcome = call.get("error") or f"найдено {call.get('count')}: {', '.join((call.get('result_ids') or [])[:3])}"
            print(f"  {i}. {call['name']}({args}) -> {outcome}")
    if result.guard and not result.guard.allowed:
        print(f"Guardrail на входе: {result.guard.reason} ({result.guard.matched})")
    print("Структурированный ответ (фиксированный формат):")
    print(textwrap.indent(json.dumps(result.to_public(), ensure_ascii=False, indent=2), "  "))


def cmd_check(args) -> None:
    """Ключ, доступ к моделям (402), tokens_count и коэффициент «символов на токен»."""
    from .experiments import probe_models
    from .index import prepare_chunks
    from .llm import LLM

    llm = LLM()
    print(f"Ключ найден, scope={llm.settings.scope}, base_url={llm.settings.base_url}, проверка TLS={llm.settings.verify_ssl_certs}")
    for row in probe_models(llm):
        print(f"  {row['model']:16} {row['level']:4} -> {row['status']}")
    _, chunks = prepare_chunks()
    sample = [c.embedding_text() for c in chunks[::max(len(chunks) // 40, 1)][:40]]
    counts, method = llm.tokens_count(sample)
    ratio = sum(len(t) for t in sample) / max(sum(counts), 1)
    print(f"tokens_count: способ={method}; на корпусе ПМЭФ ≈ {ratio:.2f} символа на токен")
    print(f"Потрачено на проверку: {llm.budget.spent:.4f} ₽")


def cmd_data(args) -> None:
    """Обзор данных без вызовов модели (демо, шаг 1)."""
    from .evaluation.golden import load_golden
    from .index import estimate_index_cost, prepare_chunks

    corpus, chunks = prepare_chunks()
    sessions = corpus.sessions
    canonical = corpus.canonical_ids
    print(f"Корпус raw_data/SPIEF_txt: {len(sessions)} файлов, уникальных сессий {len(canonical)}, дублей {len(corpus.duplicates)}")
    for dup, keep in sorted(corpus.duplicates.items())[:5]:
        print(f"  дубль {dup} -> {keep}")
    print(f"  ... всего {len(corpus.duplicates)}")
    years = Counter(sessions[s].year for s in canonical)
    langs = Counter(sessions[s].language for s in canonical)
    print(f"По годам: {dict(sorted(years.items()))}; язык: {dict(langs)}")
    longest = max((r for s in canonical for r in corpus.replicas[s]), key=lambda r: len(r.text))
    print(f"Самая длинная «минута»: {len(longest.text)} символов ({longest.source_id} {longest.time_start})")
    est = estimate_index_cost(chunks)
    print(f"Фрагментов: {est['chunks']}, индексация ≈ {est['estimated_cost_rub']} ₽ один раз")
    cases = load_golden(chunks, corpus.duplicates)
    core = [c for c in cases if c.set_name == "core"]
    print(f"\nЗолотой набор: ядро {len(core)} вопросов из xlsx + расширение {len(cases) - len(core)}")
    print("  срезы:", dict(Counter(c.slice for c in cases)))
    minutes = Counter(c.sources[0].time_start for c in core)
    print(f"  таймкоды ответов ядра: {dict(minutes)} - все ответы во вступлениях сессий (перекос набора)")


def _load_config(config_id: str):
    """Конфигурация по имени: FINAL, шаг сохранённой цепочки или статическая из config.py."""
    from .config import OPTIMIZATION_CHAIN, RAG_BASELINE, RAG_OPT
    from .experiments import final_config, load_json

    if config_id.upper() == "FINAL":
        return final_config()
    chain = load_json("chain", "chain.json") or []
    from .config import PipelineConfig

    for item in chain:
        if item["config"]["config_id"] == config_id:
            return PipelineConfig(**item["config"])
    for config in OPTIMIZATION_CHAIN + [RAG_BASELINE, RAG_OPT]:
        if config.config_id == config_id:
            return config
    raise SystemExit(f"Нет конфигурации {config_id}")


def cmd_index(args) -> None:
    """Строит индекс (один раз; повторный запуск ничего не оплачивает)."""
    from .index import build_index
    from .llm import LLM, CallLog
    from .settings import RUNS_DIR

    llm = LLM(log=CallLog(RUNS_DIR / "calls.jsonl"))
    manifest = build_index(llm)
    print(json.dumps(manifest, ensure_ascii=False, indent=1))


def cmd_ask(args) -> None:
    """Один вопрос: вызовы функций, ответ, токены и рубли."""
    from .cache import ResultCache
    from .llm import LLM, CallLog
    from .pipelines import answer_question
    from .retrieval import load_retriever
    from .settings import RUNS_DIR

    config = _load_config(args.config)
    llm = LLM(log=CallLog(RUNS_DIR / "calls.jsonl"))
    retriever = load_retriever(llm)
    cache = ResultCache() if args.cache else None
    if cache is not None:
        cached = cache.get(args.question, config)
        if cached is not None:
            print("Кеш готового результата: ПОПАДАНИЕ - модель не вызывалась, 0 ₽")
            print(json.dumps(cached, ensure_ascii=False, indent=2))
            return
    result = answer_question(args.question, config, llm, retriever)
    print(f"Конфигурация {config.config_id}: {config.name}")
    print_run(result, show_trace=not args.no_trace)
    if cache is not None and result.final and result.final.validation.passed:
        cache.put(args.question, config, result.to_public())


def cmd_eval(args) -> None:
    """Прогон конфигурации на выборке золотого набора с метриками."""
    from dataclasses import asdict

    from .evaluation.golden import select
    from .evaluation.metrics import summarize
    from .evaluation.runner import run_config
    from .experiments import make_context

    ctx = make_context(args.budget)
    config = _load_config(args.config)
    rows = run_config(config, select(ctx.cases, args.cases), llm=ctx.llm, retriever=ctx.retriever,
                      judge=None if args.no_judge else ctx.judge, experiment=args.experiment, repeats=args.repeats)
    print(json.dumps(asdict(summarize(rows, config.config_id)), ensure_ascii=False, indent=1, default=str))


def cmd_experiments(args) -> None:
    """Цепочка оптимизации и эксперименты E1–E13."""
    from .experiments import run_experiments

    run_experiments(args.names or None, budget_rub=args.budget)


def cmd_report(args) -> None:
    """Собирает отчёт."""
    from .report import build_report

    print(f"Отчёт: {build_report()}")


def cmd_demo(args) -> None:
    """Демо-сценарий для скринкаста."""
    from .demo import run_demo

    run_demo(pause=not args.no_pause, steps=args.steps)


def main(argv: list[str] | None = None) -> None:
    """Точка входа."""
    ensure_dirs()
    parser = argparse.ArgumentParser(prog="agent", description="RAG-агент по расшифровкам ПМЭФ на GigaChat")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check").set_defaults(func=cmd_check)
    sub.add_parser("data").set_defaults(func=cmd_data)
    sub.add_parser("index").set_defaults(func=cmd_index)
    ask = sub.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--config", default="FINAL")
    ask.add_argument("--cache", action="store_true", help="кеш готового результата")
    ask.add_argument("--no-trace", action="store_true")
    ask.set_defaults(func=cmd_ask)
    ev = sub.add_parser("eval")
    ev.add_argument("--config", default="FINAL")
    ev.add_argument("--cases", default="core", help="core | test | calibration | extended | all | G01,X01")
    ev.add_argument("--repeats", type=int, default=1)
    ev.add_argument("--experiment", default="manual")
    ev.add_argument("--budget", type=float, default=None, help="лимит ₽ на прогон")
    ev.add_argument("--no-judge", action="store_true")
    ev.set_defaults(func=cmd_eval)
    ex = sub.add_parser("experiments")
    ex.add_argument("names", nargs="*")
    ex.add_argument("--budget", type=float, default=None)
    ex.set_defaults(func=cmd_experiments)
    sub.add_parser("report").set_defaults(func=cmd_report)
    demo = sub.add_parser("demo")
    demo.add_argument("--no-pause", action="store_true")
    demo.add_argument("--steps", default="all", help="номера шагов через запятую")
    demo.set_defaults(func=cmd_demo)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
