"""Основной конвейер: ReAct-агент поверх нативного function calling GigaChat.

Цикл: модель получает инструкцию, описания функций и историю, возвращает
``function_call`` -> приложение проверяет имя и аргументы, исполняет функцию
(только чтение) -> результат возвращается модели парой
«сообщение с вызовом + сообщение роли function». Цикл заканчивается, когда
модель отвечает текстом, или по предохранителям: лимит шагов, повтор
одинакового вызова.

Финальный ответ - отдельный структурированный вызов со строгой схемой:

* ``finalize_from="history"`` (baseline A): в контекст идут все результаты всех
  вызовов функций подряд, как они лежат в истории;
* ``finalize_from="state"``: в контекст идёт **состояние** - уникальные найденные
  фрагменты, отобранные и склеенные (подход «явное состояние вместо истории»). У агента «платит вход», поэтому это главный рычаг экономии.
"""

from __future__ import annotations

import json
from datetime import datetime

from gigachat.models import Messages

from ..config import PipelineConfig
from ..guardrails import check_input
from ..llm import LLM, GenerationResultError, proxy_units
from ..prompts import COMPACT_AGENT_SYSTEM, VERBOSE_AGENT_SYSTEM
from ..retrieval import ContextBlock, Hit, Retriever, RetrievalTrace
from ..tools import ToolContext, dispatch, tool_specs
from .common import RunResult, finalize, local_answer, records_since, start_timer

COMPACTION_TRIGGER_UNITS = 2500
NO_EVIDENCE_NUDGE = (
    "Фрагменты расшифровок ещё не найдены: каталог сессий текстов не содержит. "
    "Вызови search_transcripts, keyword_search или get_fragment(opening_chunk_id), затем заверши."
)


def agent_system_prompt(config: PipelineConfig) -> str:
    """Системная инструкция агента; метка времени в начале - только как антипример."""
    text = VERBOSE_AGENT_SYSTEM if config.agent_prompt == "verbose" else COMPACT_AGENT_SYSTEM
    if config.timestamp_in_prefix:
        text = f"Текущее время: {datetime.now().isoformat(timespec='microseconds')}\n{text}"
    return text


def compact_messages(messages: list[Messages], strategy: str, keep_last_pairs: int = 2) -> list[Messages]:
    """Сжатие истории агента без разрыва протокола.

    * ``truncate`` - оставить системное сообщение, вопрос и последние пары дословно;
    * ``offload`` - старые результаты заменить ссылками: chunk_id и count.
      Обратимо: данные можно снова получить через ``get_fragment``.

    Пара «вызов функции + результат» никогда не разрывается.
    """
    head, tail = messages[:2], messages[2:]
    pairs = [tail[i : i + 2] for i in range(0, len(tail) - 1, 2)]
    if len(pairs) <= keep_last_pairs:
        return messages
    old, recent = pairs[:-keep_last_pairs], pairs[-keep_last_pairs:]
    if strategy == "truncate":
        return head + [m for pair in recent for m in pair]
    if strategy == "offload":
        lines = []
        for call_msg, result_msg in old:
            try:
                payload = json.loads(result_msg.content)
            except json.JSONDecodeError:
                payload = {}
            ids = [item.get("chunk_id") for item in payload.get("items", [])]
            name = call_msg.function_call.name if call_msg.function_call else result_msg.name
            args = call_msg.function_call.arguments if call_msg.function_call else {}
            lines.append(f"{name}({json.dumps(args, ensure_ascii=False)}) -> ids={ids}, count={payload.get('count')}")
        summary = Messages(
            role="user",
            content="<offloaded_results>\n" + "\n".join(lines)
            + "\nПолные тексты доступны через get_fragment(chunk_id).\n</offloaded_results>",
        )
        return head + [summary] + [m for pair in recent for m in pair]
    raise ValueError(f"Неизвестная стратегия сжатия: {strategy}")


def history_blocks(retriever: Retriever, ctx: ToolContext) -> list[ContextBlock]:
    """Контекст «как в истории»: все результаты всех вызовов подряд, с повторами."""
    blocks = []
    for call in ctx.calls:
        for chunk_id in call.get("result_ids") or []:
            chunk = retriever.chunk(chunk_id)
            text = chunk.text if len(chunk.text) <= ctx.item_chars else chunk.text[: ctx.item_chars] + "…"
            blocks.append(
                ContextBlock(
                    chunk_ids=[chunk.chunk_id], source_id=chunk.source_id, session_title=chunk.session_title,
                    time_start=chunk.time_start, time_end=chunk.time_end, speakers=chunk.speakers,
                    text=text, score=ctx.seen_chunks.get(chunk_id, 0.0),
                )
            )
    return blocks


def state_blocks(retriever: Retriever, ctx: ToolContext, config: PipelineConfig,
                 trace: RetrievalTrace, question: str) -> list[ContextBlock]:
    """Контекст из состояния: уникальные найденные фрагменты, порог, склейка соседей, лимит окон.

    Все фрагменты переоцениваются косинусом с вопросом (единая шкала: BM25 и
    косинус несравнимы). Явно запрошенные через get_fragment идут первыми и
    порогом не отсекаются.
    """
    scores, tokens = retriever.question_scores(question, list(ctx.seen_chunks))
    ctx.embedding_tokens += tokens
    order = sorted(scores.items(), key=lambda kv: (kv[0] not in ctx.pinned, -kv[1]))
    hits = [Hit(retriever.chunk(cid), score, dense_score=score, rank=rank) for rank, (cid, score) in enumerate(order, 1)]
    return retriever.build_blocks(
        hits, min_score=config.min_score, max_blocks=config.max_blocks,
        merge_neighbors=config.merge_neighbors, trace=trace, pinned=ctx.pinned,
    )


def run_agent(question: str, config: PipelineConfig, llm: LLM, retriever: Retriever) -> RunResult:
    """Прогон одного вопроса через ReAct-агента."""
    started = start_timer()
    log_start = len(llm.log.records)
    result = RunResult(question=question, config_id=config.config_id, final=None)
    result.guard = check_input(question)
    if not result.guard.allowed:
        result.final = local_answer("refused", f"guardrail:{result.guard.reason}")
        result.latency_s = start_timer() - started
        return result

    ctx = ToolContext(retriever=retriever, item_chars=config.item_chars)
    specs = tool_specs(config.tool_variant)
    shared_session = llm.new_session_id("agent") if config.session_mode == "shared" else None
    messages = [
        Messages(role="system", content=agent_system_prompt(config)),
        Messages(role="user", content=question),
    ]
    seen_calls: set[str] = set()
    nudged = False
    try:
        for step in range(1, config.max_steps + 1):
            result.steps = step
            reply = llm.chat(
                messages, tag=f"agent_step{step}", model=config.model, temperature=0.01,
                max_tokens=config.agent_max_tokens, functions=specs, function_call="auto",
                session_id=shared_session or llm.new_session_id("agent"),
            )
            call = reply.message.function_call
            if call is None:
                # Предохранитель: завершение без единого найденного фрагмента - один раз напомнить.
                if not ctx.seen_chunks and not nudged and step < config.max_steps:
                    nudged = True
                    messages.append(reply.message)
                    messages.append(Messages(role="user", content=NO_EVIDENCE_NUDGE))
                    continue
                break
            key = f"{call.name}:{json.dumps(call.arguments, ensure_ascii=False, sort_keys=True)}"
            if key in seen_calls:
                tool_result = {"error": "Этот вызов с теми же аргументами уже был. Используй другой запрос или заверши поиск."}
            else:
                seen_calls.add(key)
                tool_result = dispatch(ctx, call.name, call.arguments)
            messages.append(reply.message)
            messages.append(
                Messages(role="function", name=call.name, content=json.dumps(tool_result, ensure_ascii=False))
            )
            if config.compaction and proxy_units([m.content for m in messages]) > COMPACTION_TRIGGER_UNITS:
                messages = compact_messages(messages, config.compaction)

        result.tool_calls = ctx.calls
        result.seen_chunk_ids = list(ctx.seen_chunks)
        trace = RetrievalTrace(query=question, mode="agent", year_filter=None)
        if config.finalize_from == "history":
            result.blocks = history_blocks(retriever, ctx)
        else:
            result.blocks = state_blocks(retriever, ctx, config, trace, question)
        result.retrieval_traces.append(trace)
        finalize(llm, question, result.blocks, config,
                 session_id=shared_session or llm.new_session_id("final"), result=result)
    except GenerationResultError as error:
        result.error = f"generation_{error.reason}: {error}"
    except Exception as error:  # noqa: BLE001 - сбой фиксируется, прогон набора продолжается
        result.error = f"{type(error).__name__}: {str(error)[:300]}"
    finally:
        result.records = records_since(llm, log_start)
        result.latency_s = start_timer() - started
        if result.tool_calls == [] and ctx.calls:
            result.tool_calls = ctx.calls
    return result
