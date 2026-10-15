"""Обёртка над GigaChat SDK: вызов, учёт токенов и денег, журнал, бюджет.

Что здесь собрано из курса:

* ``require_complete`` - дальше идёт только ответ с ``finish_reason="stop"``;
* у каждого вызова сохраняются все поля ``usage``, задержка и стоимость;
* оплаченный, но отклонённый ответ тоже попадает в журнал: брак оплачен;
* ``X-Session-ID`` задаётся явно: в замерах - новая сессия на кейс, в опыте с
  кешем префикса - общая;
* бюджет проверяется до вызова, а не после;
* ``tokens_count`` поднимается отдельным клиентом на рабочем хосте и имеет
  запасной путь - локальную оценку.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterator

import gigachat.context
from gigachat import GigaChat
from gigachat.models import Chat, Function, Messages
from gigachat.models.chat import ChatFunctionCall

from .settings import GigaChatSettings
from .tariffs import normalize_model_name, tokens_cost_rub

warnings.filterwarnings("ignore", message=".*verify_ssl_certs.*")
warnings.filterwarnings("ignore", category=Warning, module="urllib3")


class GenerationResultError(RuntimeError):
    """Ответ API нельзя использовать как завершённый результат."""

    def __init__(self, reason: str, message: str) -> None:
        """``reason`` - машинный код причины (например, ``length`` или ``blacklist``)."""
        self.reason = reason
        super().__init__(message)


class BudgetExceededError(RuntimeError):
    """Прогон остановлен до превышения бюджета."""


@dataclass
class CallRecord:
    """Одна запись журнала вызовов: всё, что нужно для расчёта стоимости и разбора."""

    tag: str
    kind: str  # chat | embeddings | tokens_count
    model: str
    session_id: str | None
    latency_s: float
    finish_reason: str | None
    prompt_tokens: int
    precached_prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_rub: float
    status: str = "ok"  # ok | rejected | error
    error: str | None = None
    ts: float = field(default_factory=time.time)

    @property
    def raw_input_tokens(self) -> int:
        """Полная длина входа до учёта кеша."""
        return self.prompt_tokens + self.precached_prompt_tokens


@dataclass
class ChatResult:
    """Результат вызова чата: сообщение модели и запись журнала."""

    message: Messages
    finish_reason: str | None
    record: CallRecord

    @property
    def content(self) -> str:
        """Текст ответа (пустая строка, если модель вернула вызов функции)."""
        return self.message.content or ""


class Budget:
    """Лимит расходов на один прогон с уровнями оповещения 50/80/95%."""

    LEVELS = ((0.95, "критический порог"), (0.80, "высокое потребление"), (0.50, "половина бюджета"))

    def __init__(self, limit_rub: float | None) -> None:
        """``limit_rub`` - лимит расходов на прогон; ``None`` - без лимита."""
        self.limit = Decimal(str(limit_rub)) if limit_rub else None
        self.spent = Decimal(0)
        self._announced: set[str] = set()
        self._lock = threading.Lock()

    def check(self) -> None:
        """Вызывается до запроса: при исчерпании бюджета запрос не отправляется."""
        if self.limit is not None and self.spent >= self.limit:
            raise BudgetExceededError(
                f"Бюджет прогона исчерпан: потрачено {self.spent:.2f} ₽ из {self.limit:.2f} ₽"
            )

    def add(self, cost: Decimal) -> None:
        """Учитывает фактический расход и печатает пересечение уровней квоты."""
        with self._lock:
            self.spent += cost
            if self.limit is None:
                return
            share = float(self.spent / self.limit)
            for level, label in self.LEVELS:
                if share >= level and label not in self._announced:
                    self._announced.add(label)
                    print(f"[бюджет] {label}: {self.spent:.2f} ₽ из {self.limit:.2f} ₽")
                    break


class CallLog:
    """Журнал всех платных вызовов. Пишется в память и, при наличии пути, в JSONL."""

    def __init__(self, path: Path | None = None) -> None:
        """Журнал в памяти; если задан ``path``, записи дублируются в JSONL."""
        self.records: list[CallRecord] = []
        self.path = path
        self._lock = threading.Lock()

    def add(self, record: CallRecord) -> None:
        """Добавляет запись и дописывает её в файл журнала."""
        with self._lock:
            self.records.append(record)
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def total_cost(self, prefix: str = "") -> float:
        """Сумма в рублях по записям, чей тег начинается с ``prefix``."""
        return sum(r.cost_rub for r in self.records if r.tag.startswith(prefix))

    def since(self, index: int) -> list[CallRecord]:
        """Записи, добавленные после позиции ``index`` (для подсчёта одного вопроса)."""
        return self.records[index:]


def usage_dict(response: Any) -> dict[str, int]:
    """Поля usage ответа SDK в обычном словаре; отсутствующий кеш = 0."""
    usage = getattr(response, "usage", None)
    if usage is None:
        raise GenerationResultError("missing_usage", "GigaChat не вернул usage")
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "precached_prompt_tokens": int(getattr(usage, "precached_prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def require_complete(finish_reason: str | None, *, allow_function_call: bool = False) -> None:
    """Пропускает только завершённую генерацию; остальные исходы - явные ошибки."""
    if finish_reason == "stop":
        return
    if finish_reason == "function_call" and allow_function_call:
        return
    messages = {
        "length": "Ответ оборван по max_tokens: JSON может быть не закрыт.",
        "blacklist": "Запрос остановлен тематическим фильтром.",
        "function_call": "Модель запросила функцию, а этот шаг функции не исполняет.",
        "error": "Генерация завершилась с ошибкой.",
        None: "В ответе нет finish_reason.",
    }
    code = "missing_finish_reason" if finish_reason is None else str(finish_reason)
    raise GenerationResultError(code, messages.get(finish_reason, f"Неизвестный finish_reason={finish_reason!r}"))


class LLM:
    """Клиент GigaChat проекта.

    Один экземпляр на прогон: у него свой журнал вызовов и свой бюджет.
    """

    RETRYABLE = ("timeout", "ratelimit", "connect", "readerror", "remoteprotocol", "servererror")

    def __init__(
        self,
        settings: GigaChatSettings | None = None,
        *,
        log: CallLog | None = None,
        budget: Budget | None = None,
    ) -> None:
        """Создаёт клиент GigaChat с журналом вызовов и бюджетом прогона."""
        self.settings = settings or GigaChatSettings()
        if not self.settings.has_credentials:
            raise RuntimeError(
                "Не найден GIGACHAT_CREDENTIALS. Положите ключ в .env (корень репозитория или agent/.env)."
            )
        self.log = log or CallLog()
        self.budget = budget or Budget(None)
        self.client = GigaChat(
            base_url=self.settings.base_url,
            credentials=self.settings.credentials,
            scope=self.settings.scope,
            verify_ssl_certs=self.settings.verify_ssl_certs,
            timeout=self.settings.timeout_s,
            max_retries=0,
        )
        self._counter_client: GigaChat | None = None
        self.tokens_count_available: bool | None = None

    # ------------------------------------------------------------------ сессии
    @staticmethod
    def new_session_id(prefix: str = "spief") -> str:
        """Новый идентификатор сессии (новая сессия - без чужого кеша префикса)."""
        return f"{prefix}-{uuid.uuid4()}"

    @staticmethod
    @contextmanager
    def session(session_id: str | None) -> Iterator[None]:
        """Контекст, в котором все вызовы идут с заголовком ``X-Session-ID``."""
        if session_id is None:
            yield
            return
        token = gigachat.context.session_id_cvar.set(session_id)
        try:
            yield
        finally:
            gigachat.context.session_id_cvar.reset(token)

    # ------------------------------------------------------------------ вызовы
    def _with_retries(self, fn, *, attempts: int = 4):
        """Повтор при сетевых сбоях и 429 с экспоненциальной паузой (карта сбоев)."""
        last: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return fn()
            except Exception as error:  # noqa: BLE001 - разбираем по имени класса ниже
                name = type(error).__name__.lower()
                text = str(error).lower()
                retryable = any(key in name for key in self.RETRYABLE) or " 429 " in f" {text} " or "429" in text[:10]
                if not retryable or attempt == attempts:
                    raise
                last = error
                time.sleep(min(2 ** attempt, 20))
        raise last  # pragma: no cover

    def chat(
        self,
        messages: list[Messages | dict],
        *,
        tag: str,
        model: str | None = None,
        temperature: float | None = 0.01,
        top_p: float | None = None,
        max_tokens: int = 512,
        response_format: dict | None = None,
        functions: list[dict] | None = None,
        function_call: str | dict | None = None,
        reasoning_effort: str | None = None,
        session_id: str | None = None,
        allow_function_call: bool = False,
    ) -> ChatResult:
        """Один вызов чата с полным учётом расхода.

        ``function_call`` - ``"auto"``, ``"none"`` или ``{"name": ...}`` для
        принудительного вызова функции (режим ``forced``).
        """
        self.budget.check()
        model = model or self.settings.model
        payload: dict[str, Any] = {
            "model": model,
            "messages": [m if isinstance(m, Messages) else Messages(**m) for m in messages],
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if response_format is not None:
            payload["response_format"] = response_format
        if functions:
            payload["functions"] = [Function(**spec) for spec in functions]
            if isinstance(function_call, dict):
                payload["function_call"] = ChatFunctionCall(name=function_call["name"])
            else:
                payload["function_call"] = function_call or "auto"
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        request = Chat(**payload)

        started = time.perf_counter()
        try:
            with self.session(session_id):
                response = self._with_retries(lambda: self.client.chat(request))
        except Exception as error:
            record = CallRecord(
                tag=tag, kind="chat", model=model, session_id=session_id,
                latency_s=round(time.perf_counter() - started, 3), finish_reason=None,
                prompt_tokens=0, precached_prompt_tokens=0, completion_tokens=0, total_tokens=0,
                cost_rub=0.0, status="error", error=f"{type(error).__name__}: {str(error)[:300]}",
            )
            self.log.add(record)
            raise
        latency = time.perf_counter() - started
        usage = usage_dict(response)
        returned_model = normalize_model_name(getattr(response, "model", None), default=model)
        cost = tokens_cost_rub(usage["total_tokens"], returned_model)
        self.budget.add(cost)
        choice = response.choices[0]
        record = CallRecord(
            tag=tag, kind="chat", model=returned_model, session_id=session_id,
            latency_s=round(latency, 3), finish_reason=choice.finish_reason,
            cost_rub=float(cost), **usage,
        )
        self.log.add(record)
        require_complete(choice.finish_reason, allow_function_call=allow_function_call or bool(functions))
        return ChatResult(message=choice.message, finish_reason=choice.finish_reason, record=record)

    def embed(self, texts: list[str], *, tag: str, batch_size: int = 32) -> tuple[list[list[float]], int]:
        """Векторизует тексты батчами. Возвращает векторы в исходном порядке и число токенов."""
        model = self.settings.embedding_model
        vectors: list[list[float]] = []
        total_tokens = 0
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            self.budget.check()
            started = time.perf_counter()
            response = self._with_retries(lambda: self.client.embeddings(batch, model=model))
            items = sorted(response.data, key=lambda item: item.index)
            if len(items) != len(batch):
                raise RuntimeError("Число эмбеддингов не совпало с числом текстов")
            tokens = sum(int(getattr(getattr(i, "usage", None), "prompt_tokens", 0) or 0) for i in items)
            cost = tokens_cost_rub(tokens, model)
            self.budget.add(cost)
            self.log.add(
                CallRecord(
                    tag=tag, kind="embeddings", model=model, session_id=None,
                    latency_s=round(time.perf_counter() - started, 3), finish_reason=None,
                    prompt_tokens=tokens, precached_prompt_tokens=0, completion_tokens=0,
                    total_tokens=tokens, cost_rub=float(cost),
                )
            )
            vectors.extend(item.embedding for item in items)
            total_tokens += tokens
        return vectors, total_tokens

    def tokens_count(self, texts: list[str], *, model: str | None = None) -> tuple[list[int], str]:
        """Предварительный счётчик: API на рабочем хосте, иначе оценка по символам.

        Возвращает числа и способ, которым они получены (``api`` / ``estimate``),
        чтобы оценка не смешивалась с фактом. Годится для сравнения
        вариантов между собой, но не для расчёта счёта.
        """
        model = model or self.settings.model
        if self.tokens_count_available is not False:
            try:
                if self._counter_client is None:
                    self._counter_client = GigaChat(
                        base_url=self.settings.counter_base_url,
                        credentials=self.settings.credentials,
                        scope=self.settings.scope,
                        verify_ssl_certs=self.settings.verify_ssl_certs,
                        timeout=60,
                        max_retries=0,
                    )
                result = self._counter_client.tokens_count(texts, model=model)
                self.tokens_count_available = True
                return [int(item.tokens) for item in result], "api"
            except Exception:  # noqa: BLE001
                self.tokens_count_available = False
        return [estimate_tokens(text) for text in texts], "estimate"


# Коэффициент «символов на токен» для запасной оценки. Значение уточняется командой
# ./run.sh check на корпусе ПМЭФ (коэффициент калибруют на своих данных).
CHARS_PER_TOKEN = 3.6


def estimate_tokens(text: str) -> int:
    """Грубая локальная оценка числа токенов по длине текста."""
    return max(1, round(len(text) / CHARS_PER_TOKEN))


def proxy_units(value: Any) -> int:
    """Локальная мера объёма «слова и знаки» (не токены GigaChat)."""
    import re

    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return len(re.findall(r"\w+|[^\w\s]", text))
