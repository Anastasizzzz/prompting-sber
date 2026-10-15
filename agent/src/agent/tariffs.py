"""Тарифы GigaChat и перевод токенов в рубли.

Ставки - оплата по факту для юрлица, рубли за 1000 токенов.
Перед реальным расчётом их сверяют со страницей тарифов: цены меняются, а выводы
о том, что дешевле, от них зависят.

Что тарифицируется:

* ``total_tokens = prompt_tokens + completion_tokens`` - это и есть счёт вызова;
* ``precached_prompt_tokens`` - вход, переиспользованный из кеша сессии, не оплачивается;
* ``raw_input_tokens = prompt_tokens + precached_prompt_tokens`` - полная длина запроса,
  правильная мера *размера* промпта, но не его *цены*.
"""

from __future__ import annotations

from decimal import Decimal

TARIFF_SOURCE = "https://developers.sber.ru/docs/ru/gigachat/tariffs/legal-tariffs"

PRICE_PER_1000_RUB: dict[tuple[str, str], Decimal] = {
    ("GigaChat-2", "sync"): Decimal("0.065"),
    ("GigaChat-2", "async"): Decimal("0.0325"),
    ("GigaChat-2-Pro", "sync"): Decimal("0.5"),
    ("GigaChat-2-Pro", "async"): Decimal("0.25"),
    ("GigaChat-2-Max", "sync"): Decimal("0.65"),
    ("GigaChat-2-Max", "async"): Decimal("0.325"),
    ("EmbeddingsGigaR", "sync"): Decimal("0.014"),
    ("Embeddings", "sync"): Decimal("0.014"),
}

MODEL_LABELS = {"GigaChat-2": "Lite", "GigaChat-2-Pro": "Pro", "GigaChat-2-Max": "Max"}


def normalize_model_name(raw_model: str | None, default: str = "GigaChat-2-Max") -> str:
    """API возвращает ``GigaChat-2-Max:2.0.30``, а тариф задан для ``GigaChat-2-Max``."""
    if not raw_model:
        return default
    base = str(raw_model).split(":", 1)[0]
    return base if any(key[0] == base for key in PRICE_PER_1000_RUB) else default


def price_per_1000(model: str, mode: str = "sync") -> Decimal:
    """Ставка за 1000 токенов для модели и режима (``sync`` / ``async``)."""
    key = (normalize_model_name(model), mode)
    if key not in PRICE_PER_1000_RUB:
        raise KeyError(f"Нет тарифа для {key}. Сверьтесь с {TARIFF_SOURCE}")
    return PRICE_PER_1000_RUB[key]


def tokens_cost_rub(tokens: int, model: str, mode: str = "sync") -> Decimal:
    """Стоимость заданного числа оплачиваемых токенов."""
    return Decimal(int(tokens)) / Decimal(1000) * price_per_1000(model, mode)


def rub(value: Decimal | float | int | None, digits: int = 4) -> float | None:
    """Округление для показа. Внутри расчётов деньги остаются в ``Decimal``."""
    if value is None:
        return None
    return float(round(Decimal(str(value)), digits))
