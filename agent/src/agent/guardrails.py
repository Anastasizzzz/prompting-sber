"""Guardrails: защита слоями вокруг вызова модели.

Модель получает инструкции и данные одним каналом и не может надёжно отличить
одно от другого, поэтому защита - внешний слой, а не свойство модели:

* **вход** - дешёвый детектор по правилам (микросекунды, ноль рублей);
* **данные** - фрагменты объявлены данными в системном промпте (см. prompts.py), а перед генерацией
  проверяются детектором обращений к ассистенту (``check_fragment``): замер показал, что одной
  инструкции «фрагменты - данные» мало - «отравленный» фрагмент модель послушалась;
* **инструменты** - белый список имён и проверка аргументов (см. tools.py);
* **выход** - схема с перечислением id, проверка цитат и чисел (см. contracts.py)
  и проверка утечки системного промпта (здесь).

Детектор по правилам обходится перефразированием и даёт ложные срабатывания,
поэтому его качество меряется на наборе из атак **и** безобидных
«подозрительных» вопросов: полнота отражения против доли ложных срабатываний.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .text import normalize

INJECTION_PATTERNS = [
    r"\b(игнорир\w*|забудь\w*|отмени\w*)\b.{0,40}\b(инструкц\w*|правил\w*|документ\w*|фрагмент\w*|контекст\w*|предыдущ\w*)",
    r"\bignore\b.{0,30}\b(instruction|rule|previous|context)",
    r"(повтори|выведи|покажи|раскрой|переведи)\w*.{0,40}(системн\w* (инструкц|промпт|сообщени)|свои (инструкц|правил)|правила, которые)",
    r"\bsystem prompt\b|\bсистемн\w* промпт",
    r"\bты (теперь|больше не)\b|\bпредставь, что ты\b|\bделай вид, что\b",
    r"\b(придумай|сочини|выдумай)\b",
    r"администратор\w* .{0,30}разреши\w*|режим разработчика|developer mode|jailbreak",
    r"без (опоры на|учета|учёта) (документ|фрагмент|расшифров)",
]


def _compile(patterns: list[str]) -> list[re.Pattern]:
    """Шаблоны сравниваются с нормализованным текстом, где «й» -> «и»: та же замена нужна и в шаблонах.

    Без неё шаблоны с «й» («придумай», «отвечай», «любой») не совпадут ни с одним текстом.
    """
    return [re.compile(p.replace("й", "и").replace("Й", "И"), flags=re.IGNORECASE) for p in patterns]


_COMPILED = _compile(INJECTION_PATTERNS)


@dataclass
class InputCheck:
    """Результат проверки входа."""

    allowed: bool
    reason: str | None = None
    matched: str | None = None


def check_input(question: str, *, max_chars: int = 1000) -> InputCheck:
    """Проверка вопроса до любых платных вызовов."""
    text = " ".join(question.split())
    if not text:
        return InputCheck(False, "empty_question")
    if len(text) > max_chars:
        return InputCheck(False, "too_long")
    normalized = normalize(text)
    for pattern in _COMPILED:
        match = pattern.search(normalized)
        if match:
            return InputCheck(False, "prompt_injection", match.group(0)[:80])
    return InputCheck(True)


# Для фрагментов правила уже, чем для вопросов: в живой речи спикера «игнорировать правила» - обычная
# фраза, а обращение к ассистенту/модели и приказ, что отвечать, - нет. Ложные срабатывания меряются
# прогоном по всему корпусу (tests/test_offline.py и эксперимент guardrails).
FRAGMENT_PATTERNS = [
    r"(системн\w*|служебн\w*) (сообщени|примечани|указани|инструкц)\w*.{0,40}(ассистент|модел|ии\b|бот)",
    r"\b(ассистент|языков\w* модел|нейросет|chatgpt|gigachat|гигачат)\w*.{0,30}\b(должен|должна|обязан|отвечай|ответь|сообщи)",
    r"\b(на любой вопрос|на все вопросы)\b.{0,60}\b(отвечай|ответь|говори|сообщай)",
    r"\bне (упоминай|раскрывай|сообщай пользователю)\b",
    r"\b(assistant|language model)\b.{0,30}\b(must|should|answer|respond)\b|\bignore (all )?previous instructions\b",
]
_FRAGMENT_COMPILED = _compile(FRAGMENT_PATTERNS)


def check_fragment(text: str) -> str | None:
    """Совпадение, если фрагмент обращается к ассистенту как к исполнителю инструкций (иначе ``None``)."""
    normalized = normalize(" ".join(text.split()))
    for pattern in _FRAGMENT_COMPILED:
        match = pattern.search(normalized)
        if match:
            return match.group(0)[:80]
    return None


def leaks_system_prompt(answer: str | None, system_prompts: list[str], *, window: int = 8) -> bool:
    """Есть ли в ответе дословный кусок системной инструкции (утечка промпта, OWASP LLM07)."""
    if not answer:
        return False
    answer_words = normalize(answer).split()
    answer_text = " ".join(answer_words)
    for prompt in system_prompts:
        prompt_words = normalize(prompt).split()
        for i in range(0, max(len(prompt_words) - window, 0), window // 2):
            if " ".join(prompt_words[i : i + window]) in answer_text:
                return True
    return False
