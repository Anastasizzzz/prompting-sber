"""Нормализация текста, токенизация для лексического поиска и работа с числами.

Корпус ПМЭФ содержит артефакты кодировки (``ѐ`` вместо ``ё``, разные виды тире,
комбинируемые ударения) и смешивает русский с английским. Все сравнения -
цитат, чисел, ключевых слов - идут через одну нормализацию, иначе одна и та же
фраза будет считаться разной.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

_CHAR_MAP = str.maketrans(
    {
        "ѐ": "е",
        "Ѐ": "Е",
        "ё": "е",
        "Ё": "Е",
        "―": "-",
        "‒": "-",
        "–": "-",
        "-": "-",
        "−": "-",
        "«": '"',
        "»": '"',
        "„": '"',
        "“": '"',
        "”": '"',
        "’": "'",
        "‘": "'",
        "\xa0": " ",
        "\xad": "",
    }
)

WORD_RE = re.compile(r"[0-9a-zа-я]+", flags=re.IGNORECASE)
SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+(?=[A-ZА-ЯЁ0-9«\"(])")


def normalize(text: str) -> str:
    """Приводит текст к виду для сравнения: регистр, ё/ѐ, тире, кавычки, ударения, пробелы."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = unicodedata.normalize("NFC", text).translate(_CHAR_MAP).lower()
    return " ".join(text.split())


def words(text: str) -> list[str]:
    """Слова нормализованного текста (буквы и цифры)."""
    return WORD_RE.findall(normalize(text))


def split_sentences(text: str) -> list[str]:
    """Делит текст на предложения по знакам конца предложения перед заглавной буквой."""
    return [part.strip() for part in SENTENCE_RE.split(text) if part.strip()]


# --- транслитерация для поиска фамилий: «Амано» <-> «Amano» ---------------------------

_RU_TO_LAT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ж": "zh", "з": "z",
    "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p",
    "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "kh", "ц": "ts", "ч": "ch",
    "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}


def translit_ru_to_lat(word: str) -> str:
    """Грубая транслитерация русского слова латиницей (для совпадения фамилий)."""
    return "".join(_RU_TO_LAT.get(ch, ch) for ch in word)


def stem(word: str) -> str:
    """Лёгкий стемминг: усечение длинных слов до 6 символов.

    Полноценная морфология здесь не нужна: задача лексического поиска - поймать
    имена, организации и числа, а не склонения общих слов.
    """
    return word if len(word) <= 6 or word.isdigit() else word[:6]


_STOPWORDS = set(
    """и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне
    было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас
    нибудь опять уж вам ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя их чем
    была сам чтоб без будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой совсем
    ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех никогда можно при наконец два
    об другой хоть после над больше тот через эти нас про всего них какая много разве три эту моя впрочем
    хорошо свою этой перед иногда лучше чуть том нельзя такой им более всегда конечно всю между это
    the a an of to in and is are was were be for on at by with that this it as from or which""".split()
)


def lexical_tokens(text: str, *, add_translit: bool = False) -> list[str]:
    """Токены для BM25: нормализация, стоп-слова, стемминг, при необходимости транслит."""
    tokens: list[str] = []
    for word in words(text):
        if word in _STOPWORDS or len(word) < 2:
            continue
        tokens.append(stem(word))
        if add_translit and re.search("[а-я]", word):
            tokens.append(stem(translit_ru_to_lat(word)))
    return tokens


# --- числа -----------------------------------------------------------------------------

NUMBER_WORDS = {
    "один": "1", "одна": "1", "одного": "1", "два": "2", "две": "2", "двух": "2", "три": "3",
    "трех": "3", "четыре": "4", "четырех": "4", "пять": "5", "пяти": "5", "шесть": "6",
    "шести": "6", "семь": "7", "семи": "7", "восемь": "8", "восьми": "8", "девять": "9",
    "десять": "10", "десяти": "10", "двадцать": "20", "тридцать": "30", "сорок": "40",
    "пятьдесят": "50", "сто": "100", "тысяча": "1000", "треть": "1/3", "трети": "1/3",
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
    "eight": "8", "nine": "9", "ten": "10", "third": "1/3",
}


def extract_numbers(text: str) -> set[str]:
    """Числа в тексте в едином виде: цифры (без разделителей разрядов) и числа словами."""
    if not text:
        return set()
    norm = normalize(text)
    found = {re.sub(r"[\s.,]", "", m) for m in re.findall(r"\d[\d\s.,]*\d|\d", norm)}
    found = {item for item in found if item}
    for word in words(norm):
        if word in NUMBER_WORDS:
            found.add(NUMBER_WORDS[word])
    return found


def fabricated_numbers(answer: str | None, evidence_text: str) -> list[str]:
    """Числа ответа, которых нет в доказательствах (признак выдумки).

    Годы ПМЭФ (2011–2016) и порядковые слова не считаем выдумкой: год часто
    берётся из названия сессии, а не из цитаты.
    """
    if not answer:
        return []
    allowed = extract_numbers(evidence_text) | {str(year) for year in range(2011, 2017)}
    return sorted(extract_numbers(answer) - allowed)


def contains_quote(haystack: str, quote: str, *, min_words: int = 4) -> bool:
    """Проверяет, что цитата дословно (после нормализации) есть в тексте.

    Цитаты золотого набора содержат «…» вместо пропусков, поэтому цитата
    делится по многоточиям, и каждая содержательная часть ищется отдельно.
    """
    hay = " ".join(words(haystack))
    parts = [p for p in re.split(r"…|\.\.\.", quote)]
    checked = 0
    for part in parts:
        part_words = words(part)
        if len(part_words) < min_words:
            continue
        checked += 1
        if " ".join(part_words) not in hay:
            return False
    return checked > 0


QUOTE_MATCH_THRESHOLD = 0.85


def quote_similarity(haystack: str, quote: str, *, min_words: int = 4) -> float:
    """Наибольшее сходство частей цитаты с подряд идущими словами текста (0..1).

    1.0 - дословное совпадение. Иначе - по каждой части цитаты (между «…»)
    ищется окно текста той же длины ±2 слова с максимальным отношением
    ``SequenceMatcher`` по стеммам слов; итог - минимум по частям. Так одно
    изменённое слово («начинаем» -> «начиная») не бракует цитату, а
    пересказ или чужая фраза - бракует.
    """
    if contains_quote(haystack, quote, min_words=min_words):
        return 1.0
    hay = [stem(w) for w in words(haystack)]
    scores = []
    for part in re.split(r"…|\.\.\.", quote):
        part_words = [stem(w) for w in words(part)]
        if len(part_words) < min_words:
            continue
        n, best = len(part_words), 0.0
        for size in range(max(1, n - 2), n + 3):
            for start in range(0, max(1, len(hay) - size + 1)):
                ratio = SequenceMatcher(None, part_words, hay[start : start + size], autojunk=False).ratio()
                if ratio > best:
                    best = ratio
        scores.append(best)
    return min(scores) if scores else 0.0
