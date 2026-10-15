"""Золотой набор: ядро из ``raw_data/Золотой набор.xlsx`` и расширение проекта.

* **Ядро** - 15 вопросов курса, неизменно (по нему строится таблица «до/после»).
  Вопрос, эталон, файл, таймкод и цитата читаются из xlsx как есть; к ним
  добавляется ручная разметка атомарных фактов (``golden_core_facts.json``).
* **Расширение** - калибровочная часть (подбор порога, калибровка судьи) и
  срезы, которых нет в ядре: ``not_found``, ловушки года, кросс-язычные,
  многошаговые вопросы, атаки (``golden_extended.jsonl``).

Эталонные ``chunk_id`` вычисляются автоматически: фрагмент той же сессии
(с учётом дублей), в котором нормализованная цитата найдена дословно.
Цитаты xlsx не строго дословны (``…`` вместо пропусков, другие кавычки,
редакторская ремарка в №15), поэтому сравнение идёт после нормализации и по частям.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import openpyxl

from ..corpus import Chunk
from ..settings import AGENT_DATA_DIR, GOLDEN_XLSX
from ..text import contains_quote, split_sentences, words

CORE_FACTS = AGENT_DATA_DIR / "golden_core_facts.json"
EXTENDED = AGENT_DATA_DIR / "golden_extended.jsonl"


@dataclass
class GoldSource:
    """Подтверждающий источник: файл, минута, цитата."""

    source_id: str
    time_start: str
    quote: str


@dataclass
class GoldCase:
    """Один вопрос золотого набора со всей разметкой для оценки."""

    case_id: str
    set_name: str  # core | extended
    split: str  # calibration | test
    slice: str
    question: str
    expected_status: str
    reference_answer: str | None
    required_facts: list[str]
    forbidden_facts: list[str]
    sources: list[GoldSource]
    relevant_chunk_ids: list[str] = field(default_factory=list)

    @property
    def relevant_source_ids(self) -> list[str]:
        """Файлы-источники эталона."""
        return sorted({s.source_id for s in self.sources})

    def to_dict(self) -> dict:
        """Для сохранения в прогоне."""
        return asdict(self)


def _clean_quote(quote: str) -> str:
    """Убирает внешние кавычки и редакторские ремарки в скобках (№15)."""
    quote = quote.strip().strip("«»\"")
    return re.sub(r"\([^)]*\)\s*$", "", quote).strip()


def load_core(xlsx_path: Path = GOLDEN_XLSX, facts_path: Path = CORE_FACTS) -> list[GoldCase]:
    """Читает 15 вопросов курса из xlsx (таблица начинается с 4-й строки)."""
    workbook = openpyxl.load_workbook(xlsx_path, read_only=True)
    sheet = workbook.active
    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    cases = []
    for row in sheet.iter_rows(min_row=5, values_only=True):
        if not row or row[0] is None or row[1] is None:
            continue
        number = str(int(float(row[0])))
        _, question, reference, source_id, time_start, quote, kind = row[:7]
        annotation = facts[number]
        cases.append(
            GoldCase(
                case_id=f"G{int(number):02d}",
                set_name="core",
                split="test",
                slice=annotation["slice"],
                question=str(question).strip(),
                expected_status="answered",
                reference_answer=str(reference).strip(),
                required_facts=annotation["required_facts"],
                forbidden_facts=annotation["forbidden_facts"],
                sources=[GoldSource(str(source_id).strip(), str(time_start).strip(), _clean_quote(str(quote)))],
            )
        )
    if len(cases) != 15:
        raise ValueError(f"В золотом наборе ожидалось 15 вопросов, найдено {len(cases)}")
    return cases


def load_extended(path: Path = EXTENDED) -> list[GoldCase]:
    """Читает расширение золотого набора."""
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        raw["sources"] = [GoldSource(**s) for s in raw.get("sources", [])]
        cases.append(GoldCase(set_name="extended", **raw))
    return cases


def resolve_relevant_chunks(cases: list[GoldCase], chunks: list[Chunk], duplicates: dict[str, str]) -> list[str]:
    """Находит эталонные ``chunk_id`` для каждого источника. Возвращает список проблем.

    Цитата может попасть на границу фрагментов, поэтому достаточно совпадения
    любого предложения цитаты (не короче 3 слов). Если цитата не
    найдена дословно, берутся фрагменты той же минуты - это фиксируется как
    проблема разметки, а не замалчивается.
    """
    by_source: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_source.setdefault(chunk.source_id, []).append(chunk)
    problems = []
    for case in cases:
        found: list[str] = []
        for source in case.sources:
            sid = duplicates.get(source.source_id, source.source_id)
            candidates = by_source.get(sid, [])
            parts = [
                sentence
                for piece in re.split(r"…|\.\.\.", source.quote)
                for sentence in split_sentences(piece)
                if len(words(sentence)) >= 3
            ]
            hits = [c.chunk_id for c in candidates if any(contains_quote(c.text, p, min_words=3) for p in parts)]
            if not hits:
                hits = [c.chunk_id for c in candidates if c.time_start == source.time_start]
                problems.append(f"{case.case_id}: цитата не найдена дословно в {sid}, взяты фрагменты минуты {source.time_start}")
            if not hits:
                problems.append(f"{case.case_id}: нет фрагментов для {sid} {source.time_start}")
            found.extend(hits)
        case.relevant_chunk_ids = list(dict.fromkeys(found))
    return problems


def load_golden(chunks: list[Chunk] | None = None, duplicates: dict[str, str] | None = None,
                *, include_extended: bool = True) -> list[GoldCase]:
    """Полный золотой набор; если передан индекс - с эталонными ``chunk_id``."""
    cases = load_core()
    if include_extended:
        cases += load_extended()
    ids = [c.case_id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("case_id в золотом наборе не уникальны")
    if chunks is not None:
        problems = resolve_relevant_chunks(cases, chunks, duplicates or {})
        for problem in problems:
            print("[golden]", problem)
    return cases


def select(cases: list[GoldCase], spec: str) -> list[GoldCase]:
    """Выборка: ``core`` | ``extended`` | ``test`` | ``calibration`` | ``all`` | список id через запятую."""
    if spec == "all":
        return cases
    if spec == "core":
        return [c for c in cases if c.set_name == "core"]
    if spec == "extended":
        return [c for c in cases if c.set_name == "extended"]
    if spec in ("test", "calibration"):
        return [c for c in cases if c.split == spec]
    wanted = [item.strip() for item in spec.split(",") if item.strip()]
    by_id = {c.case_id: c for c in cases}
    missing = [w for w in wanted if w not in by_id]
    if missing:
        raise KeyError(f"Нет кейсов {missing}")
    return [by_id[w] for w in wanted]
