"""Корпус расшифровок ПМЭФ: разбор файлов, метаданные, дедупликация, нарезка.

Особенности корпуса (проверены при изучении данных):

* строка файла - «минута» вида ``[00:01:00 - 00:02:00] | Спикер | текст``,
  но внутри минуты бывает монолог до 29 тыс. символов;
* год, дата, тип и тема сессии есть только в имени файла;
* 15 пар файлов - точные дубли (суффикс ``_NNN``), ещё одна пара - почти дубль.

Нарезка: по репликам, длинные реплики -
по предложениям, таймкод и спикер сохраняются в метаданных, к тексту для эмбеддинга
добавляется короткий контекстный заголовок с сессией и годом.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .text import normalize, split_sentences, words

LINE_RE = re.compile(r"^\[(\d\d:\d\d:\d\d) - (\d\d:\d\d:\d\d)\] \| (.*?) \| (.*)$")
FILE_RE = re.compile(
    r"^TRANSFER_(?P<year>20\d\d)_(?P<month>\d\d)_(?P<day>\d\d)_(?P<kind>OC|PRAVLENIE|STRATSESSION)_"
    r"(?P<topic>[a-z]+)(?:_(?P<suffix>\d+))?\.txt$"
)
SESSION_KIND_RU = {
    "OC": "открытая сессия",
    "PRAVLENIE": "панельная сессия",
    "STRATSESSION": "стратегическая сессия",
}


@dataclass(frozen=True)
class Replica:
    """Одна строка расшифровки: минута, спикер, текст."""

    source_id: str
    line_no: int
    time_start: str
    time_end: str
    speaker: str
    text: str


@dataclass(frozen=True)
class SessionMeta:
    """Метаданные сессии, восстановленные из имени файла и текста."""

    source_id: str
    year: int
    date: str
    kind: str
    kind_ru: str
    topic: str
    title: str
    speakers: tuple[str, ...]
    n_replicas: int
    n_chars: int
    latin_share: float
    duplicate_of: str | None = None

    @property
    def language(self) -> str:
        """Грубая метка языка по доле латиницы."""
        if self.latin_share > 0.6:
            return "en"
        if self.latin_share > 0.25:
            return "ru+en"
        return "ru"


@dataclass
class Chunk:
    """Фрагмент для поиска: одна или несколько реплик либо часть длинной реплики."""

    chunk_id: str
    source_id: str
    position: int
    time_start: str
    time_end: str
    speakers: list[str]
    text: str
    session_title: str
    year: int
    line_nos: list[int] = field(default_factory=list)
    overlap: str = ""

    def embedding_text(self) -> str:
        """Текст для эмбеддинга документа: контекстный заголовок + реплика.

        В исходном тексте нет ни года, ни названия сессии, поэтому IT-завтрак
        2014 и 2016 годов почти неразличимы для векторов. Заголовок - приём
        «контекстуализации фрагмента».
        """
        speakers = ", ".join(self.speakers)
        body = f"{self.overlap}\n{self.text}".strip() if self.overlap else self.text
        return f"{self.session_title}. Спикеры: {speakers}.\n{body}"

    def to_dict(self) -> dict:
        """Сериализация для chunks.jsonl."""
        return asdict(self)


def parse_session_name(source_id: str) -> dict:
    """Разбирает имя файла ``TRANSFER_2016_06_18_OC_it.txt`` в год, дату, тип и тему."""
    match = FILE_RE.match(source_id)
    if match is None:
        raise ValueError(f"Неожиданное имя файла: {source_id}")
    data = match.groupdict()
    kind = data["kind"]
    year = int(data["year"])
    date = f"{data['day']}.{data['month']}"
    title = f"ПМЭФ-{year}, {date}, {SESSION_KIND_RU[kind]}: {data['topic']}"
    return {
        "year": year,
        "date": date,
        "kind": kind,
        "kind_ru": SESSION_KIND_RU[kind],
        "topic": data["topic"],
        "suffix": data["suffix"],
        "title": title,
    }


def parse_file(path: Path) -> list[Replica]:
    """Читает файл расшифровки. Строка, не совпавшая с форматом, - ошибка данных."""
    replicas = []
    lines = [line for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
    for line_no, line in enumerate(lines):
        match = LINE_RE.match(line)
        if match is None:
            raise ValueError(f"{path.name}:{line_no}: строка не в формате расшифровки")
        time_start, time_end, speaker, text = match.groups()
        replicas.append(Replica(path.name, line_no, time_start, time_end, speaker.strip(), text.strip()))
    return replicas


def _shingles(text: str, size: int = 5, step: int = 3) -> set[str]:
    """Шинглы из ``size`` слов с шагом ``step`` - для поиска почти-дублей по пересечению."""
    tokens = words(text)
    return {" ".join(tokens[i : i + size]) for i in range(0, max(len(tokens) - size, 1), step)}


@dataclass
class Corpus:
    """Корпус после проверок и дедупликации."""

    sessions: dict[str, SessionMeta]
    replicas: dict[str, list[Replica]]
    duplicates: dict[str, str]

    @property
    def canonical_ids(self) -> list[str]:
        """Идентификаторы уникальных сессий (дубли исключены)."""
        return sorted(sid for sid, meta in self.sessions.items() if meta.duplicate_of is None)

    def canonical(self, source_id: str) -> str:
        """Канонический файл для дубля (например, ``…economy_209.txt`` -> ``…economy.txt``)."""
        return self.duplicates.get(source_id, source_id)


def load_corpus(corpus_dir: Path, *, near_duplicate_threshold: float = 0.9) -> Corpus:
    """Загружает все файлы, строит метаданные и находит дубли.

    Точные дубли ищутся по хешу содержимого, почти-дубли - по доле общих
    5-словных шинглов. Каноническим считается файл без числового суффикса.
    """
    paths = sorted(corpus_dir.glob("*.txt"))
    if not paths:
        raise FileNotFoundError(f"В {corpus_dir} нет файлов расшифровок")

    replicas = {path.name: parse_file(path) for path in paths}
    raw_texts = {path.name: path.read_text(encoding="utf-8") for path in paths}

    def priority(source_id: str) -> tuple[int, str]:
        """Порядок выбора канонического файла: сначала имя без числового суффикса."""
        return (1 if parse_session_name(source_id)["suffix"] else 0, source_id)

    duplicates: dict[str, str] = {}
    by_hash: dict[str, list[str]] = {}
    for sid, text in raw_texts.items():
        by_hash.setdefault(hashlib.sha256(text.encode()).hexdigest(), []).append(sid)
    for group in by_hash.values():
        group = sorted(group, key=priority)
        for sid in group[1:]:
            duplicates[sid] = group[0]

    remaining = [sid for sid in sorted(raw_texts, key=priority) if sid not in duplicates]
    shingles = {sid: _shingles(raw_texts[sid]) for sid in remaining}
    for i, left in enumerate(remaining):
        if left in duplicates:
            continue
        for right in remaining[i + 1 :]:
            if right in duplicates:
                continue
            a, b = shingles[left], shingles[right]
            if not a or not b:
                continue
            containment = len(a & b) / min(len(a), len(b))
            if containment >= near_duplicate_threshold:
                keep, drop = sorted([left, right], key=priority)
                duplicates[drop] = keep

    sessions = {}
    for sid, items in replicas.items():
        info = parse_session_name(sid)
        body = " ".join(r.text for r in items)
        letters = re.findall(r"[A-Za-zА-Яа-яЁё]", body)
        latin = sum(1 for ch in letters if ch.isascii())
        speakers: dict[str, None] = {}
        for r in items:
            speakers.setdefault(r.speaker, None)
        sessions[sid] = SessionMeta(
            source_id=sid,
            year=info["year"],
            date=info["date"],
            kind=info["kind"],
            kind_ru=info["kind_ru"],
            topic=info["topic"],
            title=info["title"],
            speakers=tuple(speakers),
            n_replicas=len(items),
            n_chars=len(body),
            latin_share=round(latin / max(len(letters), 1), 3),
            duplicate_of=duplicates.get(sid),
        )
    return Corpus(sessions=sessions, replicas=replicas, duplicates=duplicates)


def _split_long_text(text: str, max_chars: int) -> list[str]:
    """Делит длинную реплику по предложениям на части не длиннее ``max_chars``."""
    pieces: list[str] = []
    current = ""
    for sentence in split_sentences(text) or [text]:
        while len(sentence) > max_chars:  # «предложение» без точек - режем по словам
            if current:
                pieces.append(current)
                current = ""
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if current and len(current) + 1 + len(sentence) > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def chunk_corpus(corpus: Corpus, *, max_chars: int = 1000, overlap_chars: int = 150) -> list[Chunk]:
    """Режет уникальные сессии на фрагменты.

    * короткие соседние реплики склеиваются до ``max_chars`` (с подписью спикера);
    * длинная реплика делится по предложениям, части наследуют её таймкод;
    * перекрытие - хвост предыдущего фрагмента до ``overlap_chars`` символов,
      чтобы условие не разрывалось на границе. Перекрытие хранится
      отдельным полем: оно участвует в эмбеддинге, но не дублируется в контексте
      генератора при склейке соседних фрагментов. В индексе ``assets``
      перекрытие было ~50%, здесь оно намеренно небольшое: каждый символ
      перекрытия оплачивается и в индексации, и в контексте генератора.
    """
    chunks: list[Chunk] = []
    for sid in corpus.canonical_ids:
        meta = corpus.sessions[sid]
        units: list[tuple[Replica, str]] = []
        for replica in corpus.replicas[sid]:
            for piece in _split_long_text(replica.text, max_chars - 40):
                units.append((replica, f"{replica.speaker}: {piece}"))

        group: list[tuple[Replica, str]] = []
        position = 0

        def flush(group: list[tuple[Replica, str]], overlap: str) -> None:
            """Собирает накопленные реплики в один фрагмент и добавляет его в список."""
            nonlocal position
            if not group:
                return
            speakers: dict[str, None] = {}
            for replica, _ in group:
                speakers.setdefault(replica.speaker, None)
            text = "\n".join(text for _, text in group)
            chunks.append(
                Chunk(
                    chunk_id=f"{sid}#{position:04d}",
                    source_id=sid,
                    position=position,
                    time_start=group[0][0].time_start,
                    time_end=group[-1][0].time_end,
                    speakers=list(speakers),
                    text=text,
                    session_title=meta.title,
                    year=meta.year,
                    line_nos=sorted({replica.line_no for replica, _ in group}),
                    overlap=overlap,
                )
            )
            position += 1

        overlap = ""
        for unit in units:
            size = sum(len(text) + 1 for _, text in group) + len(unit[1])
            if group and size > max_chars:
                flush(group, overlap)
                tail = group[-1][1]
                overlap = "… " + tail[-overlap_chars:].split(" ", 1)[-1] if overlap_chars else ""
                group = []
            group.append(unit)
        flush(group, overlap)
    return chunks


def session_catalog_rows(corpus: Corpus) -> list[dict]:
    """Строки каталога сессий для инструмента ``list_sessions`` и отчёта."""
    rows = []
    for sid in corpus.canonical_ids:
        meta = corpus.sessions[sid]
        rows.append(
            {
                "source_id": sid,
                "title": meta.title,
                "year": meta.year,
                "topic": meta.topic,
                "kind": meta.kind_ru,
                "language": meta.language,
                "speakers": list(meta.speakers[:8]),
                "n_speakers": len(meta.speakers),
            }
        )
    return rows


def first_replica_snippet(corpus: Corpus, source_id: str, chars: int = 160) -> str:
    """Начало первой реплики сессии: обычно модератор называет тему и участников."""
    replicas = corpus.replicas[corpus.canonical(source_id)]
    return normalize(replicas[0].text)[:chars] if replicas else ""
