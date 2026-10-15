"""Общие фикстуры офлайн-тестов: маленький индекс без вызовов API."""

from __future__ import annotations

import numpy as np
import pytest

from agent.corpus import Chunk
from agent.retrieval import Retriever


def make_chunk(source: str, position: int, text: str, year: int = 2016, speakers=("С. Плуготаренко",)) -> Chunk:
    """Фрагмент для тестов."""
    return Chunk(
        chunk_id=f"{source}#{position:04d}", source_id=source, position=position,
        time_start=f"00:0{position}:00", time_end=f"00:0{position + 1}:00", speakers=list(speakers),
        text=text, session_title=f"ПМЭФ-{year}, 18.06, открытая сессия: it", year=year, line_nos=[position],
    )


@pytest.fixture()
def small_retriever(tmp_path) -> Retriever:
    """Индекс из пяти фрагментов двух сессий с детерминированными векторами."""
    chunks = [
        make_chunk("TRANSFER_2016_06_18_OC_it.txt", 0, "Мы рады приветствовать вас на шестом IT-завтраке. Я Сергей Плуготаренко, директор РАЭК."),
        make_chunk("TRANSFER_2016_06_18_OC_it.txt", 1, "Каждому спикеру отводится по 5 минут."),
        make_chunk("TRANSFER_2016_06_18_OC_it.txt", 2, "Тема - роль IT в Национальной технологической инициативе."),
        make_chunk("TRANSFER_2014_05_24_OC_it.txt", 0, "Мы начинаем четвертый IT- завтрак, соорганизаторы РВК, РУССОФТ и РАЭК.", year=2014),
        make_chunk("TRANSFER_2015_06_18_STRATSESSION_nuclear.txt", 0, "There are now 438 nuclear power reactors in operation. Another 67 plants are under construction.", year=2015, speakers=("Y. Amano",)),
    ]
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(len(chunks), 8)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    catalog = [
        {"source_id": c.source_id, "title": c.session_title, "year": c.year, "topic": "it", "kind": "открытая сессия",
         "language": "ru", "speakers": c.speakers, "n_speakers": 1}
        for c in chunks
    ]
    return Retriever(chunks, matrix, catalog, {"TRANSFER_2016_06_18_OC_it_999.txt": "TRANSFER_2016_06_18_OC_it.txt"},
                     llm=None, index_dir=tmp_path)
