from __future__ import annotations

import json
import random
from pathlib import Path
from collections.abc import Iterator
from typing import Any

TEXT_FIELDS = ("text", "content", "body")


def load_documents_from_jsonl(path: str, limit: int | None = None) -> dict[int, str]:
    """Загружает документы из JSONL или JSON-массива.

    Текст берется из первого найденного поля: text, content или body.
    Некорректные строки и записи без текста пропускаются.
    """

    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative or None")

    file_path = Path(path)
    if not file_path.exists():
        print(f"Файл с данными не найден: {file_path}")
        return {}

    documents: dict[int, str] = {}
    json_errors = 0
    skipped_without_text = 0

    for record in _iter_records(file_path):
        if limit is not None and len(documents) >= limit:
            break

        if record is None:
            json_errors += 1
            continue

        text = _extract_text(record)
        if text is None:
            skipped_without_text += 1
            continue

        documents[len(documents)] = _repair_mojibake(text)

    if json_errors:
        print(f"Пропущено строк/записей с некорректным JSON: {json_errors}")
    if skipped_without_text:
        print(f"Пропущено записей без текстового поля: {skipped_without_text}")

    return documents


def _iter_records(path: Path) -> Iterator[dict[str, Any] | None]:
    with path.open("r", encoding="utf-8") as file:
        first_char = _read_first_non_space(file)
        file.seek(0)
        if first_char == "[":
            yield from _iter_json_array_records(file)
            return

        for line in file:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                yield None
                continue

            yield record if isinstance(record, dict) else None


def _read_first_non_space(file: Any) -> str:
    while True:
        char = file.read(1)
        if not char:
            return ""
        if not char.isspace():
            return char


def _iter_json_array_records(file: Any) -> Iterator[dict[str, Any] | None]:
    in_string = False
    escaped = False
    depth = 0
    buffer: list[str] = []

    while True:
        chunk = file.read(8192)
        if not chunk:
            break

        for char in chunk:
            if depth == 0:
                if char == "{":
                    depth = 1
                    buffer = [char]
                continue

            buffer.append(char)

            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        record = json.loads("".join(buffer))
                    except json.JSONDecodeError:
                        yield None
                    else:
                        yield record if isinstance(record, dict) else None
                    buffer = []


def _repair_mojibake(text: str) -> str:
    """Исправляет частый случай UTF-8, ошибочно прочитанного как cp1251."""

    if "Р" not in text and "С" not in text:
        return text

    try:
        repaired = text.encode("cp1251").decode("utf-8")
    except UnicodeError:
        return text

    original_score = text.count("Р") + text.count("С")
    repaired_score = repaired.count("Р") + repaired.count("С")
    return repaired if repaired_score < original_score else text


def generate_documents(count: int, seed: int = 42) -> dict[int, str]:
    """Создает воспроизводимый набор документов для fallback-эксперимента."""

    if count < 0:
        raise ValueError("count must be non-negative")

    rng = random.Random(seed)
    vocabulary = [
        "университет",
        "наука",
        "студент",
        "факультет",
        "библиотека",
        "кампус",
        "лекция",
        "исследование",
        "данные",
        "алгоритм",
        "поиск",
        "индекс",
        "education",
        "science",
        "student",
        "faculty",
        "library",
        "campus",
        "lecture",
        "research",
        "data",
        "algorithm",
        "search",
        "index",
        "university",
        "rector",
    ]

    documents: dict[int, str] = {}
    for doc_id in range(count):
        words = [rng.choice(vocabulary) for _ in range(rng.randint(8, 16))]
        phrases: list[str] = []
        if doc_id % 10 == 0:
            phrases.append("Ректор СПбГУ")
        if doc_id % 15 == 0:
            phrases.append("Ректор МГУ")
        if doc_id % 7 == 0:
            phrases.append("university rector")

        documents[doc_id] = " ".join(words + phrases)

    return documents


def _extract_text(record: dict[str, Any]) -> str | None:
    for field in TEXT_FIELDS:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return None
