from __future__ import annotations

import re

_NON_WORD_RE = re.compile(r"[^0-9A-Za-zА-Яа-яЁё]+", flags=re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Приводит текст к нижнему регистру и убирает пунктуацию."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    lowered = text.casefold()
    without_punctuation = _NON_WORD_RE.sub(" ", lowered)
    return _SPACE_RE.sub(" ", without_punctuation).strip()


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    return normalized.split()
