from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterator

from compressed_inverted_index.tokenizer import tokenize


class InvertedIndex:
    def __init__(self) -> None:
        self._index: dict[str, list[int]] = {}

    def add_document(self, doc_id: int, text: str) -> None:
        if not isinstance(doc_id, int):
            raise TypeError("doc_id must be an integer")
        if doc_id < 0:
            raise ValueError("doc_id must be non-negative")

        for term in set(tokenize(text)):
            postings = self._index.setdefault(term, [])
            position = bisect_left(postings, doc_id)
            if position == len(postings) or postings[position] != doc_id:
                postings.insert(position, doc_id)

    def build(self, documents: dict[int, str]) -> None:
        self._index.clear()
        for doc_id, text in documents.items():
            self.add_document(doc_id, text)

    def search(self, query: str) -> list[int]:
        terms = tokenize(query)
        if not terms:
            return []

        posting_lists = [self.get_postings(term) for term in terms]
        if any(not postings for postings in posting_lists):
            return []

        result = set(posting_lists[0])
        for postings in posting_lists[1:]:
            result.intersection_update(postings)
            if not result:
                break
        return sorted(result)

    def get_postings(self, term: str) -> list[int]:
        tokens = tokenize(term)
        if len(tokens) != 1:
            return []
        return list(self._index.get(tokens[0], []))

    def storage_size_bytes(self) -> int:
        """Оценка размера: байты термов и 4 байта на каждый doc_id."""

        size = 0
        for term, postings in self._index.items():
            size += len(term.encode("utf-8"))
            size += len(postings) * 4
        return max(size, 1)

    def items(self) -> Iterator[tuple[str, list[int]]]:
        for term, postings in self._index.items():
            yield term, list(postings)
