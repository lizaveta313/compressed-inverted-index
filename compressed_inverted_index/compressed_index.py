from __future__ import annotations

from compressed_inverted_index.elias_gamma import gamma_decode_list, gamma_encode_list
from compressed_inverted_index.inverted_index import InvertedIndex
from compressed_inverted_index.tokenizer import tokenize


def delta_encode_doc_ids(doc_ids: list[int]) -> list[int]:
    """Первый doc_id хранится как doc_id + 1, остальные - как разности."""

    deltas: list[int] = []
    previous = -1

    for position, doc_id in enumerate(doc_ids):
        if not isinstance(doc_id, int):
            raise TypeError("doc_id values must be integers")
        if doc_id < 0:
            raise ValueError("doc_id values must be non-negative")

        delta = doc_id + 1 if position == 0 else doc_id - previous
        if delta <= 0:
            raise ValueError("doc_id values must be sorted and unique")

        deltas.append(delta)
        previous = doc_id

    return deltas


def delta_decode_doc_ids(deltas: list[int]) -> list[int]:
    if not deltas:
        return []

    doc_ids: list[int] = []
    previous = -1

    for position, delta in enumerate(deltas):
        if not isinstance(delta, int):
            raise TypeError("delta values must be integers")
        if delta <= 0:
            raise ValueError("delta values must be positive")

        doc_id = delta - 1 if position == 0 else previous + delta
        if doc_id < 0:
            raise ValueError("decoded doc_id must be non-negative")

        doc_ids.append(doc_id)
        previous = doc_id

    return doc_ids


class CompressedInvertedIndex:
    def __init__(self) -> None:
        self._plain_index = InvertedIndex()
        self._compressed_postings: dict[str, str] = {}

    def add_document(self, doc_id: int, text: str) -> None:
        self._plain_index.add_document(doc_id, text)
        self._recompress()

    def build(self, documents: dict[int, str]) -> None:
        self._plain_index = InvertedIndex()
        self._plain_index.build(documents)
        self._recompress()

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

        encoded = self._compressed_postings.get(tokens[0])
        if encoded is None:
            return []
        return self._decode_postings(encoded)

    def storage_size_bytes(self) -> int:
        """Оценка размера: байты термов и упакованные гамма-коды."""

        size = 0
        for term, bits in self._compressed_postings.items():
            size += len(term.encode("utf-8"))
            size += (len(bits) + 7) // 8
        return max(size, 1)

    def compression_ratio(self, uncompressed_size: int) -> float:
        if uncompressed_size <= 0:
            raise ValueError("uncompressed_size must be positive")
        return self.storage_size_bytes() / uncompressed_size

    def _recompress(self) -> None:
        self._compressed_postings = {
            term: self._encode_postings(postings)
            for term, postings in self._plain_index.items()
        }

    @staticmethod
    def _encode_postings(postings: list[int]) -> str:
        deltas = delta_encode_doc_ids(postings)
        return gamma_encode_list(deltas)

    @staticmethod
    def _decode_postings(bits: str) -> list[int]:
        deltas = gamma_decode_list(bits)
        return delta_decode_doc_ids(deltas)
