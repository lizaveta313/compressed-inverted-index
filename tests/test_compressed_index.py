import pytest

from compressed_inverted_index.compressed_index import (
    CompressedInvertedIndex,
    delta_decode_doc_ids,
    delta_encode_doc_ids,
)
from compressed_inverted_index.inverted_index import InvertedIndex


def test_delta_encoding_and_decoding_with_zero_doc_id() -> None:
    doc_ids = [0, 2, 5, 11]

    deltas = delta_encode_doc_ids(doc_ids)

    assert deltas == [1, 2, 3, 6]
    assert delta_decode_doc_ids(deltas) == doc_ids


@pytest.mark.parametrize("doc_ids", [[2, 2], [3, 1], [-1, 2]])
def test_delta_encoding_rejects_invalid_doc_ids(doc_ids: list[int]) -> None:
    with pytest.raises(ValueError):
        delta_encode_doc_ids(doc_ids)


def test_compressed_index_builds_and_decodes_postings() -> None:
    index = CompressedInvertedIndex()
    index.build(
        {
            0: "Ректор СПбГУ",
            1: "Ректор МГУ",
            3: "Ректор СПбГУ university",
        }
    )

    assert index.get_postings("Ректор") == [0, 1, 3]
    assert index.get_postings("СПбГУ") == [0, 3]


def test_compressed_index_search_matches_plain_index() -> None:
    documents = {
        0: "Ректор СПбГУ university rector",
        1: "Ректор МГУ",
        2: "СПбГУ research",
        3: "university rector index",
    }
    plain = InvertedIndex()
    compressed = CompressedInvertedIndex()

    plain.build(documents)
    compressed.build(documents)

    for query in ["Ректор", "Ректор СПбГУ", "university rector", "missing"]:
        assert compressed.search(query) == plain.search(query)


def test_compressed_index_add_document_updates_search() -> None:
    index = CompressedInvertedIndex()

    index.add_document(2, "alpha beta")
    index.add_document(0, "alpha gamma")

    assert index.search("alpha") == [0, 2]
    assert index.search("alpha gamma") == [0]


def test_compressed_storage_size_is_positive() -> None:
    index = CompressedInvertedIndex()

    assert index.storage_size_bytes() > 0

    index.build({0: "alpha beta"})

    assert index.storage_size_bytes() > 0


def test_compression_ratio() -> None:
    documents = {doc_id: "common" for doc_id in range(100)}
    plain = InvertedIndex()
    compressed = CompressedInvertedIndex()

    plain.build(documents)
    compressed.build(documents)

    ratio = compressed.compression_ratio(plain.storage_size_bytes())

    assert ratio > 0
    assert ratio < 1


def test_compression_ratio_rejects_non_positive_uncompressed_size() -> None:
    index = CompressedInvertedIndex()

    with pytest.raises(ValueError):
        index.compression_ratio(0)
