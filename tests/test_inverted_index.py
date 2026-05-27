from compressed_inverted_index.inverted_index import InvertedIndex


def test_build_index_and_get_sorted_postings() -> None:
    index = InvertedIndex()
    index.build(
        {
            3: "Ректор СПбГУ",
            1: "СПбГУ университет",
            2: "МГУ университет",
        }
    )

    assert index.get_postings("спбгу") == [1, 3]
    assert index.get_postings("университет") == [1, 2]


def test_repeated_words_do_not_duplicate_doc_id() -> None:
    index = InvertedIndex()
    index.add_document(1, "index index INDEX индекс")

    assert index.get_postings("index") == [1]


def test_search_one_word() -> None:
    index = InvertedIndex()
    index.build({0: "alpha beta", 1: "beta gamma", 2: "gamma"})

    assert index.search("beta") == [0, 1]


def test_search_multiple_words_returns_intersection() -> None:
    index = InvertedIndex()
    index.build(
        {
            0: "Ректор СПбГУ university",
            1: "Ректор МГУ university",
            2: "Ректор СПбГУ МГУ",
        }
    )

    assert index.search("Ректор СПбГУ") == [0, 2]
    assert index.search("Ректор university") == [0, 1]


def test_search_returns_empty_result() -> None:
    index = InvertedIndex()
    index.build({0: "alpha beta", 1: "gamma"})

    assert index.search("alpha gamma") == []
    assert index.search("missing") == []


def test_storage_size_is_positive() -> None:
    index = InvertedIndex()

    assert index.storage_size_bytes() > 0

    index.build({0: "alpha beta"})

    assert index.storage_size_bytes() > 0
