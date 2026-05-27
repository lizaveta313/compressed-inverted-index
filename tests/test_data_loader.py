from compressed_inverted_index.data_loader import (
    generate_documents,
    load_documents_from_jsonl,
)


def test_load_documents_from_jsonl_reads_supported_text_fields(tmp_path) -> None:
    path = tmp_path / "docs.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"url": "https://spbu.ru", "text": "Ректор СПбГУ"}',
                '{"id": 123, "content": "Ректор МГУ"}',
                '{"body": "university rector"}',
            ]
        ),
        encoding="utf-8",
    )

    assert load_documents_from_jsonl(str(path)) == {
        0: "Ректор СПбГУ",
        1: "Ректор МГУ",
        2: "university rector",
    }


def test_load_documents_from_json_array(tmp_path) -> None:
    path = tmp_path / "pages.json"
    path.write_text(
        '[{"url": "https://spbu.ru", "text": "Ректор СПбГУ"}, {"body": "Ректор МГУ"}]',
        encoding="utf-8",
    )

    assert load_documents_from_jsonl(str(path)) == {
        0: "Ректор СПбГУ",
        1: "Ректор МГУ",
    }


def test_load_documents_from_jsonl_skips_bad_json_and_missing_text(tmp_path) -> None:
    path = tmp_path / "docs.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"text": "valid"}',
                "{bad json",
                '{"url": "https://example.com"}',
                "",
                '{"content": "also valid"}',
            ]
        ),
        encoding="utf-8",
    )

    assert load_documents_from_jsonl(str(path)) == {0: "valid", 1: "also valid"}


def test_load_documents_from_jsonl_respects_limit(tmp_path) -> None:
    path = tmp_path / "docs.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"text": "one"}',
                '{"text": "two"}',
                '{"text": "three"}',
            ]
        ),
        encoding="utf-8",
    )

    assert load_documents_from_jsonl(str(path), limit=2) == {0: "one", 1: "two"}


def test_load_documents_from_jsonl_missing_file_returns_empty_dict(tmp_path) -> None:
    assert load_documents_from_jsonl(str(tmp_path / "missing.jsonl")) == {}


def test_generate_documents_returns_requested_count() -> None:
    documents = generate_documents(5)

    assert list(documents) == [0, 1, 2, 3, 4]
