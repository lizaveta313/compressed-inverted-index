from compressed_inverted_index.tokenizer import normalize_text, tokenize


def test_tokenize_russian_and_english_text() -> None:
    text = "Ректор СПбГУ and University Rector"

    assert tokenize(text) == ["ректор", "спбгу", "and", "university", "rector"]


def test_tokenizer_removes_punctuation_and_empty_tokens() -> None:
    text = "Hello, мир!!!  Python... тест?"

    assert normalize_text(text) == "hello мир python тест"
    assert tokenize(text) == ["hello", "мир", "python", "тест"]


def test_tokenizer_returns_empty_list_for_blank_text() -> None:
    assert tokenize(" , ! ? ") == []
