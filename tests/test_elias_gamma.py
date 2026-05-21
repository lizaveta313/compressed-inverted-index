import pytest

from compressed_inverted_index.elias_gamma import (
    gamma_decode_list,
    gamma_decode_number,
    gamma_encode_list,
    gamma_encode_number,
)


@pytest.mark.parametrize(
    ("number", "bits"),
    [
        (1, "1"),
        (2, "010"),
        (3, "011"),
        (4, "00100"),
        (5, "00101"),
        (8, "0001000"),
    ],
)
def test_gamma_encode_number(number: int, bits: str) -> None:
    assert gamma_encode_number(number) == bits


@pytest.mark.parametrize("number", [0, -1, -100])
def test_gamma_encode_number_rejects_non_positive_values(number: int) -> None:
    with pytest.raises(ValueError):
        gamma_encode_number(number)


def test_gamma_decode_number_returns_value_and_next_position() -> None:
    bits = "01000101"

    first, position = gamma_decode_number(bits)
    second, next_position = gamma_decode_number(bits, position)

    assert first == 2
    assert position == 3
    assert second == 5
    assert next_position == len(bits)


@pytest.mark.parametrize("bits", ["", "0", "00", "01", "0010", "abc", "102"])
def test_gamma_decode_number_rejects_invalid_bit_strings(bits: str) -> None:
    with pytest.raises(ValueError):
        gamma_decode_number(bits)


def test_gamma_encode_and_decode_list() -> None:
    numbers = [1, 2, 3, 7, 16, 31]

    encoded = gamma_encode_list(numbers)

    assert gamma_decode_list(encoded) == numbers


def test_gamma_decode_empty_list() -> None:
    assert gamma_decode_list("") == []


def test_gamma_decode_list_rejects_incomplete_trailing_code() -> None:
    with pytest.raises(ValueError):
        gamma_decode_list("0100")
