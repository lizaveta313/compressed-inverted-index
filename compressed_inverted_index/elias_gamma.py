from __future__ import annotations


def _ensure_bit_string(bits: str) -> None:
    if not isinstance(bits, str):
        raise TypeError("bits must be a string")
    if any(bit not in {"0", "1"} for bit in bits):
        raise ValueError("bit string must contain only '0' and '1'")


def gamma_encode_number(n: int) -> str:
    if not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n <= 0:
        raise ValueError("Elias gamma coding supports only positive integers")

    binary = bin(n)[2:]
    return "0" * (len(binary) - 1) + binary


def gamma_decode_number(bits: str, start: int = 0) -> tuple[int, int]:
    _ensure_bit_string(bits)
    return _gamma_decode_number_unchecked(bits, start)


def _gamma_decode_number_unchecked(bits: str, start: int = 0) -> tuple[int, int]:
    if not isinstance(start, int):
        raise TypeError("start must be an integer")
    if start < 0 or start >= len(bits):
        raise ValueError("start position is outside the bit string")

    position = start
    while position < len(bits) and bits[position] == "0":
        position += 1

    if position >= len(bits):
        raise ValueError("invalid Elias gamma code: stop bit not found")

    zero_count = position - start
    end = position + zero_count + 1
    if end > len(bits):
        raise ValueError("invalid Elias gamma code: incomplete binary payload")

    return int(bits[position:end], 2), end


def gamma_encode_list(numbers: list[int]) -> str:
    return "".join(gamma_encode_number(number) for number in numbers)


def gamma_decode_list(bits: str) -> list[int]:
    _ensure_bit_string(bits)
    numbers: list[int] = []
    position = 0
    while position < len(bits):
        number, position = _gamma_decode_number_unchecked(bits, position)
        numbers.append(number)
    return numbers
