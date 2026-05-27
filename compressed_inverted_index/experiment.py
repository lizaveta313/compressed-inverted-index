from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from compressed_inverted_index.compressed_index import CompressedInvertedIndex
from compressed_inverted_index.data_loader import generate_documents
from compressed_inverted_index.inverted_index import InvertedIndex

QUERIES = (
    "Ректор СПбГУ",
    "Ректор МГУ",
    "ректор университет",
    "university rector",
    "студент университет",
)


def _measure_search(
    search: Callable[[str], list[int]], query: str, repeats: int = 50
) -> tuple[list[int], float]:
    result: list[int] = []
    started_at = time.perf_counter()
    for _ in range(repeats):
        result = search(query)
    elapsed = time.perf_counter() - started_at
    return result, elapsed / repeats


def run_experiment(document_count: int = 1000) -> dict[str, Any]:
    """Собирает метрики для обычного и сжатого индекса."""

    documents = generate_documents(document_count)

    plain_index = InvertedIndex()
    started_at = time.perf_counter()
    plain_index.build(documents)
    plain_build_time = time.perf_counter() - started_at

    compressed_index = CompressedInvertedIndex()
    started_at = time.perf_counter()
    compressed_index.build(documents)
    compressed_build_time = time.perf_counter() - started_at

    plain_size = plain_index.storage_size_bytes()
    compressed_size = compressed_index.storage_size_bytes()

    search_results: dict[str, dict[str, Any]] = {}
    for query in QUERIES:
        plain_results, plain_search_time = _measure_search(plain_index.search, query)
        compressed_results, compressed_search_time = _measure_search(
            compressed_index.search, query
        )
        search_results[query] = {
            "uncompressed_time_seconds": plain_search_time,
            "compressed_time_seconds": compressed_search_time,
            "result_count": len(plain_results),
            "results_equal": plain_results == compressed_results,
            "results": plain_results,
        }

    return {
        "document_count": document_count,
        "build_time_seconds": {
            "uncompressed": plain_build_time,
            "compressed": compressed_build_time,
        },
        "storage_size_bytes": {
            "uncompressed": plain_size,
            "compressed": compressed_size,
        },
        "compression_ratio": compressed_size / plain_size,
        "search": search_results,
    }


def main() -> None:
    results = run_experiment()
    build_time = results["build_time_seconds"]
    storage_size = results["storage_size_bytes"]

    print("Эксперимент со сжатым инвертированным индексом")
    print(f"Документов: {results['document_count']}")
    print("Время построения:")
    print(f"  обычный индекс: {build_time['uncompressed']:.6f} сек.")
    print(f"  сжатый индекс: {build_time['compressed']:.6f} сек.")
    print("Размер хранения:")
    print(f"  обычный индекс: {storage_size['uncompressed']} байт")
    print(f"  сжатый индекс: {storage_size['compressed']} байт")
    print(f"  compression ratio: {results['compression_ratio']:.3f}")
    print("Проверка поиска:")

    for query, metrics in results["search"].items():
        equal = "да" if metrics["results_equal"] else "нет"
        print(
            f"  {query}: найдено {metrics['result_count']}, "
            f"результаты совпали: {equal}"
        )


if __name__ == "__main__":
    main()
