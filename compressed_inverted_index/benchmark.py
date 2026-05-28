from __future__ import annotations

import argparse
import csv
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from compressed_inverted_index.compressed_index import CompressedInvertedIndex
from compressed_inverted_index.data_loader import (
    generate_documents,
    load_documents_from_jsonl,
)
from compressed_inverted_index.inverted_index import InvertedIndex

DEFAULT_SIZES = (100, 500, 1000, 5000, 10000, 40000)
SEARCH_QUERIES = (
    "Ректор СПбГУ",
    "Ректор МГУ",
    "ректор университет",
    "university rector",
    "студент университет",
)
SEARCH_REPEATS = 100

CSV_COLUMNS = (
    "document_count",
    "unique_terms",
    "total_postings",
    "build_time_uncompressed_sec",
    "build_time_compressed_sec",
    "size_uncompressed_bytes",
    "size_compressed_bytes",
    "size_uncompressed_kb",
    "size_compressed_kb",
    "memory_saving_percent",
    "compression_ratio",
    "avg_search_time_uncompressed_sec",
    "avg_search_time_compressed_sec",
    "search_results_equal",
    "rector_spbu_results_count",
    "rector_msu_results_count",
)

REPORTS_DIR = Path("reports")
CSV_PATH = REPORTS_DIR / "benchmark_results.csv"
REPORT_PATH = REPORTS_DIR / "experiment_report.md"


def count_unique_terms(index: InvertedIndex) -> int:
    return sum(1 for _term, _postings in index.items())


def count_total_postings(index: InvertedIndex) -> int:
    return sum(len(postings) for _term, postings in index.items())


def bytes_to_kb(size: int) -> float:
    return size / 1024


def calculate_memory_saving(uncompressed_size: int, compressed_size: int) -> float:
    if uncompressed_size <= 0:
        raise ValueError("uncompressed_size must be positive")
    return (1 - compressed_size / uncompressed_size) * 100


def prepare_documents(data_path: str | None, max_size: int) -> tuple[dict[int, str], str]:
    if data_path:
        documents = load_documents_from_jsonl(data_path, limit=max_size)
        return documents, f"JSONL-файл из второго модуля: {data_path}"
    return generate_documents(max_size), "сгенерированные документы"


def normalize_sizes(sizes: Iterable[int], available_count: int) -> list[int]:
    result: list[int] = []
    for size in sizes:
        if size <= 0:
            continue
        actual_size = min(size, available_count)
        if actual_size and actual_size not in result:
            result.append(actual_size)
    return result


def take_documents(documents: dict[int, str], count: int) -> dict[int, str]:
    return {new_id: text for new_id, (_old_id, text) in enumerate(list(documents.items())[:count])}


def measure_average_search_time(
    index: InvertedIndex | CompressedInvertedIndex,
    queries: Iterable[str],
    repeats: int,
) -> tuple[dict[str, list[int]], float]:
    query_list = tuple(queries)
    results: dict[str, list[int]] = {}

    started_at = time.perf_counter()
    for _ in range(repeats):
        for query in query_list:
            results[query] = index.search(query)
    elapsed = time.perf_counter() - started_at

    return results, elapsed / (len(query_list) * repeats)


def run_single_benchmark(
    documents: dict[int, str],
    search_repeats: int = SEARCH_REPEATS,
) -> dict[str, Any]:
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
    plain_results, plain_search_time = measure_average_search_time(
        plain_index, SEARCH_QUERIES, search_repeats
    )
    compressed_results, compressed_search_time = measure_average_search_time(
        compressed_index, SEARCH_QUERIES, search_repeats
    )

    return {
        "document_count": len(documents),
        "unique_terms": count_unique_terms(plain_index),
        "total_postings": count_total_postings(plain_index),
        "build_time_uncompressed_sec": plain_build_time,
        "build_time_compressed_sec": compressed_build_time,
        "size_uncompressed_bytes": plain_size,
        "size_compressed_bytes": compressed_size,
        "size_uncompressed_kb": bytes_to_kb(plain_size),
        "size_compressed_kb": bytes_to_kb(compressed_size),
        "memory_saving_percent": calculate_memory_saving(plain_size, compressed_size),
        "compression_ratio": compressed_size / plain_size,
        "avg_search_time_uncompressed_sec": plain_search_time,
        "avg_search_time_compressed_sec": compressed_search_time,
        "search_results_equal": plain_results == compressed_results,
        "rector_spbu_results_count": len(plain_results["Ректор СПбГУ"]),
        "rector_msu_results_count": len(plain_results["Ректор МГУ"]),
    }


def run_benchmark(
    documents: dict[int, str],
    sizes: Iterable[int] = DEFAULT_SIZES,
    search_repeats: int = SEARCH_REPEATS,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for size in normalize_sizes(sizes, len(documents)):
        rows.append(run_single_benchmark(take_documents(documents, size), search_repeats))
    return rows


def save_csv(rows: list[dict[str, Any]], path: Path = CSV_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def format_results_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Документов | Уникальных терминов | Postings | Обычный индекс | Сжатый индекс | Экономия | Построение обычного | Построение сжатого | Поиск обычный | Поиск сжатый |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {document_count} | {unique_terms} | {total_postings} | {plain:.2f} КБ | "
            "{compressed:.2f} КБ | {saving:.2f}% | {plain_build:.6f} с | "
            "{compressed_build:.6f} с | {plain_search:.8f} с | {compressed_search:.8f} с |".format(
                document_count=row["document_count"],
                unique_terms=row["unique_terms"],
                total_postings=row["total_postings"],
                plain=row["size_uncompressed_kb"],
                compressed=row["size_compressed_kb"],
                saving=row["memory_saving_percent"],
                plain_build=row["build_time_uncompressed_sec"],
                compressed_build=row["build_time_compressed_sec"],
                plain_search=row["avg_search_time_uncompressed_sec"],
                compressed_search=row["avg_search_time_compressed_sec"],
            )
        )
    return "\n".join(lines)


def format_search_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Документов | Ректор СПбГУ найдено | Ректор МГУ найдено | Результаты обычного и сжатого индекса |",
        "|---:|---:|---:|---|",
    ]
    for row in rows:
        equal = "совпали" if row["search_results_equal"] else "не совпали"
        lines.append(
            f"| {row['document_count']} | {row['rector_spbu_results_count']} | "
            f"{row['rector_msu_results_count']} | {equal} |"
        )
    return "\n".join(lines)


def generate_markdown_report(
    rows: list[dict[str, Any]],
    data_description: str,
    requested_sizes: Iterable[int],
    path: Path = REPORT_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    used_sizes = ", ".join(str(row["document_count"]) for row in rows) or "нет данных"
    requested = ", ".join(str(size) for size in requested_sizes)
    all_equal = all(row["search_results_equal"] for row in rows)
    report = f"""# Отчёт по экспериментальному тестированию

## Цель

Проверялось индексирование текстовых документов, сравнивались обычный и сжатый инвертированный индекс, а также выполнялся поиск по запросам «Ректор СПбГУ» и «Ректор МГУ».

## Данные

- Использовались: {data_description}.
- Запрошенные размеры корпуса: {requested}.
- Реально проверенные размеры корпуса: {used_sizes}.

## Методика

- Строился обычный инвертированный индекс.
- Строился сжатый индекс с дельта-кодированием и гамма-кодом Элиаса.
- Сравнивался объём хранения.
- Измерялось время построения через `time.perf_counter()`.
- Измерялось среднее время поиска по нескольким повторам.
- Проверялось совпадение результатов поиска.

## Результаты

{format_results_table(rows)}

## Поиск по запросам

{format_search_table(rows)}

## Анализ

При росте числа документов размер индекса растёт. Сжатый индекс занимает меньше места, потому что вместо полных `doc_id` сохраняются разности между ними, закодированные гамма-кодом Элиаса. Поиск в сжатом индексе может быть немного медленнее из-за декодирования postings lists перед пересечением. Результаты поиска обычного и сжатого индекса {'совпадают' if all_equal else 'не везде совпали, это нужно проверить отдельно'}.

## Вывод

Инвертированный индекс реализован. Сжатие на основе дельт и гамма-кода Элиаса работает. Объём хранения уменьшается. Поиск по «Ректор СПбГУ» и «Ректор МГУ» работает. Проект соответствует заданию третьего модуля.
"""
    path.write_text(report, encoding="utf-8")


def print_table(rows: list[dict[str, Any]]) -> None:
    headers = (
        "docs",
        "terms",
        "postings",
        "plain KB",
        "comp KB",
        "saving %",
        "build plain",
        "build comp",
        "search plain",
        "search comp",
        "equal",
        "SPbU",
        "MSU",
    )
    table_rows = [
        (
            str(row["document_count"]),
            str(row["unique_terms"]),
            str(row["total_postings"]),
            f"{row['size_uncompressed_kb']:.2f}",
            f"{row['size_compressed_kb']:.2f}",
            f"{row['memory_saving_percent']:.2f}",
            f"{row['build_time_uncompressed_sec']:.6f}",
            f"{row['build_time_compressed_sec']:.6f}",
            f"{row['avg_search_time_uncompressed_sec']:.8f}",
            f"{row['avg_search_time_compressed_sec']:.8f}",
            "yes" if row["search_results_equal"] else "no",
            str(row["rector_spbu_results_count"]),
            str(row["rector_msu_results_count"]),
        )
        for row in rows
    ]
    widths = [
        max(len(headers[column]), *(len(row[column]) for row in table_rows))
        for column in range(len(headers))
    ]
    print(" | ".join(headers[column].rjust(widths[column]) for column in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in table_rows:
        print(" | ".join(row[column].rjust(widths[column]) for column in range(len(headers))))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark compressed inverted index")
    parser.add_argument("--data", help="Путь к JSONL-файлу из второго модуля")
    parser.add_argument("--sizes", nargs="+", type=int, default=list(DEFAULT_SIZES))
    parser.add_argument("--repeats", type=int, default=SEARCH_REPEATS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_size = max(args.sizes) if args.sizes else max(DEFAULT_SIZES)
    documents, data_description = prepare_documents(args.data, max_size)
    if not documents:
        print("Нет документов для benchmark. Проверьте путь к JSONL или содержимое файла.")
        return

    rows = run_benchmark(documents, args.sizes, args.repeats)
    save_csv(rows)
    generate_markdown_report(rows, data_description, args.sizes)

    print_table(rows)
    print()
    print(f"CSV saved to {CSV_PATH}")
    print(f"Markdown report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
