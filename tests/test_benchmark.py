import csv

from compressed_inverted_index.benchmark import (
    CSV_COLUMNS,
    generate_markdown_report,
    run_benchmark,
    save_csv,
)
from compressed_inverted_index.compressed_index import CompressedInvertedIndex
from compressed_inverted_index.inverted_index import InvertedIndex


def test_benchmark_creates_csv_and_markdown_report(tmp_path) -> None:
    documents = {
        0: "Ректор СПбГУ university rector",
        1: "Ректор МГУ университет",
        2: "студент университет",
    }
    rows = run_benchmark(documents, sizes=[2, 3], search_repeats=2)
    csv_path = tmp_path / "benchmark_results.csv"
    report_path = tmp_path / "experiment_report.md"

    save_csv(rows, csv_path)
    generate_markdown_report(rows, "тестовый JSONL", [2, 3], report_path)

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        assert tuple(reader.fieldnames or ()) == CSV_COLUMNS
        assert len(list(reader)) == 2

    report = report_path.read_text(encoding="utf-8")
    assert "# Отчёт по экспериментальному тестированию" in report
    assert "Ректор СПбГУ" in report
    assert "Ректор МГУ" in report


def test_benchmark_search_results_match_between_indexes() -> None:
    documents = {
        0: "Ректор СПбГУ",
        1: "Ректор МГУ",
        2: "Ректор СПбГУ университет",
    }
    plain = InvertedIndex()
    compressed = CompressedInvertedIndex()
    plain.build(documents)
    compressed.build(documents)

    for query in ["Ректор СПбГУ", "Ректор МГУ", "ректор университет"]:
        assert plain.search(query) == compressed.search(query)
