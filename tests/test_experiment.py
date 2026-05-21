from compressed_inverted_index.experiment import generate_documents, run_experiment


def test_generate_documents_is_reproducible_and_bilingual() -> None:
    first = generate_documents(30, seed=123)
    second = generate_documents(30, seed=123)

    assert first == second
    assert any("ректор СПбГУ" in text for text in first.values())
    assert any("ректор МГУ" in text for text in first.values())
    assert any("university rector" in text for text in first.values())


def test_run_experiment_returns_expected_metrics() -> None:
    results = run_experiment(document_count=60)

    assert results["document_count"] == 60
    assert results["storage_size_bytes"]["uncompressed"] > 0
    assert results["storage_size_bytes"]["compressed"] > 0
    assert results["compression_ratio"] > 0

    for query in ["ректор СПбГУ", "ректор МГУ", "university rector"]:
        metrics = results["search"][query]
        assert metrics["uncompressed_time_seconds"] >= 0
        assert metrics["compressed_time_seconds"] >= 0
        assert metrics["result_count"] == len(metrics["results"])
        assert metrics["results_equal"] is True
