from efficientgraphbench.profiling.capabilities import cupti_log_errors, gpu_capture_status


def test_cupti_failure_metadata_preserves_actual_error(tmp_path):
    log = tmp_path / "worker.log"
    log.write_text(
        "CUPTI_ERROR_NOT_INITIALIZED (15)\n"
        "CUPTI initialization failed - CUDA profiler activities will be missing\n"
        "If you see CUPTI_ERROR_INSUFFICIENT_PRIVILEGES, refer to docs\n",
        encoding="utf8",
    )
    diagnostic = {}
    gpu_capture_status(diagnostic, [], "cuda:0", log)
    assert diagnostic["gpu_profiling_status"] == "unavailable"
    assert "CUPTI_ERROR_NOT_INITIALIZED" in diagnostic["cuda_capture_reason"]
    assert "INSUFFICIENT_PRIVILEGES" not in str(diagnostic)
    assert len(diagnostic["cupti_errors"]) == 2
    gpu_capture_status(diagnostic, [{"cat": "kernel"}], "cuda:0", log)
    assert diagnostic["gpu_profiling_status"] == "available"


def test_log_missing_and_bounded(tmp_path):
    assert cupti_log_errors(tmp_path / "missing") == []
    log = tmp_path / "large.log"
    log.write_text("CUPTI_ERROR_NOT_INITIALIZED\n" + "x" * 1000000, encoding="utf8")
    assert cupti_log_errors(log) == ["CUPTI_ERROR_NOT_INITIALIZED"]
