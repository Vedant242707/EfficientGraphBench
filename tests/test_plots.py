import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from efficientgraphbench.results.plots import plot_summary  # noqa: E402


def table():
    return pd.DataFrame(
        [
            dict(
                dataset="texas",
                model="gcn",
                runs=10,
                accuracy_mean=0.6,
                accuracy_sd=0.03,
                inference_latency_median_ms=2,
                peak_gpu_allocated_mib=64,
            ),
            dict(
                dataset="texas",
                model="gat",
                runs=0,
                accuracy_mean=None,
                accuracy_sd=None,
                inference_latency_median_ms=None,
                peak_gpu_allocated_mib=None,
            ),
        ]
    )


def test_exports_keep_units_and_exclude_failures(tmp_path):
    paths = plot_summary(table(), tmp_path)
    assert len(paths) == 4
    assert all(p.exists() and p.stat().st_size > 1000 for p in paths)
    data = pd.read_csv(tmp_path / "plot_data.csv")
    assert data.model.tolist() == ["gcn"]
    assert data.accuracy_mean.tolist() == [0.6]
    assert data.peak_gpu_allocated_mib.tolist() == [64]


def test_cpu_memory_is_not_fabricated(tmp_path):
    frame = table()
    frame["peak_gpu_allocated_mib"] = None
    paths = plot_summary(frame, tmp_path)
    assert len(paths) == 2
    assert all("accuracy_vs_latency" in p.name for p in paths)


def test_rejects_mixed_configs_and_failed_only(tmp_path):
    frame = table()
    with pytest.raises(ValueError, match="Multiple groups"):
        plot_summary(pd.concat([frame, frame]), tmp_path)
    with pytest.raises(ValueError, match="No successful"):
        plot_summary(frame.iloc[1:], tmp_path)
