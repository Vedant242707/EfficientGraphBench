import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from efficientgraphbench.results.plots import (
    plot_summary,  # noqa: E402
    plot_training_time,  # noqa: E402
)


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


def test_training_curve_averages_seeds_sorts_nodes_and_excludes_oom(tmp_path):
    rows = [
        dict(
            dataset="custom",
            model="gcn",
            status="SUCCESS",
            num_nodes=n,
            training_time_sec=t,
            hardware_fingerprint="gpu1",
        )
        for n, t in [(200, 4), (100, 1), (200, 6)]
    ]
    rows.append(dict(rows[0], status="OOM", num_nodes=300, training_time_sec=99))
    rows.append(dict(rows[0], hardware_fingerprint="gpu2", training_time_sec=20))
    paths = plot_training_time(rows, tmp_path)
    assert len(paths) == 2 and all(p.stat().st_size > 1000 for p in paths)
    points = pd.read_csv(tmp_path / "nodes_vs_training_time.csv")
    assert len(points) == 3 and points.num_nodes.max() == 200
    assert set(points.training_time_mean_sec) == {1, 5, 20}
    assert points.protocol.nunique() == 2
