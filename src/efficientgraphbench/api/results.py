"""Public results retain failures and never fabricate missing metrics."""

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from efficientgraphbench.results.status import status_of


@dataclass
class BenchmarkResult:
    record: dict

    def __getattr__(self, name):
        aliases = {
            "training_time": "training_time_sec",
            "preprocessing_time": "preprocessing_time_sec",
            "inference_latency": "inference_latency_median_ms",
            "gpu_memory": "peak_gpu_allocated_mib",
            "system_memory": "peak_cpu_memory_mb",
            "model_source": "source_type",
            "latency_mean": "inference_latency_mean_ms",
            "latency_median": "inference_latency_median_ms",
            "latency_std": "inference_latency_std_ms",
            "peak_gpu_allocated": "peak_gpu_allocated_mib",
            "peak_gpu_reserved": "peak_gpu_reserved_mib",
            "peak_system_ram": "peak_cpu_memory_mb",
            "epochs": "epochs_run",
        }
        if name.startswith("_"):
            raise AttributeError(name)
        return self.record.get(aliases.get(name, name))

    def to_dict(self):
        return dict(self.record)


class BenchmarkResults:
    def __init__(self, records, output_dir=None):
        self.records = [r.to_dict() if isinstance(r, BenchmarkResult) else dict(r) for r in records]
        self.output_dir = Path(output_dir) if output_dir else None

    def __iter__(self):
        return (BenchmarkResult(r) for r in self.records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return BenchmarkResult(self.records[index])

    def to_dataframe(self):
        return pd.DataFrame(self.records)

    def summary(self):
        rows = []
        groups = {}
        for record in self.records:
            key = (record.get("dataset"), record.get("comparison_group"), record.get("model"))
            groups.setdefault(key, []).append(record)
        for (dataset, group, model), records in groups.items():
            successful = [r for r in records if status_of(r) == "SUCCESS"]
            scores = pd.Series(
                [r["accuracy"] for r in successful if r.get("accuracy") is not None], dtype=float
            )
            rows.append(
                {
                    "dataset": dataset,
                    "comparison_group": group,
                    "model": model,
                    "source": records[-1].get("source_type"),
                    "runs": len(successful),
                    "accuracy_mean": scores.mean() if len(scores) else None,
                    "accuracy_sd": (scores.std(ddof=1) if len(scores) > 1 else 0.0)
                    if len(scores)
                    else None,
                    "status": ", ".join(dict.fromkeys(status_of(r) for r in records)),
                    **{
                        key: pd.Series(
                            [r[key] for r in successful if r.get(key) is not None], dtype=float
                        ).mean()
                        if any(r.get(key) is not None for r in successful)
                        else None
                        for key in (
                            "preprocessing_time_sec",
                            "training_time_sec",
                            "inference_latency_median_ms",
                            "peak_gpu_allocated_mib",
                            "peak_gpu_reserved_mib",
                            "peak_cpu_memory_mb",
                        )
                    },
                }
            )
        return pd.DataFrame(rows)

    def _path(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def save_json(self, path):
        path = self._path(path)
        path.write_text(json.dumps(self.records, indent=2, allow_nan=False), encoding="utf-8")
        return path

    def save_jsonl(self, path):
        path = self._path(path)
        path.write_text(
            "".join(json.dumps(r, allow_nan=False) + "\n" for r in self.records), encoding="utf-8"
        )
        return path

    def save_csv(self, path):
        path = self._path(path)
        rows = [
            {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()}
            for r in self.records
        ]
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def save_markdown(self, path):
        frame = self.summary().fillna("—")

        def cell(value):
            return str(value).replace("|", "\\|").replace("\n", " ")

        lines = [
            "# EfficientGraphBench results",
            "",
            "| " + " | ".join(frame.columns) + " |",
            "| " + " | ".join(["---"] * len(frame.columns)) + " |",
        ]
        lines += [
            "| " + " | ".join(cell(v) for v in row) + " |"
            for row in frame.itertuples(index=False, name=None)
        ]
        lines += ["", "Failures and skipped runs:", ""]
        lines += [
            f"- {r.get('model')}, seed {r.get('seed')}: {status_of(r)} — {r.get('failure_reason', '')}"  # noqa: E501
            for r in self.records
            if status_of(r) != "SUCCESS"
        ]
        path = self._path(path)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path
