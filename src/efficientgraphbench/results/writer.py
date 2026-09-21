import json
from pathlib import Path

import pandas as pd

from efficientgraphbench.results.schema import normalize_record
from efficientgraphbench.results.status import is_success


def read_results(output_dir: str | Path):
    path = Path(output_dir) / "raw" / "results.jsonl"
    if not path.exists():
        return []
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            record = json.loads(line)
            if not isinstance(record, dict) or "run_id" not in record:
                raise ValueError("record must contain run_id")
            records.append(record)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"corrupt result at {path}:{number}: {exc}") from exc
    return records


def latest_results(records, dataset):
    rows = [row for row in records if row.get("dataset") == dataset]
    if not rows:
        return []
    batch = rows[-1].get("batch_id")
    if batch:
        rows = [row for row in rows if row.get("batch_id") == batch]
    successful = [row for row in rows if is_success(row)]
    if successful:
        group = successful[-1]["comparison_group"]
        rows = [
            row
            for row in rows
            if row.get("comparison_group") == group or (batch and not is_success(row))
        ]
    return rows


def replace_results(records, output_dir):
    """Replace active result tables; preserve historical logs and checkpoints."""
    root = Path(output_dir) / "raw"
    root.mkdir(parents=True, exist_ok=True)
    previous = read_results(output_dir)
    encoded = "".join(json.dumps(row, allow_nan=False) + "\n" for row in records)
    temp = root / "results.jsonl.tmp"
    temp.write_text(encoded, encoding="utf-8")
    temp.replace(root / "results.jsonl")
    flattened = [
        {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()}
        for row in records
    ]
    temp = root / "results.csv.tmp"
    pd.DataFrame(flattened).to_csv(temp, index=False)
    temp.replace(root / "results.csv")
    retained = {row["run_id"] for row in records}
    for row in previous:
        identifier = row["run_id"]
        # Only remove an obsolete per-run JSON directly inside this raw directory.
        if identifier not in retained and Path(identifier).name == identifier:
            path = root / f"{identifier}.json"
            if path.resolve().parent == root.resolve():
                path.unlink(missing_ok=True)


def write_result(record, output_dir):
    """Single-writer storage. JSONL is canonical; CSV is rebuilt atomically."""
    normalize_record(record)
    root = Path(output_dir) / "raw"
    root.mkdir(parents=True, exist_ok=True)
    records = read_results(output_dir)
    if any(row["run_id"] == record["run_id"] for row in records):
        raise ValueError("duplicate run_id")
    dataset = record.get("dataset")
    batch = record.get("batch_id")
    if dataset and batch:
        records = [
            row
            for row in records
            if row.get("dataset") != dataset
            or (
                row.get("batch_id") == batch
                and not (
                    row.get("model") == record.get("model")
                    and row.get("seed") == record.get("seed")
                )
            )
        ]
    elif dataset:
        records = [
            row
            for row in records
            if not (
                row.get("dataset") == dataset
                and row.get("model") == record.get("model")
                and row.get("seed") == record.get("seed")
            )
        ]
    encoded = json.dumps(record, allow_nan=False)
    (root / f"{record['run_id']}.json").write_text(encoded + "\n", encoding="utf-8")
    records.append(record)
    replace_results(records, output_dir)
