"""Compact presentation of incomplete experiments; raw records stay intact."""

from collections import defaultdict

from efficientgraphbench.names import dataset_name, model_name
from efficientgraphbench.results.status import failure_reason, status_of


def failure_summaries(records):
    grouped = defaultdict(list)
    for row in records:
        grouped[(row.get("dataset"), row.get("model"))].append(row)
    summaries = []
    for (dataset, model), rows in grouped.items():
        failed = [r for r in rows if status_of(r) != "SUCCESS"]
        if not failed:
            continue
        causes = [r for r in failed if not status_of(r).startswith("SKIPPED")]
        cause = (causes or failed)[0]
        skipped = sum(status_of(r).startswith("SKIPPED") for r in failed)
        succeeded = sum(status_of(r) == "SUCCESS" for r in rows)
        summaries.append(
            {
                "Dataset": dataset_name(str(dataset)),
                "Model": model_name(str(model)),
                "Status": "Incomplete" if succeeded else "Not completed",
                "Reason": f"{status_of(cause)}: " + failure_reason(cause),
                "Skipped": skipped,
                "Note": (
                    "Statistics use successful runs only; this model did not finish all attempts."
                )
                if succeeded
                else "",
            }
        )
    return summaries


def successful_records(records):
    """Keep every measured success, including successes from incomplete models."""
    return [r for r in records if status_of(r) == "SUCCESS"]
