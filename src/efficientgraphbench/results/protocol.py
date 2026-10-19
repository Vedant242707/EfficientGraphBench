"""Prevent stage records from entering static benchmark statistics."""


def validate_progressive_records(records):
    for row in records:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("fidelity"), dict)
            or row.get("benchmark_mode", "progressive_multi_fidelity")
            != "progressive_multi_fidelity"
        ):
            raise ValueError(
                "ProgressiveResults requires Progressive stage records; "
                "mixed protocols are not allowed"
            )


def validate_full_records(records):
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("Expected Full Benchmark record dictionaries")
        progressive = any(k in row for k in ("fidelity", "candidate_status", "adaptation"))
        if row.get("benchmark_mode", "full") != "full" or progressive:
            raise ValueError(
                "Full Benchmark aggregation cannot accept Progressive or mixed-protocol "
                "records; use ProgressiveResults"
            )
        if ("model" not in row and not ("run_id" in row and "status" in row)) or (
            "status" not in row and "accuracy" not in row
        ):
            raise ValueError("Ambiguous legacy record: model and status or accuracy required")
