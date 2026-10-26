"""Exclusive wall-clock accounting; GPU times are deliberately not added here."""

import heapq
from collections import defaultdict


def category(phase):
    failed = phase.startswith("failed_")
    name = phase.removeprefix("failed_")
    if name in {
        "optimization_seconds",
        "training_forward",
        "loss_calculation",
        "training_backward",
        "optimizer_step",
        "zero_gradients",
        "scheduler_step",
    }:
        value = "training"
    elif "checkpoint" in name:
        value = "checkpoint_operations"
    elif "evaluation" in name or "validation" in name:
        value = "evaluation"
    elif name in {
        "dataset_loading",
        "data_validation",
        "graph_conversion",
        "mask_preparation",
        "graph_stage_construction",
        "structural_preprocessing",
    }:
        value = "preprocessing"
    elif name in {
        "dependency_imports",
        "reference_initialization",
        "environment_verification",
        "hardware_discovery",
        "device_initialization",
        "model_initialization",
    }:
        value = "initialization"
    elif "unattributed" in name:
        value = "unattributed"
    elif name in {"worker_attempt", "subprocess_lifetime", "attempt_lifetime"}:
        value = "attempt_lifetime_scope"
    else:
        value = "other_measured_execution"
    return ("failed_attempt_" if failed else "successful_") + value


def reconcile(events, attempt_status=None):
    """Partition each completed root interval into deepest active measured spans.

    Clock domains are independent. Incomplete spans carry no invented duration.
    Ancestor gaps remain explicitly unattributed rather than assumed training.
    """
    domains = defaultdict(list)
    incomplete = 0
    for event in events:
        if event.get("duration_ns") is None or event.get("end_offset_ns") is None:
            incomplete += 1
            continue
        start, end = event["start_offset_ns"], event["end_offset_ns"]
        if not isinstance(start, int) or not isinstance(end, int) or end < start:
            raise ValueError("Invalid event interval")
        domains[event["clock_domain"]].append(event)
    output = []
    for domain, rows in domains.items():
        by_id = {r["event_id"]: r for r in rows}

        def depth(row):
            seen = {row["event_id"]}
            while row.get("parent_event_id") in by_id:
                row = by_id[row["parent_event_id"]]
                if row["event_id"] in seen:
                    raise ValueError("Cyclic event ancestry")
                seen.add(row["event_id"])
            return len(seen)

        depths = {r["event_id"]: depth(r) for r in rows}
        points = sorted({r[k] for r in rows for k in ("start_offset_ns", "end_offset_ns")})
        totals = defaultdict(int)
        starts, ends = defaultdict(list), defaultdict(list)
        for i, row in enumerate(rows):
            if row["end_offset_ns"] > row["start_offset_ns"]:
                starts[row["start_offset_ns"]].append(i)
                ends[row["end_offset_ns"]].append(i)
        active, heap, counts = set(), [], defaultdict(int)
        failed_roots = 0
        for start, end in zip(points, points[1:]):
            for i in ends[start]:
                active.discard(i)
                counts[depths[rows[i]["event_id"]]] -= 1
                if rows[i].get("parent_event_id") is None and rows[i]["status"] not in {
                    "success",
                    "SUCCESS",
                }:
                    failed_roots -= 1
            for i in starts[start]:
                active.add(i)
                d = depths[rows[i]["event_id"]]
                counts[d] += 1
                heapq.heappush(heap, (-d, i))
                if rows[i].get("parent_event_id") is None and rows[i]["status"] not in {
                    "success",
                    "SUCCESS",
                }:
                    failed_roots += 1
            while heap and heap[0][1] not in active:
                heapq.heappop(heap)
            if not heap:
                continue
            row = rows[heap[0][1]]
            # Ties between overlapping siblings cannot be exclusively attributed.
            if counts[depths[row["event_id"]]] > 1:
                phase = "overlapping_wall_intervals"
            else:
                phase = row["phase"]
                if phase in {
                    "attempt",
                    "worker_execution",
                    "run_execution",
                    "stage_execution",
                    "training",
                    "benchmark_lifecycle",
                }:
                    phase = "unattributed_within_measured_scope"
                if (
                    failed_roots > 0
                    or row.get("status") not in {"success", "SUCCESS"}
                    or attempt_status is not None
                    and attempt_status.lower() != "success"
                ):
                    phase = "failed_" + phase
            totals[phase] += end - start
        categories = defaultdict(int)
        for phase, duration in totals.items():
            categories[category(phase)] += duration
        output.append(
            {
                "clock_domain": domain,
                "exclusive_seconds": {k: v / 1e9 for k, v in totals.items()},
                "category_seconds": {k: v / 1e9 for k, v in categories.items()},
                "measured_union_seconds": sum(totals.values()) / 1e9,
            }
        )
    return {
        "schema_version": 1,
        "domains": output,
        "incomplete_events": incomplete,
        "note": "Each domain reconciles independently. Never add worker and parent "
        "or CPU and CUDA totals. Root gaps are unattributed.",
    }
