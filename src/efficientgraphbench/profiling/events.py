"""Failure-safe wall-clock events. Importable by the pinned Python 3.10 workers."""

import contextvars
import functools
import json
import os
import sys
import time
import uuid
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path

CURRENT = contextvars.ContextVar("execution_recorder", default=None)


def options_for(config):
    value = getattr(config, "profiling", {})
    return value if isinstance(value, dict) else {}


def utc():
    return datetime.now(timezone.utc).isoformat()


def error_status(exc):
    if isinstance(exc, KeyboardInterrupt):
        return "interrupted"
    if (
        isinstance(exc, MemoryError)
        or "outofmemory" in type(exc).__name__.lower()
        or "out of memory" in str(exc).lower()
    ):
        return "oom"
    if getattr(exc, "status", None) == "RESOURCE_LIMIT":
        return "timeout"
    return "failed"


class EventRecorder:
    def __init__(self, path, identity, options=None):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.identity = dict(identity)
        self.options = options if isinstance(options, dict) else {}
        self.origin = time.perf_counter_ns()
        self.domain = uuid.uuid4().hex
        self.stack = []
        self.events = []
        self.last_completed_epoch = None
        self.artifacts = []
        self.notes = []
        self.handle = self.path.open("a", encoding="utf8")

    def emit(self, value):
        self.handle.write(json.dumps(value, allow_nan=False) + "\n")
        self.handle.flush()

    @contextmanager
    def span(self, phase, **attributes):
        event = {
            **self.identity,
            "schema_version": 1,
            "clock_domain": self.domain,
            "event_id": uuid.uuid4().hex,
            "parent_event_id": self.stack[-1]["event_id"] if self.stack else None,
            "phase": phase,
            "status": "running",
            "pid": os.getpid(),
            "environment": sys.executable,
            "wall_start_utc": utc(),
            "start_offset_ns": time.perf_counter_ns() - self.origin,
            "timing_source": "monotonic_wall_clock",
            "synchronization": "none",
            **attributes,
        }
        self.emit({**event, "record_type": "start"})
        self.stack.append(event)
        try:
            yield event
        except BaseException as exc:
            event.update(
                status=error_status(exc),
                error_type=type(exc).__name__,
                failure_reason=str(exc)[:2000],
            )
            raise
        finally:
            event["end_offset_ns"] = time.perf_counter_ns() - self.origin
            event["duration_ns"] = event["end_offset_ns"] - event["start_offset_ns"]
            event["wall_end_utc"] = utc()
            if event["status"] == "running":
                event["status"] = "success"
            self.stack.pop()
            self.events.append(event)
            self.emit({**event, "record_type": "end"})

    def summary(self):
        from efficientgraphbench.profiling.accounting import reconcile

        return {
            "accounting": reconcile(self.events),
            "schema_version": 1,
            "events_path": str(self.path),
            "clock_domain": self.domain,
            "events": self.events,
            "last_completed_epoch": self.last_completed_epoch,
            "failure_phase": next(
                (e["phase"] for e in self.events if e.get("status") not in {"success", "running"}),
                None,
            ),
            "instrumentation": self.options,
            "overhead": "Included in measured durations; no per-workload calibration applied",
            "traces": self.artifacts,
            "notes": self.notes,
            "scope": "process-local; nested spans inclusive; excludes final result serialization",
        }

    def close(self):
        self.handle.close()


def span(phase, **attributes):
    recorder = CURRENT.get()
    return recorder.span(phase, **attributes) if recorder else nullcontext({})


def observed(phase):
    def decorate(function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            with span(phase):
                return function(*args, **kwargs)

        return wrapped

    return decorate


def read_journal(path):
    """Recover completed spans and leave crashed/partial spans unavailable."""
    events = {}
    if not Path(path).exists():
        return []
    for line in Path(path).read_text(encoding="utf8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except (ValueError, TypeError):
            continue  # A killed writer may leave a truncated final line.
        if not isinstance(row, dict) or row.get("schema_version") != 1 or "event_id" not in row:
            continue
        if row.get("record_type") == "start":
            row.update(status="incomplete", end_offset_ns=None, duration_ns=None)
        events[row["event_id"]] = row
    return list(events.values())


def attempt(mode):
    """Instrument existing entry points without modifying their training contracts."""

    def decorate(function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            if not args and mode == "full" and "config" in kwargs:
                supplied = dict(kwargs)
                return wrapped(supplied.pop("config"), **supplied)
            if not args:
                return function(*args, **kwargs)
            if CURRENT.get() is not None and CURRENT.get().identity.get("role") == "worker":
                with span("stage_execution" if mode != "full" else "run_execution"):
                    return function(*args, **kwargs)
            config = args[0]
            if options_for(config).get("level") == "none":
                return function(*args, **kwargs)
            stage = args[2].name if mode != "full" else "full"
            output = Path(args[7]) if mode != "full" else Path(config.output_dir)
            identifier = uuid.uuid4().hex
            config._execution_run_id = identifier
            identity = dict(
                benchmark_id=(
                    CURRENT.get().identity.get("benchmark_id")
                    if CURRENT.get()
                    else getattr(config, "_result_batch", None)
                ),
                role="attempt",
                run_id=identifier,
                mode=mode,
                model=config.model.name,
                dataset=config.dataset,
                seed=config.seed,
                stage=stage,
                device=str(args[6]) if mode != "full" else config.device,
            )
            recorder = EventRecorder(
                output / "execution" / f"{identifier}.jsonl",
                identity,
                getattr(config, "profiling", None),
            )
            outer = span("attempt_lifetime", model=config.model.name, seed=config.seed, stage=stage)
            outer_event = outer.__enter__()
            token = CURRENT.set(recorder)
            result = None
            try:
                with recorder.span("attempt") as event:
                    result = function(*args, **kwargs)
                    if isinstance(result, dict):
                        event["status"] = result.get("status", "SUCCESS").lower()
                        outer_event["status"] = event["status"]
                        event["result_run_id"] = result.get("run_id")
                        if result.get("failure_reason"):
                            event["failure_reason"] = result["failure_reason"]
                result["execution"] = recorder.summary()
                result.setdefault("run_id", identifier)
                result["instrumented"] = recorder.options.get("level") == "detailed"
                # Link new events to the existing result ID; do not rewrite old runs.
                if mode == "full" and result.get("run_id"):
                    from efficientgraphbench.results.writer import write_result

                    write_result(result, config.output_dir)
                return result
            except BaseException as exc:
                measured = getattr(exc, "progressive_measurements", {})
                exc.progressive_measurements = {**measured, "execution": recorder.summary()}
                if mode == "full" and hasattr(recorder, "result"):
                    from efficientgraphbench.results.writer import write_result

                    failed = recorder.result
                    failed.update(
                        execution=recorder.summary(),
                        failure_reason=f"{type(exc).__name__}: {exc}",
                        interrupted=isinstance(exc, KeyboardInterrupt),
                    )
                    write_result(failed, config.output_dir)
                raise
            finally:
                recorder.close()
                CURRENT.reset(token)
                del config._execution_run_id
                outer.__exit__(*sys.exc_info())

        return wrapped

    return decorate


def lifecycle(mode):
    """An independent controller clock covering loading, attempts and reporting."""

    def decorate(function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            if not args and mode == "full" and "jobs" in kwargs:
                supplied = dict(kwargs)
                return wrapped(supplied.pop("jobs"), **supplied)
            if not args:
                return function(*args, **kwargs)
            config = (
                args[0][0]
                if mode == "full" and args[0]
                else (next(iter(args[0].configs.values())) if mode != "full" else None)
            )
            if config is None or options_for(config).get("level") == "none":
                return function(*args, **kwargs)
            identifier = uuid.uuid4().hex
            recorder = EventRecorder(
                Path(config.output_dir) / "execution" / (identifier + "-lifecycle.jsonl"),
                dict(
                    benchmark_id=identifier,
                    role="controller",
                    mode=mode,
                    dataset=config.dataset,
                    device=config.device,
                ),
            )
            token = CURRENT.set(recorder)
            result = None
            try:
                with recorder.span("benchmark_lifecycle"):
                    result = function(*args, **kwargs)
                if mode == "full":
                    target = Path(config.output_dir) / "full-benchmark.json"
                    if target.exists():
                        saved = json.loads(target.read_text(encoding="utf8"))
                        saved["metadata"]["execution"] = recorder.summary()
                        target.write_text(json.dumps(saved, allow_nan=False), encoding="utf8")
                else:
                    result.metadata["execution"] = recorder.summary()
                    result.save()
                return result
            finally:
                recorder.path.with_suffix(".summary.json").write_text(
                    json.dumps(recorder.summary(), allow_nan=False), encoding="utf8"
                )
                recorder.close()
                CURRENT.reset(token)

        return wrapped

    return decorate
