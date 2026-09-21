"""Stable run outcomes and compatibility with historical result files."""


class UnsupportedGraphSize(Exception):
    def __init__(self, message, limit):
        super().__init__(message)
        self.limit = limit


def status_of(record):
    status = record.get("status", "RUNTIME_ERROR")
    if status == "oom" and "at most 4096 nodes" in record.get("error", ""):
        return "UNSUPPORTED_GRAPH_SIZE"
    return {
        "ok": "SUCCESS",
        "oom": "OOM",
        "error": "RUNTIME_ERROR",
        "running": "RUNTIME_ERROR",
    }.get(status, status)


def is_success(record):
    return status_of(record) == "SUCCESS"


def failure_reason(record):
    return record.get("failure_reason") or record.get("error") or "Run did not complete"
