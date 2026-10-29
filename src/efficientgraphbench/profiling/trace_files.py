"""Identity-bearing trace names; compatible with isolated Python 3.10 workers."""

import re
import uuid


def component(value):
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(value if value is not None else "unknown"))[:24]


def capture_path(recorder, epoch):
    capture_id = uuid.uuid4().hex
    identity = recorder.identity
    name = (
        "-".join(
            [
                "b" + component(identity.get("benchmark_id")),
                component(identity.get("model")),
                "seed" + component(identity.get("seed")),
                "stage" + component(identity.get("stage")),
                "epoch" + str(epoch),
                component(identity.get("role")),
                recorder.domain[:12],
                capture_id,
            ]
        )
        + ".trace.json"
    )
    return recorder.path.with_name(name), capture_id
