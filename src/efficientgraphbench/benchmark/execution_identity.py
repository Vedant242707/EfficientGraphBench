"""Execution identity excludes presentation, documentation and output paths."""

import json
import platform
import subprocess
from functools import lru_cache
from pathlib import Path

from efficientgraphbench.benchmark.interchange import file_sha256, fingerprint

TIMING_REVISION = "disjoint-optimization-evaluation-v2"


def execution_sources():
    root = Path(__file__).resolve().parents[1]
    folders = {
        "benchmark",
        "training",
        "models",
        "datasets",
        "profiling",
        "runners",
        "progressive",
        "environments",
    }
    return {
        p.relative_to(root).as_posix(): file_sha256(p)
        for p in sorted(root.rglob("*.py"))
        if (p.relative_to(root).parts[0] in folders or p.name == "config.py")
        and p.name not in {"results.py", "plots.py", "report.py", "profile.py"}
    }


@lru_cache(maxsize=8)
def runtime_versions(python=None):
    if python:
        code = (
            "import json,platform,torch,numpy,scipy,torch_geometric; "
            "print(json.dumps(dict(python=platform.python_version(),"
            "torch=str(torch.__version__),pyg=torch_geometric.__version__,"
            "numpy=numpy.__version__,scipy=scipy.__version__,"
            "cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version())))"
        )
        return json.loads(
            subprocess.check_output([python, "-c", code], text=True, timeout=120)
            .strip()
            .splitlines()[-1]
        )
    import numpy
    import scipy
    import torch
    import torch_geometric

    return dict(
        python=platform.python_version(),
        torch=str(torch.__version__),
        pyg=torch_geometric.__version__,
        numpy=numpy.__version__,
        scipy=scipy.__version__,
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
    )


def execution_identity(config, graph, hardware, spec):
    recipe = config.to_dict() if hasattr(config, "to_dict") else config
    material = {
        k: recipe.get(k)
        for k in (
            "model",
            "training",
            "model_options",
            "threads",
            "split_mode",
            "split_index",
            "split_seed",
            "seed",
        )
    }
    payload = dict(
        graph=graph["canonical_graph_hash"],
        split=graph["split_hash"],
        graph_semantics="static-transductive-node-classification",
        comparison_reference_protocol="full",
        recipe=material,
        hardware=hardware,
        source_type=spec["source_type"],
        commit=spec.get("commit"),
        runtime=runtime_versions(spec.get("python") if spec["runner"] == "repository" else None),
        execution_sources=execution_sources(),
        timing_revision=TIMING_REVISION,
    )
    return {"fingerprint": fingerprint(payload), "definition": payload}
