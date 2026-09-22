"""Load the unchanged Net class from PyG's pinned citation benchmark.

Only select the class definition: importing the script normally would parse argv,
download a dataset and start 100 training runs. No class AST nodes are rewritten.
"""

import ast
import hashlib
import sys
import types

import torch
import torch.nn.functional as F
from torch_geometric.nn import APPNP

from efficientgraphbench.paths import RESOURCES

COMMIT = "79d33965a40b7fa83616a9f598a0f8619f25d939"
SOURCE_SHA256 = "7025ef5f94c9e50beed6aef89870c0d534f02104bece23f6713edc86a231caf4"
SOURCE = RESOURCES / "vendor/appnp.py"


def build_reference_appnp(config, in_channels, out_channels):
    if not SOURCE.exists():
        raise ImportError("Missing bundled PyG APPNP source; reinstall EfficientGraphBench")
    if hashlib.sha256(SOURCE.read_text(encoding="utf-8").encode()).hexdigest() != SOURCE_SHA256:
        raise ImportError("PyG APPNP source differs from the verified reference snapshot")
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    definition = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Net"
    )
    module = types.ModuleType("efficientgraphbench_reference_appnp")
    module.__file__ = str(SOURCE)
    module.__dict__.update(
        torch=torch,
        F=F,
        Linear=torch.nn.Linear,
        APPNP=APPNP,
        args=types.SimpleNamespace(
            hidden=config.hidden_dim,
            dropout=config.dropout,
            K=config.propagation_steps,
            alpha=config.alpha,
        ),
    )
    sys.modules[module.__name__] = module
    exec(
        compile(ast.Module(body=[definition], type_ignores=[]), str(SOURCE), "exec"),
        module.__dict__,
    )
    return module.Net(types.SimpleNamespace(num_features=in_channels, num_classes=out_channels))
