"""Capability declarations for verified existing adapters, not substitute models."""

from efficientgraphbench.models.catalog import model_spec

# Explicit allowlist: future in-process adapters are unsupported until verified.
VERIFIED_DYNAMIC_ADAPTERS = frozenset(
    {
        "gcn",
        "graphsage",
        "gat",
        "gatv2",
        "mlp",
        "sgc",
        "appnp",
        "sgformer_pyg",
    }
)


def progressive_capabilities(name: str) -> dict:
    spec = model_spec(name)
    supported = (name in VERIFIED_DYNAMIC_ADAPTERS and spec["runner"] == "in_process") or name in {
        "sgformer",
        "sgformer_reference",
        "graphgps",
    }
    return {
        "supports_progressive_resume": supported,
        "supports_new_nodes": supported,
        "supports_optimizer_resume": supported,
        "supports_scheduler_resume": supported,
        "supports_graph_expansion": supported,
        "supports_dynamic_graph_expansion": supported,
        "reason": None if supported else "progressive_resume_unsupported",
        "evaluation_semantics": "expanded_transductive_retraining",
        "runner": spec["runner"],
    }
