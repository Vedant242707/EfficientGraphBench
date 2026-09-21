"""Primary identities never silently fall back to experimental architectures."""

from efficientgraphbench.environments.manager import environment_spec
from efficientgraphbench.paths import home

ROOT = home()
PRIMARY_MODELS = (
    "gcn",
    "graphsage",
    "gat",
    "gatv2",
    "appnp",
    "sgc",
    "mlp",
    "sgformer",
    "graphgps",
    "graphormer",
)
EXPERIMENTAL_MODELS = ("graphgps_adapted", "graphormer_adapted", "appnp_adapted", "sgformer_pyg")


def model_spec(name):
    if name == "appnp":
        from efficientgraphbench.models.reference_appnp import COMMIT

        return {
            "runner": "in_process",
            "category": "primary",
            "source_type": "official_repo",
            "implementation_type": "reference_repository",
            "environment": "pyg",
            "repository_url": "https://github.com/pyg-team/pytorch_geometric",
            "commit": COMMIT,
            "architecture_modified": False,
        }
    if name in {"graphgps", "sgformer", "sgformer_reference"}:
        env = "graphgps" if name == "graphgps" else "sgformer"
        spec = environment_spec(env)
        return {
            **spec,
            "runner": "repository",
            "category": "primary",
            "source_type": "official_repo",
            "implementation_type": "reference_repository",
        }
    if name == "graphormer":
        return {
            "runner": "unsupported",
            "category": "primary",
            "source_type": "official_repo",
            "implementation_type": "reference_repository",
            "environment": "graphormer",
            "repository_url": "https://github.com/microsoft/Graphormer",
            "reason": "Official molecular/graph prediction pipeline has no validated adapter "
            "for continuous-feature citation node classification. No architecture "
            "replacement is permitted; use graphormer_adapted explicitly.",
        }
    experimental = name.endswith("_adapted")
    return {
        "runner": "in_process",
        "environment": "pyg",
        "category": "experimental"
        if experimental
        else ("baseline" if name == "mlp" else "primary"),
        "source_type": "adapted"
        if name in {"graphgps_adapted", "appnp_adapted"}
        else ("custom_reimplementation" if experimental or name == "mlp" else "library"),
        "implementation_type": "experimental" if experimental else "library",
        "repository_url": None
        if experimental or name == "mlp"
        else "https://github.com/pyg-team/pytorch_geometric",
    }


def availability(name):
    spec = model_spec(name)
    if name == "appnp":
        import hashlib

        from efficientgraphbench.models.reference_appnp import SOURCE, SOURCE_SHA256

        if (
            not SOURCE.exists()
            or hashlib.sha256(SOURCE.read_text(encoding="utf-8").encode()).hexdigest()
            != SOURCE_SHA256
        ):
            return "DEPENDENCY_ERROR", "Pinned APPNP citation benchmark source is missing/modified"
    if spec["runner"] == "unsupported":
        return "UNSUPPORTED_TASK", spec["reason"]
    if spec["runner"] == "repository":
        if not spec.get("python") or not (ROOT / spec["python"]).exists():
            return "ENVIRONMENT_NOT_INSTALLED", "Run egbench setup " + spec["environment"]
        if not spec.get("repository") or not (ROOT / spec["repository"]).exists():
            return "DEPENDENCY_ERROR", "Pinned reference repository is not installed"
        if not spec.get("validated"):
            return "unverified", "Installed; reference smoke validation not yet completed"
    return "available", spec["source_type"]
