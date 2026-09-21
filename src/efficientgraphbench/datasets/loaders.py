"""Public dataset loading with the same canonical validation as the CLI."""

from pathlib import Path

from efficientgraphbench.datasets.custom import _load_csv
from efficientgraphbench.datasets.registry import DatasetBundle
from efficientgraphbench.datasets.registry import load_dataset as _load
from efficientgraphbench.datasets.validation import graph_fingerprint, validate_graph


class InvalidDatasetError(ValueError):
    """The supplied graph does not satisfy the node-classification contract."""


def load_dataset(name, root="data", *, nodes=None, edges=None, **options):
    try:
        if nodes is not None or edges is not None:
            if name.lower() != "custom" or nodes is None or edges is None:
                raise ValueError("Supply both nodes and edges with name='custom'")
            if options:
                raise ValueError(
                    "Named CSV inputs already contain splits; do not supply dataset_path or split options"  # noqa: E501
                )
            data, classes, mapping = _load_csv(Path("."), Path(nodes), Path(edges))
            validate_graph(data, classes)
            return DatasetBundle(
                data,
                classes,
                split="user-provided",
                preprocessing="as-provided",
                fingerprint=graph_fingerprint(data),
                label_mapping=mapping,
            )
        return _load(name, root, **options)
    except (ValueError, OSError) as exc:
        raise InvalidDatasetError(str(exc)) from exc
