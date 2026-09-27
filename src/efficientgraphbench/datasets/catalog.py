"""Dataset registration without importing PyTorch or downloading anything."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    loader: str
    argument: str
    split: str
    notes: str = ""


DATASETS = {
    "cora": DatasetSpec("Cora", "planetoid", "Cora", "public"),
    "citeseer": DatasetSpec("CiteSeer", "planetoid", "CiteSeer", "public"),
    "pubmed": DatasetSpec("PubMed", "planetoid", "PubMed", "public"),
    "amazon-computers": DatasetSpec("Amazon Computers", "amazon", "Computers", "generated"),
    "amazon-photo": DatasetSpec("Amazon Photo", "amazon", "Photo", "generated"),
    "coauthor-cs": DatasetSpec("Coauthor CS", "coauthor", "CS", "generated"),
    "coauthor-physics": DatasetSpec("Coauthor Physics", "coauthor", "Physics", "generated"),
    "wikics": DatasetSpec("WikiCS", "wikics", "", "official (20 choices)"),
    "flickr": DatasetSpec("Flickr", "flickr", "", "official", "More RAM; transductive evaluation"),
    "ogbn-arxiv": DatasetSpec(
        "ogbn-arxiv", "ogb", "ogbn-arxiv", "official time split", "Optional ogb package; more RAM"
    ),
    "custom": DatasetSpec("Custom graph", "custom", "", "user-provided", "CSV folder or NPZ"),
    "ogbn-products": DatasetSpec(
        "ogbn-products",
        "ogb",
        "ogbn-products",
        "official sales-rank split",
        "Optional ogb package; very large full-batch graph",
    ),
    "roman-empire": DatasetSpec(
        "Roman Empire", "heterophilous", "Roman-empire", "official (10 choices)"
    ),
    "amazon-ratings": DatasetSpec(
        "Amazon Ratings", "heterophilous", "Amazon-ratings", "official (10 choices)"
    ),
    "actor": DatasetSpec("Actor", "actor", "", "official (10 choices)"),
    "texas": DatasetSpec("Texas", "webkb", "Texas", "official (10 choices)"),
    "wisconsin": DatasetSpec("Wisconsin", "webkb", "Wisconsin", "official (10 choices)"),
}

OFFICIAL_SPLIT_COUNTS = {
    "wikics": 20,
    **dict.fromkeys(("roman-empire", "amazon-ratings", "actor", "texas", "wisconsin"), 10),
}


def canonical_dataset(name: str) -> str:
    return name.strip().lower().replace("_", "-").replace(" ", "-")
