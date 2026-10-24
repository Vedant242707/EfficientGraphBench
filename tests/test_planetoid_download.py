import pytest
from torch_geometric.datasets import Planetoid
from torch_geometric.io import fs

from efficientgraphbench.datasets import registry
from efficientgraphbench.datasets.download import DirectPlanetoid


@pytest.mark.parametrize("name", ["Cora", "CiteSeer", "PubMed"])
def test_planetoid_uses_direct_upstream_files(monkeypatch, tmp_path, name):
    # Exercise the inherited downloader without network or dataset processing.
    dataset = DirectPlanetoid.__new__(DirectPlanetoid)
    dataset.root, dataset.name, dataset.split = str(tmp_path), name, "public"
    calls = []
    monkeypatch.setattr(fs, "cp", lambda source, target: calls.append((source, target)))
    dataset.download()
    assert len(calls) == 8
    assert all(
        source == f"https://raw.githubusercontent.com/kimiyoung/planetoid/master/data/{filename}"
        and target == dataset.raw_dir
        for (source, target), filename in zip(calls, dataset.raw_file_names)
    )
    assert registry.Planetoid is DirectPlanetoid
    assert DirectPlanetoid.process is Planetoid.process
    assert DirectPlanetoid.raw_dir is Planetoid.raw_dir
    assert DirectPlanetoid.processed_dir is Planetoid.processed_dir
