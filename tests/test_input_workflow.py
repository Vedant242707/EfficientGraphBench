import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests
from typer.testing import CliRunner

from efficientgraphbench.cli.main import app
from efficientgraphbench.datasets.download import download_file, validate_download
from efficientgraphbench.datasets.prepare import prepare_csv
from efficientgraphbench.datasets.registry import load_dataset


@pytest.fixture
def source_csvs(tmp_path):
    nodes = tmp_path / "people.csv"
    edges = tmp_path / "connections.csv"
    pd.DataFrame(
        {
            "person": [f"p{i}" for i in range(12)],
            "category": ["a"] * 6 + ["b"] * 6,
            "age": range(20, 32),
            "description": ["text"] * 12,
        }
    ).to_csv(nodes, index=False)
    pd.DataFrame(
        {"from": [f"p{i}" for i in range(12)], "to": [f"p{(i + 1) % 12}" for i in range(12)]}
    ).to_csv(edges, index=False)
    return nodes, edges


def test_prepare_arbitrary_column_names_and_fixed_split(source_csvs, tmp_path):
    nodes, edges = source_csvs
    options = dict(node_id="person", label="category", source="from", target="to")
    summary = prepare_csv(nodes, edges, tmp_path / "first", **options)
    prepare_csv(nodes, edges, tmp_path / "second", **options)
    assert summary["features"] == ["age"]
    assert summary["ignored_columns"] == ["description"]
    assert (tmp_path / "first/nodes.csv").read_bytes() == (
        tmp_path / "second/nodes.csv"
    ).read_bytes()
    bundle = load_dataset("custom", tmp_path, dataset_path=tmp_path / "first")
    assert bundle.num_classes == 2
    assert bundle.data.num_node_features == 1


def test_prepare_protects_labels_and_existing_files(source_csvs, tmp_path):
    nodes, edges = source_csvs
    options = dict(node_id="person", label="category", source="from", target="to")
    with pytest.raises(ValueError, match="must not include"):
        prepare_csv(nodes, edges, tmp_path / "bad", features=["category"], **options)
    assert not (tmp_path / "bad").exists()
    target = tmp_path / "existing"
    target.mkdir()
    (target / "keep.txt").write_text("preserve me")
    with pytest.raises(ValueError, match="not overwritten"):
        prepare_csv(nodes, edges, target, **options)
    assert (target / "keep.txt").read_text() == "preserve me"


def test_prepare_preserves_user_splits(tmp_path):
    example = Path(__file__).resolve().parents[1] / "examples/custom_graph"
    prepare_csv(example / "nodes.csv", example / "edges.csv", tmp_path / "prepared")
    before = pd.read_csv(example / "nodes.csv")
    after = pd.read_csv(tmp_path / "prepared/nodes.csv")
    assert before.split.tolist() == after.split.tolist()


def test_cli_prepare_and_inspect_parameters(source_csvs, tmp_path):
    nodes, edges = source_csvs
    output = tmp_path / "prepared"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "prepare",
            "--nodes",
            str(nodes),
            "--edges",
            str(edges),
            "--node-id",
            "person",
            "--label",
            "category",
            "--source",
            "from",
            "--target",
            "to",
            "--output-dir",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    inspected = runner.invoke(
        app, ["inspect", "--dataset", "custom", "--dataset-path", str(output)]
    )
    assert inspected.exit_code == 0, inspected.output
    assert "Trainable parameters" in inspected.output
    assert "GraphSAGE" in inspected.output
    assert "Input tensors" in inspected.output


class Response:
    def __init__(self, blob, status=200, headers=None):
        self.blob = blob
        self.status_code = status
        self.headers = headers or {"Content-Length": str(len(blob))}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self.blob


def npy_bytes():
    buffer = io.BytesIO()
    np.save(buffer, np.ones((4, 3), dtype=np.float32))
    return buffer.getvalue()


@pytest.mark.parametrize("supports_resume", [True, False])
def test_download_repairs_partial_npy(tmp_path, monkeypatch, supports_resume):
    blob = npy_bytes()
    destination = tmp_path / "feats.npy"
    destination.write_bytes(blob[:80])
    assert not validate_download(destination)

    def get(url, headers, **kwargs):
        assert headers == {"Range": "bytes=80-"}
        assert kwargs["timeout"] == (15, 60)
        if supports_resume:
            return Response(
                blob[80:],
                206,
                {
                    "Content-Length": str(len(blob) - 80),
                    "Content-Range": f"bytes 80-{len(blob) - 1}/{len(blob)}",
                },
            )
        return Response(blob)

    monkeypatch.setattr("efficientgraphbench.datasets.download.requests.get", get)
    download_file("https://example.invalid/data", destination)
    assert destination.read_bytes() == blob
    assert validate_download(destination)
    assert not (tmp_path / "feats.npy.part").exists()


def test_download_timeout_preserves_partial(tmp_path, monkeypatch):
    class Interrupted(Response):
        def iter_content(self, chunk_size):
            yield self.blob[:50]
            raise requests.Timeout("stalled")

    monkeypatch.setattr(
        "efficientgraphbench.datasets.download.requests.get",
        lambda *args, **kwargs: Interrupted(npy_bytes()),
    )
    with pytest.raises(OSError, match="Partial data is kept"):
        download_file("https://example.invalid/data", tmp_path / "feats.npy")
    assert not (tmp_path / "feats.npy").exists()
    assert (tmp_path / "feats.npy.part").stat().st_size == 50


def test_download_rejects_server_html(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "efficientgraphbench.datasets.download.requests.get",
        lambda *args, **kwargs: Response(b"<html>quota exceeded</html>"),
    )
    with pytest.raises(ValueError, match="not the expected"):
        download_file("https://example.invalid/data", tmp_path / "feats.npy")
    assert not (tmp_path / "feats.npy").exists()


def test_short_prepare_detects_common_names(source_csvs, tmp_path):
    nodes, edges = source_csvs
    frame = pd.read_csv(nodes).rename(columns={"person": "person_id"})
    frame.to_csv(nodes, index=False)
    result = CliRunner().invoke(
        app,
        ["prepare", "-n", str(nodes), "-e", str(edges), "--output-dir", str(tmp_path / "short")],
    )
    assert result.exit_code == 0, result.output
    assert "person_id" in result.output
    assert "age" in result.output
    frame["id"] = frame["person_id"]
    frame.to_csv(nodes, index=False)
    result = CliRunner().invoke(
        app,
        [
            "prepare",
            "-n",
            str(nodes),
            "-e",
            str(edges),
            "--output-dir",
            str(tmp_path / "ambiguous"),
        ],
    )
    assert result.exit_code == 1
    assert "--node-id" in result.output
    assert not (tmp_path / "ambiguous").exists()
