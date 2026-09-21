import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from efficientgraphbench.cli.main import app
from efficientgraphbench.datasets.prepare import prepare_csv
from efficientgraphbench.datasets.viewer import dataset_view, write_view


def test_original_columns_survive_preparation(tmp_path):
    example = Path(__file__).resolve().parents[1] / "examples/custom_graph"
    nodes = pd.read_csv(example / "nodes.csv", dtype=str)
    nodes = nodes.rename(columns={"feature_1": "marks"})
    nodes["name"] = [f"Person {i}" for i in range(len(nodes))]
    source = tmp_path / "people.csv"
    nodes.to_csv(source, index=False)
    target = tmp_path / "prepared"
    prepare_csv(source, example / "edges.csv", target)
    source.unlink()
    frame, roles, note = dataset_view("custom", target)
    assert frame.columns.tolist() == nodes.columns.tolist()
    assert frame["marks"].tolist() == nodes["marks"].tolist()
    assert roles["marks"] == "Prediction input"
    assert roles["name"] == "Not used for prediction"
    assert roles["label"] == "Answer to predict"
    assert "Original" in note


def test_view_safe_payload_and_overwrite_protection(tmp_path):
    frame = pd.DataFrame({"name": ["</script><script>alert(1)</script>", "001"]})
    output = tmp_path / "view.html"
    write_view(frame, {}, "original", output, "test")
    html = output.read_text(encoding="utf-8")
    assert "</script><script>alert(1)" not in html
    payload = html.split('id="data">')[1].split("</script>")[0]
    assert json.loads(payload)["rows"] == frame.values.tolist()
    with pytest.raises(ValueError, match="already exists"):
        write_view(frame, {}, "", output, "test")


def test_view_cli_original_table(tmp_path):
    source = tmp_path / "people.csv"
    source.write_text("name,age,marks\nAlice,20,85\nBob,21,90\n")
    output = tmp_path / "people.html"
    result = CliRunner().invoke(app, ["view", "--file", str(source), "--output", str(output)])
    assert result.exit_code == 0, result.output
    assert '"name", "age", "marks"' in output.read_text()
