import json

from efficientgraphbench.results.writer import (
    latest_results,
    read_results,
    replace_results,
    write_result,
)


def row(identifier, batch, dataset="custom", seed=42, group="same"):
    return dict(
        run_id=identifier,
        batch_id=batch,
        dataset=dataset,
        seed=seed,
        comparison_group=group,
        model="gcn",
        status="ok",
    )


def test_new_matrix_replaces_previous_even_with_same_settings(tmp_path):
    write_result(row("other", "x", dataset="cora"), tmp_path)
    write_result(row("old", "a"), tmp_path)
    write_result(row("new", "b"), tmp_path)
    write_result(row("new2", "b", seed=43), tmp_path)
    saved = read_results(tmp_path)
    assert [r["run_id"] for r in saved] == ["other", "new", "new2"]
    assert not (tmp_path / "raw/old.json").exists()
    assert [r["run_id"] for r in latest_results(saved, "custom")] == ["new", "new2"]
    assert "old" not in (tmp_path / "raw/results.csv").read_text()


def test_legacy_groups_latest_only_and_single_run_replacement(tmp_path):
    saved = [row("old", None, group="old"), row("new", None, group="new")]
    replace_results(saved, tmp_path)
    assert latest_results(saved, "custom") == [saved[1]]
    write_result(row("replacement", None, group="new"), tmp_path)
    assert len(read_results(tmp_path)) == 1


def test_failed_new_batch_does_not_show_old_success(tmp_path):
    write_result(row("old", "a"), tmp_path)
    failed = row("failure", "b")
    failed["status"] = "failed"
    write_result(failed, tmp_path)
    assert latest_results(read_results(tmp_path), "custom") == [failed]
    assert json.loads((tmp_path / "raw/results.jsonl").read_text())["status"] == "failed"
