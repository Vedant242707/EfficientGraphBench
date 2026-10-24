import pytest

from efficientgraphbench.results.artifact import ReportOutput

DisplayFormatter = pytest.importorskip("IPython.core.formatters").DisplayFormatter


def test_notebook_final_expression_does_not_dump_or_repeat_report(tmp_path, monkeypatch):
    from IPython import display

    calls = []
    monkeypatch.setattr(display, "display", lambda value: calls.append(value))
    report = ReportOutput(
        report=tmp_path / "report.html",
        pareto={"source_snapshot": {"internal.py": "private-display-detail"}},
        plots=[],
        notices=[],
        notebook_displayed=True,
    )
    assert DisplayFormatter().format(report) == ({}, {})
    assert calls == []
    assert "source_snapshot" not in repr(report)
    assert report["pareto"]["source_snapshot"]["internal.py"] == "private-display-detail"


def test_explicit_display_of_undisplayed_artifact_renders_saved_report(tmp_path, monkeypatch):
    from IPython import display

    path = tmp_path / "report.html"
    path.write_text("<h1>Full report</h1><table>All models</table>", encoding="utf-8")
    calls = []
    monkeypatch.setattr(display, "display", lambda value: calls.append(value))
    report = ReportOutput(report=path, pareto={}, plots=[], notices=[])
    assert DisplayFormatter().format(report) == ({}, {})
    assert len(calls) == 1
    assert calls[0].data == path.read_text(encoding="utf-8")
