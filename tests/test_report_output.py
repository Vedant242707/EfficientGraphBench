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


def test_interactive_report_is_isolated_and_downloads_are_self_contained(tmp_path, monkeypatch):
    import base64
    import html

    from IPython import display

    from efficientgraphbench.results.artifact import InteractiveReportOutput

    report_path = tmp_path / "execution.html"
    document = '<html><script>window.ready=true</script><p title="x">Model</p></html>'
    report_path.write_text(document, encoding="utf8")
    data_path = tmp_path / "execution.json"
    data_path.write_text('{"sessions": []}', encoding="utf8")
    result = InteractiveReportOutput(report=report_path, data=data_path)
    calls = []
    monkeypatch.setattr(display, "display", calls.append)
    DisplayFormatter().format(result)
    markup = calls[0].data
    assert '<iframe title="EfficientGraphBench execution analysis"' in markup
    assert 'srcdoc="' + html.escape(document, quote=True) + '"' in markup
    assert 'allow-downloads allow-modals' in markup
    encoded_data = base64.b64encode(data_path.read_bytes()).decode()
    assert 'data:application/json;base64,' + encoded_data in markup
    assert 'data:text/html;base64,' + base64.b64encode(report_path.read_bytes()).decode() in markup
    assert result["report"] == report_path
