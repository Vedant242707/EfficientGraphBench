"""Report files retain mapping access without dumping provenance in notebooks."""

import base64
from html import escape


class ReportOutput(dict):
    """Backward-compatible report mapping with a notebook-aware display hook."""

    def __init__(self, *args, notebook_displayed=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.notebook_displayed = notebook_displayed

    def __repr__(self):
        return f"EfficientGraphBench report: {self['report']}"

    def _repr_pretty_(self, printer, cycle):
        printer.text(repr(self))

    def _ipython_display_(self):
        # show_all explicitly displays even when its return value is assigned.
        # Jupyter also displays a cell's final expression: do not repeat it.
        if not self.notebook_displayed:
            from IPython.display import HTML, display

            display(HTML(self["report"].read_text(encoding="utf-8")))


class InteractiveReportOutput(ReportOutput):
    """Run generated scripts in their own document, not Jupyter's HTML renderer."""

    def notebook_html(self):
        document = self["report"].read_text(encoding="utf-8")
        links = []
        for key, mime, label in (
            ("report", "text/html", "Download interactive report (HTML)"),
            ("data", "application/json", "Download execution data (JSON)"),
        ):
            path = self[key]
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            links.append(
                f'<a download="{escape(path.name, quote=True)}" '
                f'href="data:{mime};base64,{encoded}">{label}</a>'
            )
        return (
            '<div><p>The interactive report runs below. '
            'For a full-size view, download the HTML and open it in your browser. '
            'Use its PDF button and choose Save as PDF in the print dialog.</p><p>'
            + " &nbsp; | &nbsp; ".join(links)
            + '</p><p>If the embedded page is blocked by notebook trust settings, '
            'use the downloaded HTML.</p>'
            '<iframe title="EfficientGraphBench execution analysis" '
            'style="width:100%;height:950px;border:1px solid #536b90" '
            'sandbox="allow-scripts allow-same-origin allow-downloads allow-modals" '
            f'srcdoc="{escape(document, quote=True)}"></iframe></div>'
        )

    def _ipython_display_(self):
        if not self.notebook_displayed:
            from IPython.display import HTML, display

            display(HTML(self.notebook_html()))
