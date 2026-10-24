"""Report files retain mapping access without dumping provenance in notebooks."""


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
