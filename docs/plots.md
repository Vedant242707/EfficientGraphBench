# Plot saved benchmark results

No training or working GPU is required. Install plotting support once:

```bash
python -m pip install matplotlib
```

From the project folder on the college server:

```bash
egbench plot --summary experiments/college-ogbn-arxiv-fixed-10/summary.csv --output-dir experiments/college-ogbn-arxiv-fixed-10/plots
```

The output directory contains `accuracy_vs_latency.png`, `accuracy_vs_latency.pdf`, `gpu_memory_allocation.png`, `gpu_memory_allocation.pdf`, and `plot_data.csv`.

- Scatter: higher accuracy and lower latency are better. Latency uses a logarithmic axis. Points use mean test accuracy and mean per-run median inference latency. Vertical bars show one sample standard deviation across successful runs, not confidence intervals.
- Memory: mean of the per-run peak GPU **allocated** memory in MiB; this excludes reserved-only allocator memory. CPU-only rows with missing GPU measurements are omitted, not shown as zero. If all GPU measurements are missing, only the scatter is exported.
- Failed or skipped runs contribute no accuracy or memory measurements. Counts describe successful runs, not attempted runs. Partial failures may therefore leave fewer measurements for some models. Inspect the summary status and attempted counts before presenting a comparison.
- Use one dataset and experiment per chart. The CSV must come from the same hardware and intended comparison protocol; a summary CSV alone cannot verify those conditions. Duplicate successful model rows are rejected instead of silently merging configurations. Charts show measured configurations, not tuned model rankings.

In a notebook, after `results = Benchmark(...).run()`:

```python
from efficientgraphbench.results.plots import plot_summary
from IPython.display import Image, display

plot_summary(results.summary(), "experiments/notebook-plots")
display(Image(filename="experiments/notebook-plots/accuracy_vs_latency.png"))
display(Image(filename="experiments/notebook-plots/gpu_memory_allocation.png"))
```

From a remote Jupyter session, browse to the `plots` folder and download the PNG/PDF files to your laptop. Use `egbench plot --help` for command options.
