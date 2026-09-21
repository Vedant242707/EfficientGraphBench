# Run EfficientGraphBench on a friend's laptop

Use these steps in order. Start with CPU; GPU setup is optional.

## 1. Install Python and download the project

Install **Python 3.11**. Open https://github.com/Vedant242707/EfficientGraphBench and choose **Code → Download ZIP**, then extract it.

Open a terminal inside the extracted folder (the folder containing pyproject.toml).

## 2. Install a separate notebook environment

On **Windows**, copy these commands into the terminal:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe -m pip install jupyterlab ipykernel ipywidgets
.venv\Scripts\python.exe -m ipykernel install --user --name efficientgraphbench --display-name "EfficientGraphBench"
.venv\Scripts\python.exe -m jupyterlab
```

On **Linux/macOS**, use these instead:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m pip install jupyterlab ipykernel ipywidgets
.venv/bin/python -m ipykernel install --user --name efficientgraphbench --display-name "EfficientGraphBench"
.venv/bin/python -m jupyterlab
```

Select the **EfficientGraphBench** kernel when creating or opening the notebook. Keep the extracted project folder: this editable installation uses its source files.

## 3. Run a small test in a notebook cell

```python
from efficientgraphbench import Benchmark

result = Benchmark.run_model(
    model="gcn",
    dataset="cora",
    device="auto",
    epochs=2,
)

print("Status:", result.status)
print("Accuracy:", result.accuracy)
print("Failure reason:", result.failure_reason)
```

Cora downloads on the first run. SUCCESS means the test completed. Two epochs only checks execution; use 200 for a longer benchmark.

## 4. Compare models

```python
from IPython.display import display

results = Benchmark(
    dataset="cora",
    models=["gcn", "graphsage", "gat", "gatv2", "appnp", "sgc", "mlp"],
    device="auto",
    epochs=200,
    seeds=[1, 2, 3],
).run()

display(results.summary()[["model", "runs", "accuracy_mean", "status"]])
```

These models do not need separate research environments.

## 5. Optional: use an NVIDIA GPU

In the notebook:

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
```

If True, set `device="cuda"`. If False, CPU execution with `auto` still works. For an NVIDIA GPU, install a suitable driver and use the official PyTorch installation selector at https://pytorch.org/get-started/locally/ to choose a compatible CUDA build. Run its installation command with this environment's Python, then restart the notebook kernel. Do not copy the original laptop's CUDA settings blindly. Apple/AMD GPUs are not supported by this package's CUDA backend.

## 6. Optional: add reference SGFormer and GraphGPS

**Currently supported by the automated installer on Windows x86-64 only.** Install Git, then run in the project terminal:

```powershell
.venv\Scripts\python.exe -m pip install uv
.venv\Scripts\egbench.exe setup sgformer
.venv\Scripts\egbench.exe setup graphgps
.venv\Scripts\egbench.exe environments
```

The setup commands need Git and uv on PATH. If uv is not found, temporarily add the environment to PATH in that PowerShell window:

```powershell
$env:PATH = "$PWD\.venv\Scripts;$env:PATH"
```

Then retry setup. Separate environments are downloaded; manual activation is not needed.

Once ready, add `"sgformer"` and `"graphgps"` to the models list. Official Graphormer reports UNSUPPORTED_TASK for this node-classification workflow. A larger GPU does not change task compatibility.

## 7. Save results or inspect a failure

```python
results.save_csv("my_results.csv")

for result in results:
    if result.status != "SUCCESS":
        print(result.model, result.status, result.failure_reason)
```

Results are relative to the notebook's working directory. Your friend's hardware results should be kept separate from laptop results.

## More examples

- PYTHON_COMMANDS.md: simple notebook recipes.
- PYTHON_API_REFERENCE.md: all options and terminal commands.
- Copy custom node/edge CSV files separately; personal datasets are not in GitHub.
- Downloaded models, virtual environments, local notebooks and historical benchmark outputs are intentionally not shipped in this repository.
