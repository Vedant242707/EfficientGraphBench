Python library and installed CLI reference: [PYTHON_COMMANDS.md](PYTHON_COMMANDS.md).

# Commands

## Reference models

```powershell
.venv\Scripts\egbench.exe models
.venv\Scripts\egbench.exe run --dataset cora --model graphgps --device cuda
.venv\Scripts\egbench.exe matrix --dataset cora --models gcn,graphgps,sgformer --device cuda
.venv\Scripts\egbench.exe provenance
```

The model environments launch automatically. `graphgps` and `sgformer` use authors'
code. Previous implementations: `graphgps_adapted`, `graphormer_adapted`, `sgformer_pyg`.
Primary Graphormer reports `UNSUPPORTED_TASK`. See [REFERENCE_BENCHMARKS.md](REFERENCE_BENCHMARKS.md).

## Open the project

```powershell
cd C:\path\to\EfficientGraphBench
```

## Show help

```powershell
.venv\Scripts\egbench --help
.venv\Scripts\egbench run --help
```

## List datasets

```powershell
.venv\Scripts\egbench datasets
```

## Check GPU availability

```powershell
.venv\Scripts\egbench hardware
```

## Prepare CSV files

```powershell
.venv\Scripts\egbench prepare -n people.csv -e connections.csv
```

## Preview an input table

```powershell
.venv\Scripts\egbench preview --file people.csv
.venv\Scripts\egbench preview --file graph.xlsx --sheet Nodes
.venv\Scripts\egbench preview --file graph.pdf --page 1 --table 1
```

## Prepare Excel sheets

```powershell
.venv\Scripts\egbench prepare --nodes graph.xlsx --nodes-sheet Nodes --edges graph.xlsx --edges-sheet Edges --output-dir data/my_excel
```

## Prepare PDF tables

```powershell
.venv\Scripts\egbench prepare --nodes graph.pdf --nodes-page 1 --nodes-table 1 --edges graph.pdf --edges-page 2 --edges-table 1 --output-dir data/my_pdf
```

## Check a dataset and model parameter counts

```powershell
.venv\Scripts\egbench inspect --dataset Cora
.venv\Scripts\egbench inspect --dataset custom --dataset-path data/my_dataset
```

## Train one model on GPU

```powershell
.venv\Scripts\egbench run --dataset Cora --model GCN --device cuda
.venv\Scripts\egbench run --dataset custom --dataset-path data/my_dataset --model GCN --device cuda --output-dir experiments/my_dataset
```

## Train all ten models on GPU

```powershell
.venv\Scripts\egbench matrix --dataset Cora --seeds 42,43,44 --device cuda
.venv\Scripts\egbench matrix --dataset custom --dataset-path data/my_dataset --seeds 42,43,44 --device cuda --output-dir experiments/my_dataset
```

## Use a configuration file

```powershell
.venv\Scripts\egbench run --config configs/experiments/cora.yaml --model GraphSAGE --device cuda
.venv\Scripts\egbench matrix --config configs/experiments/cora.yaml --device cuda
```

## View saved results

```powershell
.venv\Scripts\egbench compare --dataset Cora
.venv\Scripts\egbench compare --dataset Cora --output-dir experiments/cora-gpu
.venv\Scripts\egbench compare --dataset custom --output-dir experiments/my_dataset
```

## Save a comparison report

```powershell
.venv\Scripts\egbench compare --dataset custom --output-dir experiments/my_dataset --report
```

## View a searchable data sheet

```powershell
.venv\Scripts\egbench view --dataset Cora --output data/Cora-view.html
.venv\Scripts\egbench view --dataset PubMed --output data/PubMed-view.html
.venv\Scripts\egbench view --file people.csv --output data/people-view.html
.venv\Scripts\egbench view --file graph.xlsx --sheet Nodes --output data/Excel-view.html
.venv\Scripts\egbench view --dataset custom --dataset-path data/my_dataset --output data/custom-view.html
```

Open the resulting HTML file in your browser.

New matrix runs replace the active results for the same dataset in the output folder. Compare and reports show only the latest comparison. Individual runs replace the previous result for the same dataset, model, and seed. Previous logs and checkpoints remain available.

## Increase graph size until OOM

```powershell
.venv\Scripts\egbench scale --dataset PubMed --step 2000 --seeds 42,43,44 --epochs 200 --device cuda --output-dir experiments/pubmed-scaling
```

Tests all primary models at 2,000, 4,000, 6,000 nodes and so on, including the full dataset as the final size. Each size trains a fresh model on a deterministic induced subgraph with original split memberships. This is a graph-size experiment, not mini-batch training of the whole graph. Accuracy across sizes measures different subgraphs.

After an observed OOM, remaining seeds and larger sizes for that model are skipped. Other models continue. The table shows the last successful node count and accuracy, the observed OOM size, and the last status. The complete last successful result and skipped-run list are saved in `scaling-summary.json`; measured runs are in `scaling-results.json`. These sizes are observations, not guaranteed memory limits.

The ordinary `matrix` command also skips remaining seeds for a model after OOM, saving the skipped list in `matrix-skipped.json`.
