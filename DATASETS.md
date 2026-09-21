# Use your dataset

## Choose a common benchmark

See the available names:

```powershell
.venv\Scripts\egbench datasets
```

Built-in adapters: **Cora, CiteSeer, PubMed, Amazon Computers, Amazon Photo,
Coauthor CS, Coauthor Physics, WikiCS, Flickr, and ogbn-arxiv**.

For example, compare the four models on CiteSeer:

```powershell
.venv\Scripts\egbench matrix --dataset CiteSeer
```

For names containing spaces, use their short identifier from the list:

```powershell
.venv\Scripts\egbench run --dataset amazon-photo --model GCN
```

Data downloads on first use. **An available loader does not mean every model will
fit in your RAM/VRAM.** All current models train on the entire graph at once.
Flickr and ogbn-arxiv need more memory. Large-scale sampling is a later feature.

## Use your own CSV files

Already have different column names? Use the `prepare` example below to convert them.

Make a folder containing **nodes.csv** and **edges.csv**. You can copy the small
example in `examples/custom_graph` and replace its contents with your data.

**nodes.csv** describes each item in your graph:

```csv
node_id,label,split,feature_1,feature_2
alice,buyer,train,0.8,0.2
bob,seller,train,0.1,0.9
carol,buyer,val,0.7,0.3
dave,seller,test,0.2,0.8
```

- `node_id`: a unique name or ID for each node.
- `label`: the category you want the model to predict, such as buyer or seller.
- `split`: `train` teaches the model; `val` selects the best checkpoint;
  `test` measures its final accuracy. Every split must have at least one node.
- `feature_1`, `feature_2`, etc.: numeric properties the model can learn from.
  Do not include the answer (label) as a feature.

**edges.csv** describes connections:

```csv
source,target
alice,bob
bob,alice
carol,alice
dave,bob
```

Every endpoint must exist in `nodes.csv`. Each row is a directed connection;
for an undirected connection, include both directions. Self-loops and duplicate
edges are preserved as supplied. String labels are converted to numeric classes
and the mapping is saved in the results. Every class needs a training example.
Optional unlabeled nodes use an empty label and `unused` split.

**First check your files** (no training):

```powershell
.venv\Scripts\egbench inspect --dataset custom --dataset-path examples/custom_graph
```

**Then compare all four models**:

```powershell
.venv\Scripts\egbench matrix --dataset custom --dataset-path examples/custom_graph --output-dir experiments/my-graph
```

Replace `examples/custom_graph` with your folder path. Quote paths containing spaces.
The example is only a tiny format demonstration; its accuracy has no research meaning.

View the results:

```powershell
.venv\Scripts\egbench compare --dataset custom --output-dir experiments/my-graph
```

Invalid files produce a specific error before training. Graph contents and splits
are fingerprinted, so different uploads are shown in separate comparison groups.
This is local file import; there is no hosted upload website yet.

## Convert existing CSV columns

Suppose your nodes use `person_id`, `category`, `age`, and `score`, and your edges
use `from_id` and `to_id`:

```powershell
.venv\Scripts\egbench prepare --nodes people.csv --edges connections.csv --node-id person_id --label category --source from_id --target to_id --features age,score --output-dir data/my_dataset
```

This creates the standard files without changing the originals. The output folder
must be new or empty. Without a `split` column, a fixed per-class 60/20/20 split is
generated (at least three labeled nodes per category required). Use `--split-column`
to name an existing split column. Use `--split-seed` to change generated splits.

Omitting `--features` selects numeric columns automatically, excluding ID, label,
and split. Selected and ignored columns are printed and saved in `preparation.json`.
Review them: the program cannot identify target leakage hidden in another column.

Check your data and model sizes:

```powershell
.venv\Scripts\egbench inspect --dataset custom --dataset-path data/my_dataset
```

This reports nodes, edges, features, classes, split sizes, input memory, and parameter
counts for all four models. Pass `--config your_config.yaml` to inspect different
model settings. These are parameter counts, not automatic hyperparameter tuning.

Run on GPU after `hardware` confirms CUDA is available:

```powershell
.venv\Scripts\egbench hardware
.venv\Scripts\egbench run --dataset custom --dataset-path data/my_dataset --model GCN --device cuda --output-dir experiments/my_dataset
```

Model input/output dimensions adapt to the features/classes. An arbitrary spreadsheet
still needs meaningful connections and a target category; those cannot be guessed.

## Flickr download recovery

Flickr now shows downloaded bytes, speed, and elapsed time. Reads time out after
60 seconds without data. Partial files use `.part`; rerun the same command to retry.
Downloads resume when the server supports HTTP ranges; otherwise they restart.
NPY size, NPZ integrity, and JSON validity are checked before accepting files.
GPU acceleration helps computation, not downloads, and does not guarantee higher accuracy.

## What kind of problem is supported?

Currently: **predicting one category per node in one graph**, with numeric node
features. The graph may be directed or undirected. All models can see the whole
graph and its features, but only training labels contribute to optimization.

Predicting missing links, classifying whole molecules, multilabel prediction, and
heterogeneous graphs need different training/evaluation pipelines. Arbitrary tables
cannot be benchmarked until nodes, connections, features, and target labels are defined.

## Technical reference (optional)

| Identifier | Split used | Feature/edge treatment |
|---|---|---|
| `cora`, `citeseer`, `pubmed` | Planetoid public | PyG feature normalization; provided edges |
| `amazon-computers`, `amazon-photo` | Per-class 60/20/20, split seed 0 | PyG feature normalization |
| `coauthor-cs`, `coauthor-physics` | Per-class 60/20/20, split seed 0 | PyG feature normalization |
| `wikics` | Official split 0; select 0–19 with `--split-index` | Provided features; undirected edges |
| `flickr` | Provided masks | Provided features/edges; transductive full-graph protocol |
| `ogbn-arxiv` | OGB official time split | Provided features; converted to undirected |
| `custom` | User-provided, required | Features and directed edge rows preserved |

Generated splits are explicitly our benchmark protocol, not an official paper split.
They use a separate `--split-seed` (default 0) that stays fixed across model seeds.
Rounding is per class, with at least one node in each split. Validation checks masks
do not overlap, feature values are finite, and labels/edge endpoints are valid.
Custom features become float32; NPZ labels/edges become int64. No automatic imputation.

WikiCS uses its selected validation mask for checkpoint selection; its separate
stopping mask is not used. Flickr runs here are transductive and should not be
compared directly with an inductive benchmark. These are explicit local protocols,
not claims to reproduce each dataset's published leaderboard.

ogbn-arxiv needs the optional OGB package:

```powershell
uv pip install --python .venv/Scripts/python.exe -e '.[ogb]'
.venv\Scripts\egbench inspect --dataset ogbn-arxiv
```

Numeric `.npz` archives are also accepted through `--dataset-path`. Required arrays:
`x` (N × F numbers), `edge_index` (2 × E integers), `y` (N integer class IDs from zero,
or -1 for unused unlabeled nodes), and boolean `train_mask`, `val_mask`, `test_mask`
(each length N). Object arrays/pickled files are not accepted.

Sources: [PyG dataset catalog](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html)
and [OGB node prediction documentation](https://ogb.stanford.edu/docs/nodeprop/).

## Verification status

- 47 automated tests pass, including small-fixture coverage of every loader family,
  CSV/NPZ validation, fixed generated splits, and result separation by graph fingerprint.
- Cora: full initial four-model benchmark completed; new data validation also passes.
- CiteSeer: real download and two-epoch execution checks completed for all four models.
- CSV example: two-epoch execution checks completed for all four models.
- CUDA verified on the RTX 4050: 12 full Cora runs (three seeds, 200 epochs per model)
  and four two-epoch runs using the prepared CSV example completed successfully.
  GPU reports are in `experiments/cora-gpu/reports/cora.md` and
  `experiments/prepared-demo-gpu/reports/custom.md`.
- Amazon Photo: real download stalled and was stopped; its incomplete file was set
  aside for a clean retry. Adapter behavior is tested using a small fixture.
- Remaining built-ins, including OGB: adapter tests pass, but real downloads/full
  benchmarks have not been verified on this machine. OGB remains an optional dependency.

Short execution checks establish compatibility, not meaningful accuracy rankings.


## Excel, TSV, and PDF input

Use `egbench preview --file <path>` to check a table before `egbench prepare`.
Prepare accepts CSV, tab-separated TSV, XLSX, and PDFs containing readable ruled tables.
For workbooks with multiple sheets, select `--nodes-sheet` and `--edges-sheet`.
For PDFs with multiple tables, select `--nodes-page`, `--nodes-table`, `--edges-page`, and `--edges-table` (numbers start at 1).
Use `--nodes-header-row` or `--edges-header-row` for a header below the first row.
The nodes and edges may come from the same workbook or PDF, or different supported files.

The tables must still provide unique node IDs, numeric features, class labels, and source/target connections. Use the existing column-mapping options. Excel IDs with leading zeros must be stored as text. Formula cells must be replaced with values. Scanned PDFs, unruled PDF tables, merged headers, and automatic joining of tables across pages are not supported; export these as CSV or XLSX first. Preview PDF extraction carefully before preparing. Imports do not infer edges or turn arbitrary documents into graph datasets.

Prepared output remains a validated CSV folder, usable by `inspect`, `run`, and `matrix` on CPU or GPU. Input paths and selectors are saved in `preparation.json`.
