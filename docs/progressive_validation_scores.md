# What Progressive selects on

In the college dataset notebook the Progressive configuration explicitly sets
`selection_metric="val_f1_macro"`. The API default when no metric is supplied
is `val_accuracy`, so use the saved run configuration to interpret an old report.

1. **Validation population.** `datasets/registry.py::_load_ogb` sets the mask from
   OGB's `valid` indices (the time split for ogbn-arxiv). `ProgressionBuilder.build`
   constructs each stage in its own node order and slices all three original
   masks using the same IDs. Earlier stages use the validation nodes present in
   that stage; a 100% stage includes the complete original validation set.
2. **Predictions.** At stage end, `progressive/training.py::_train_stage` evaluates
   the last-epoch model in eval/inference mode. The whole-stage selection score
   uses `data.val_mask`, distinct from the old/new masks used for adaptation.
   The reference worker calls the same stage trainer. The classification helper
   masks labels and logits and takes the argmax over classes.
3. **Metrics.** Accuracy is correct predictions divided by validation nodes.
   Macro-F1 averages per-class F1 equally over classes present in the targets
   **or** predictions; a prediction-only class receives zero F1. Absent-in-both
   classes are excluded. Micro-F1 equals accuracy for this single-label task.
   New successful stage records retain all three values from this same forward
   pass in `validation_metrics`; `validation_metric` remains the selected scalar.
4. **Selection.** `select_candidates` requires successful, finite scores for the
   full requested seed set. It ranks the arithmetic mean of the seed scores
   descending, breaking exact ties by model identifier. The configured retain
   count, margin and optional sample-SD overlap rule govern advancement. At the
   final stage all eligible completed candidates advance. Test scores never feed
   selection. With one 100% stage there is no between-stage pruning.
5. **Report.** Stage tables summarize successful seeds for that model/stage
   (sample SD; zero for one seed). The finalist table uses the last-stage scores
   and excludes incomplete candidates. The mean is not computed across epochs
   or from pooled predictions. Validation and test use different masks; a
   comparison with test accuracy also changes the metric when selecting on F1.

The selected ID and stage settings are stored at `metadata.config`; stage graph
and split hashes and selected node IDs are saved per record. New runs also store
`metadata.selection_protocol` with the metric definition, population, checkpoint
scope, aggregation and direction. Older runs do not gain historical validation
accuracy by relabelling a table. Do not use a test metric in its place.

To audit an existing college 100% ogbn-arxiv run without training, run
`%run /home/ise/EfficientGraphBench/examples/audit_progressive_validation.py`
in Jupyter. By default it checks GCN, GraphSAGE and SGFormer (PyG), across all
successful full-stage seeds in the latest matching notebook run. It prints the
selected results file and official split file; supply both paths to select
another run. It checks checkpoint and graph hashes, maps the stage mask back to
original node IDs, compares those IDs against OGB's official valid.csv.gz,
and computes both metrics from one forward pass. Replayed predictions are not
archived original predictions: inspect the delta from the saved selection score.
A difference greater than 1e-6 fails verification; report it rather than treating
the measurements as identical.
