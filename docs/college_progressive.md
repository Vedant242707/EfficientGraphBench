# Progressive Multi-Fidelity on the college GPU

Run the following inside the **Ubuntu SSH terminal**, where the prompt shows `ise@ritise`. The program executes on the college server even though you type on your laptop. Start from your existing checkout and environment.

## 1. Update and check CUDA

```bash
cd ~/EfficientGraphBench
git pull --ff-only origin main
source .venv/bin/activate
python -m pip install -e '.[ogb,plots,setup]'
nvidia-smi
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA unavailable in this environment'; print('GPU:', torch.cuda.get_device_name(0))"
```

Continue only if the Python check succeeds. A failed `nvidia-smi` indicates a server GPU/driver problem; reinstalling this project will not fix it. Do not replace a working CUDA-enabled PyTorch installation just to update the project. If Git reports local changes, preserve them before updating; do not reset the checkout blindly.

## 2. Short test first

```bash
egbench progressive --config configs/college_progressive_smoke.yaml
```

This runs Texas on GCN, GraphSAGE, GAT, GATv2, APPNP, SGC, MLP and **SGFormer PyG**, one seed, 50% then 100% of nodes, two additional epochs per stage. Pruning is disabled for this smoke test so every successful candidate exercises continuation. SGFormer PyG is a separate maintained library implementation, not a replacement labeled as official SGFormer.

## 3. PubMed experiment

```bash
egbench progressive --config configs/college_progressive_pubmed.yaml
```

This uses the same eight models and **10 training seeds on a fixed split**. The graph grows through 25%, 50%, 100%, with 20, 60, 120 additional epochs. Survivors resume weights and optimizer state. Validation Macro-F1 determines promotion; uncertainty protection may retain additional candidates. Only full-graph finalists receive final test metrics. These budgets are a starting configuration, not tuned optimal settings.

Edit the YAML to change `dataset`, `runs`, selected models or stage budgets. Keep the last fraction at `1.0`. Each invocation writes a new experiment folder. Progressive growth is not mini-batch training: the final stage still needs memory for the full graph. Ten seeds do not create ten random train/test splits.

For a long run, keep the SSH connection open or use the server's approved session manager. If `tmux` is installed:

```bash
tmux new -s egbench-progressive
source .venv/bin/activate
egbench progressive --config configs/college_progressive_pubmed.yaml
```

Detach with **Ctrl+B**, then **D**. Reconnect using `tmux attach -t egbench-progressive`. This preserves a running process; it is not restart recovery after termination.

## 4. Official GraphGPS on Linux

```bash
egbench setup graphgps
egbench environments
egbench progressive --config configs/college_progressive_graphgps.yaml
```

This independently checks GCN and official GraphGPS on Texas, across two stages. The installer uses a separate pinned Python/PyTorch environment and does not replace controller dependencies. Actual official continuation has been tested on Windows CPU; the Linux CUDA path still needs this server check. GraphGPS recomputes structural encodings at each stage; CPU RAM and dense attention can limit larger graphs even with a large GPU. After this succeeds, add `graphgps` to the PubMed configuration if desired.

Official SGFormer continuation is implemented and tested, but its automatic Linux environment installer is not available. The Windows-oriented `configs/progressive_transformers.yaml` therefore is not the first college setup command. Do not rename `sgformer_pyg` to imply official provenance. Graphormer Progressive remains unsupported.

## 5. Status and output

In another SSH terminal:

```bash
nvidia-smi
pgrep -af 'egbench progressive|reference_worker'
find ~/EfficientGraphBench/experiments/college-progressive-pubmed -name progressive-summary.csv -print
```

The summary is saved at stage boundaries, so its absence during the first stage does not prove a stalled process. GPU utilization alone also does not establish a failure during CPU preprocessing.

Results are saved under:

```text
experiments/college-progressive-pubmed/progressive-<unique-id>/
  progressive-report.html
  progressive-summary.csv
  progressive-results.json
  checkpoints/
```

To download reports, open a **local PowerShell** window and replace `HOST` with the SSH host you already use:

```powershell
scp -r ise@HOST:~/EfficientGraphBench/experiments/college-progressive-pubmed ./college-progressive-pubmed
```

Open `progressive-report.html` in your laptop browser. For PDF, print with **Background graphics** enabled. No Full baseline or efficiency percentage is invented; a compatible Full result must be supplied explicitly for that comparison.
