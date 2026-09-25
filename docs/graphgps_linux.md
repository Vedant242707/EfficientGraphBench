# GraphGPS on a Linux GPU server

The Linux x86-64 installer uses the unchanged pinned GraphGPS source, Python 3.10.20, PyTorch 1.13.1/CUDA 11.7 and PyG 2.2.0. It installs the bundled dependency versions using Linux wheels. Setup verifies imports on the target machine before declaring readiness. This path has not yet completed a real Linux GPU benchmark; the first server run is required to validate it. SGFormer automatic Linux setup is still unavailable.

In the existing project folder, with the controller environment active:

```bash
git pull
source .venv/bin/activate
python -m pip install uv
egbench setup graphgps
```

Check a single run first:

```bash
egbench run --dataset pubmed --model graphgps --epochs 2 --device cuda --output-dir experiments/college-graphgps-smoke
```

Then run the benchmark:

```bash
egbench matrix --dataset pubmed --models graphgps --seeds 1,2,3 --epochs 200 --device cuda --output-dir experiments/college-graphgps
```

The reference environment is separate from the controller; do not replace the controller's PyTorch. GraphGPS's selected recipe computes dense Laplacian positional encodings on CPU and uses dense attention. GPU VRAM, CPU RAM and preprocessing time can all limit the run. Two epochs do not reduce preprocessing cost. Default worker timeout is 7200 seconds per run. Keep the SSH session connected, or use the server's approved job/session manager.

If installation or execution fails, retain the exact error and worker log. More VRAM does not guarantee success, and the node-classification recipe is unchanged by this setup addition.
