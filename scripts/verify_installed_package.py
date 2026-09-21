# ruff: noqa: E501
"""Installed-wheel integration smoke; invokes an isolated interpreter outside the checkout."""

import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
python = ROOT / ".package-test/Scripts/python.exe"
source = r"""
import json
from pathlib import Path
import efficientgraphbench as egb
from efficientgraphbench.datasets import load_dataset
from efficientgraphbench.environments import inspect_environment
from efficientgraphbench.paths import PACKAGE
assert '.package-test' in str(PACKAGE), PACKAGE
Path('nodes.csv').write_text('node_id,feature_1,feature_2,label,split\n0,1,0,a,train\n1,0,1,b,train\n2,1,0,a,val\n3,0,1,b,val\n4,1,0,a,test\n5,0,1,b,test\n')
Path('edges.csv').write_text('source,target\n0,2\n2,4\n1,3\n3,5\n2,0\n4,2\n3,1\n5,3\n')
data=load_dataset('custom',nodes='nodes.csv',edges='edges.csv')
results=egb.Benchmark(dataset=data,models=['gcn','appnp','sgformer','graphgps','graphormer'],seeds=[42],epochs=2,device='cpu',latency_warmup=1,latency_repeats=2).run()
statuses={r.model:r.status for r in results}
assert statuses == {'gcn':'SUCCESS','appnp':'SUCCESS','sgformer':'SUCCESS','graphgps':'SUCCESS','graphormer':'UNSUPPORTED_TASK'}, [(r.model,r.status,r.failure_reason) for r in results]
results.save_json('results.json')
results.save_csv('results.csv')
results.save_jsonl('results.jsonl')
results.save_markdown('report.md')
print(json.dumps({'package':str(PACKAGE),'version':egb.__version__,'statuses':statuses}))
"""
with tempfile.TemporaryDirectory(prefix="egbench-wheel-") as directory:
    env = os.environ.copy()
    env["EGBENCH_HOME"] = str(ROOT)
    result = subprocess.run(
        [str(python), "-c", source],
        cwd=directory,
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    print(result.stdout)
    if result.returncode:
        print(result.stderr)
        raise SystemExit(result.returncode)
    out = ROOT / "audits/library-release"
    out.mkdir(parents=True, exist_ok=True)
    (out / "wheel-smoke.json").write_text(result.stdout, encoding="utf-8")
