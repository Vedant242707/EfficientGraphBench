"""Offline, searchable dataset tables with explicit column meanings."""

import json
from pathlib import Path

import pandas as pd

from efficientgraphbench.datasets.registry import load_dataset
from efficientgraphbench.datasets.tables import read_table


def dataset_view(dataset, dataset_path=None, data_dir="data", columns=None):
    if dataset.lower() == "custom" and dataset_path and Path(dataset_path).is_dir():
        folder = Path(dataset_path)
        metadata = folder / "preparation.json"
        info = json.loads(metadata.read_text(encoding="utf-8")) if metadata.exists() else {}
        original = folder / "original_nodes.csv"
        frame = read_table(original if original.exists() else folder / "nodes.csv")
        roles = {}
        if original.exists():
            roles = {name: "Not used for prediction" for name in frame.columns}
            for name in info.get("features", []):
                roles[name] = "Prediction input"
            for role, name in info.get("columns", {}).items():
                if role in {"node_id", "label", "split"} and name in roles:
                    roles[name] = {
                        "node_id": "Identifier",
                        "label": "Answer to predict",
                        "split": "Training / validation / test assignment",
                    }.get(role, role)
            note = "Original imported table. Prediction inputs are marked below each column name."
        else:
            mapping = {f"feature_{i + 1}": name for i, name in enumerate(info.get("features", []))}
            frame = frame.rename(columns=mapping)
            note = (
                "Prepared table. Original values and unused columns were not saved by older "
                "imports. Prepare again to preserve the complete original table."
            )
        if columns:
            missing = set(columns) - set(frame.columns)
            if missing:
                raise ValueError(f"Unknown columns: {', '.join(sorted(missing))}")
            frame = frame[columns]
        return frame, roles, note
    bundle = load_dataset(dataset, data_dir, dataset_path=dataset_path)
    data = bundle.data
    splits = ["unused"] * data.num_nodes
    for name in ("train", "val", "test"):
        for index in getattr(data, f"{name}_mask").nonzero().flatten().tolist():
            splits[index] = name
    names = [f"feature_{i + 1}" for i in range(data.num_node_features)]
    selected = columns or names[:12]
    if any(name not in names for name in selected):
        raise ValueError("Select feature columns such as feature_1,feature_2")
    frame = pd.DataFrame(
        {"node_id": range(data.num_nodes), "label": data.y.tolist(), "split": splits}
    )
    for name in selected:
        frame[name] = data.x[:, names.index(name)].tolist()
    roles = {name: "Numeric prediction input; original name unavailable" for name in selected}
    roles.update(node_id="Benchmark row ID", label="Class ID to predict", split="Evaluation split")
    note = (
        f"Model input values ({bundle.preprocessing}). Showing {len(selected)} of "
        f"{len(names)} features and all nodes. Use --columns to choose other features. "
        "The installed benchmark provides no original feature-name dictionary."
    )
    if dataset.lower() in {"cora", "citeseer", "pubmed"}:
        note += " Rows represent papers; features encode word information, not age or marks."
    return frame, roles, note


def write_view(frame, roles, note, output, title):
    output = Path(output)
    if output.exists():
        raise ValueError(f"Choose a new output filename; already exists: {output}")
    payload = json.dumps(
        {
            "columns": frame.columns.tolist(),
            "rows": frame.values.tolist(),
            "roles": roles,
            "note": note,
            "title": title,
        },
        ensure_ascii=True,
    )
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    html = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Dataset viewer</title>
<style>body{font:16px system-ui;margin:30px;color:#192c40;background:#f5f7fa}
input,button{padding:10px;margin:5px;border:1px solid #b7c4d2;border-radius:5px}
.scroll{overflow:auto;max-height:70vh;background:white}table{border-collapse:collapse;width:100%}
th,td{padding:10px;text-align:left;border:1px solid #dde3ea;white-space:nowrap}
th{position:sticky;top:0;background:#e8eff8}small{display:block;font-weight:normal;color:#52647a}
#note{max-width:1000px;line-height:1.6}button:disabled{opacity:.4}</style>
<h1 id="title"></h1><p id="note"></p>
<label>Search rows <input id="search" placeholder="Type a value or label"></label>
<button id="prev">Previous</button><button id="next">Next</button><span id="count"></span>
<div class="scroll"><table><thead id="head"></thead><tbody id="body"></tbody></table></div>
<script type="application/json" id="data">PAYLOAD</script><script>
const d=JSON.parse(document.getElementById('data').textContent);
document.getElementById('title').textContent=d.title;
document.getElementById('note').textContent=d.note;
const head=document.createElement('tr');
for(const name of d.columns){const th=document.createElement('th');th.textContent=name;
const sub=document.createElement('small');sub.textContent=d.roles[name]||'';
th.append(sub);head.append(th);}document.getElementById('head').append(head);
let rows=d.rows,page=0;const size=50;
function draw(){const body=document.getElementById('body');body.replaceChildren();
for(const row of rows.slice(page*size,(page+1)*size)){const tr=document.createElement('tr');
for(const value of row){const td=document.createElement('td');td.textContent=String(value);
tr.append(td);}
body.append(tr);}document.getElementById('count').textContent=
`${rows.length} matching rows / ${d.rows.length} total · Page ${page+1} of ` +
`${Math.max(1,Math.ceil(rows.length/size))}`;
document.getElementById('prev').disabled=page===0;
document.getElementById('next').disabled=(page+1)*size>=rows.length;}
document.getElementById('search').oninput=e=>{const q=e.target.value.toLowerCase();
rows=d.rows.filter(r=>r.some(v=>String(v).toLowerCase().includes(q)));page=0;draw();};
document.getElementById('prev').onclick=()=>{page--;draw();};
document.getElementById('next').onclick=()=>{page++;draw();};draw();</script></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html.replace("PAYLOAD", payload), encoding="utf-8")
    return output.resolve()
