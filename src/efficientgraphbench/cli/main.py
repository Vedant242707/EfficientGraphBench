import copy
import json
import logging
import subprocess
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from efficientgraphbench.benchmark.runner import run_benchmark
from efficientgraphbench.config import load_config
from efficientgraphbench.datasets.catalog import DATASETS, canonical_dataset
from efficientgraphbench.datasets.prepare import prepare_csv
from efficientgraphbench.datasets.registry import load_dataset
from efficientgraphbench.datasets.tables import read_table
from efficientgraphbench.datasets.viewer import dataset_view, write_view
from efficientgraphbench.models.catalog import PRIMARY_MODELS, availability, model_spec
from efficientgraphbench.models.registry import MODEL_METADATA, build_model
from efficientgraphbench.names import MODEL_NAMES, dataset_name, model_name
from efficientgraphbench.profiling.hardware import hardware_info
from efficientgraphbench.results.report import write_report
from efficientgraphbench.results.status import failure_reason, is_success, status_of
from efficientgraphbench.results.writer import latest_results, read_results

app = typer.Typer(no_args_is_help=True, help="Hardware-aware graph model benchmarks.")

console = Console()


@app.command()
def scale(
    dataset: str = "cora",
    sizes: str = "500,1000,2708",
    models: Annotated[str, typer.Option("--models", "--model")] = ",".join(PRIMARY_MODELS),
    seeds: str = "42,43,44",
    epochs: int = 20,
    device: str = "auto",
    output_dir: str = "experiments/scaling",
    dataset_path: Annotated[Path | None, typer.Option(exists=True)] = None,
    step: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Increase node count by this amount up to the full dataset; overrides --sizes.",
        ),
    ] = None,
):
    """Measure deterministic induced subgraphs; retains original split memberships."""
    from efficientgraphbench.benchmark.scaling import run_scaling

    cfg = load_config()
    cfg.dataset, cfg.device, cfg.output_dir = dataset, device, output_dir
    cfg.dataset_path = str(dataset_path) if dataset_path else None
    cfg.training.epochs = epochs
    # Keep per-size training messages in run logs, not the terminal.
    package_logger = logging.getLogger("efficientgraphbench")
    previous_propagate = package_logger.propagate
    package_logger.propagate = False
    try:
        result = run_scaling(
            cfg,
            list(dict.fromkeys(int(n) for n in sizes.split(","))),
            [m.strip().lower() for m in models.split(",")],
            list(dict.fromkeys(int(n) for n in seeds.split(","))),
            step=step,
        )
    except (ValueError, OSError) as exc:
        console.print(str(exc), markup=False)
        raise typer.Exit(1) from exc
    finally:
        package_logger.propagate = previous_propagate
    summary = json.loads(result.with_name("scaling-summary.json").read_text(encoding="utf-8"))
    table = Table(title="Final result per model (nodes)")
    for title in ("Model", "Last successful nodes", "Accuracy", "OOM nodes", "Last status"):
        table.add_column(title)
    for model, outcome in summary["models"].items():
        success, oom = outcome["last_success"], outcome["first_oom"]
        accuracy = success["result"].get("accuracy") if success else None
        table.add_row(
            model_name(model),
            str(success["num_nodes"]) if success else "None",
            f"{accuracy:.4f}" if accuracy is not None else "—",
            str(oom["num_nodes"]) if oom else "Not observed",
            outcome.get("last_status", "Not measured"),
        )
    console.print(table)
    console.print(f"Saved results: {result.parent}")


@app.command()
def provenance(output: str = "audits/review20/model_provenance.json"):
    """Write model sources, versions, modifications and measured parameter counts."""
    from efficientgraphbench.models.provenance import generate_provenance

    console.print(str(generate_provenance(output)), markup=False)


@app.command()
def recommend(
    dataset: str = "cora",
    output_dir: str = "experiments",
    minimum_accuracy: float = 0.8,
    maximum_latency_ms: float = 10.0,
    maximum_gpu_mib: float = 6141.0,
    group: str | None = None,
):
    """Filter primary measured results; GPU budget uses maximum reserved MiB across seeds."""
    import math

    from efficientgraphbench.results.recommend import recommend as filter_results

    if not (
        0 <= minimum_accuracy <= 1
        and math.isfinite(maximum_latency_ms)
        and maximum_latency_ms > 0
        and math.isfinite(maximum_gpu_mib)
        and maximum_gpu_mib > 0
    ):
        raise typer.BadParameter(
            "Accuracy must be in [0,1]; latency and memory must be finite positive"
        )
    records = [
        r for r in read_results(output_dir) if r.get("dataset") == canonical_dataset(dataset)
    ]
    try:
        matches = filter_results(
            records, minimum_accuracy, maximum_latency_ms, maximum_gpu_mib, group
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if not matches:
        console.print("No complete primary measurements meet these constraints.")
        return
    table = Table(title="Measured primary candidates")
    for heading in ("Model", "Accuracy", "Latency ms", "Max reserved MiB", "Seeds"):
        table.add_column(heading)
    for row in matches:
        table.add_row(
            model_name(row["model"]),
            f"{row['accuracy']:.4f}",
            f"{row['latency_ms']:.3f}",
            f"{row['gpu_reserved_mib']:.1f}",
            str(row["seeds"]),
        )
    console.print(table)
    console.print("Measured implementation costs; this is not an architecture-only ranking.")


@app.command()
def models(name: Annotated[str | None, typer.Argument()] = None):
    """List model sources, environment readiness, and experimental alternatives."""
    if name is not None:
        from efficientgraphbench.models.metadata import get_model

        try:
            console.print_json(data=get_model(name))
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        return
    table = Table(title="Model implementations")
    for heading in ("ID", "Model", "Status", "Source", "Environment"):
        table.add_column(heading)
    for name in MODEL_NAMES:
        spec = model_spec(name)
        status, _ = availability(name)
        table.add_row(name, model_name(name), status, spec["source_type"], spec["environment"])
    Console(width=max(console.width, 125)).print(table)
    console.print("Graphormer: no validated reference adapter for this node-classification task.")


@app.command()
def hardware():
    """Show whether this environment can use your GPU."""

    info = hardware_info()

    console.print(f"PyTorch: {info['torch_version']}")

    console.print(f"Default device: {info['device_name']}")

    console.print(f"CUDA available: {info['cuda_available']}")

    if info["cuda_available"]:
        console.print(
            f"GPU memory: {info['gpu_free_memory_mb']:.0f} MiB free / "
            f"{info['gpu_memory_mb']:.0f} MiB total"
        )


@app.command()
def preview(
    file: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    sheet: str | None = None,
    page: int | None = None,
    table: int | None = None,
    header_row: int = 1,
):
    """Show columns and the first five rows before mapping a table into a graph."""

    try:
        frame = read_table(file, sheet=sheet, page=page, table=table, header_row=header_row)

        display = Table(title=f"{file.name}: {len(frame)} rows")

        for column in frame.columns:
            display.add_column(column)

        for row in frame.head(5).itertuples(index=False, name=None):
            display.add_row(*row)

        console.print(display)

    except (ValueError, OSError) as exc:
        console.print(str(exc), markup=False)

        raise typer.Exit(1) from exc


@app.command()
def view(
    dataset: str = "Cora",
    dataset_path: Annotated[Path | None, typer.Option(exists=True)] = None,
    file: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    sheet: str | None = None,
    page: int | None = None,
    table: int | None = None,
    header_row: int = 1,
    data_dir: str = "data",
    columns: str | None = None,
    output: str = "data/dataset-view.html",
):
    """Create a searchable HTML data sheet; open the saved file in your browser."""

    try:
        selected = [name.strip() for name in columns.split(",")] if columns else None

        if file:
            frame = read_table(file, sheet=sheet, page=page, table=table, header_row=header_row)

            if selected:
                if any(name not in frame for name in selected):
                    raise ValueError("Selected columns must exist in the input table")

                frame = frame[selected]

            roles, note = (
                {},
                "Original imported table. No prediction columns have been selected yet.",
            )

            title = file.name

        else:
            if sheet is not None or page is not None or table is not None or header_row != 1:
                raise ValueError("Sheet, page, table, and header-row options require --file")

            frame, roles, note = dataset_view(dataset, dataset_path, data_dir, selected)

            title = dataset_name(canonical_dataset(dataset))

        path = write_view(frame, roles, note, output, title)

        console.print(f"Open in your browser: {path}", markup=False)

    except (ValueError, OSError) as exc:
        console.print(str(exc), markup=False)

        raise typer.Exit(1) from exc


@app.command()
def prepare(
    nodes: Annotated[Path, typer.Option("--nodes", "-n", exists=True, dir_okay=False)],
    edges: Annotated[Path, typer.Option("--edges", "-e", exists=True, dir_okay=False)],
    output_dir: str = "data/my_dataset",
    node_id: str | None = None,
    label: str | None = None,
    source: str | None = None,
    target: str | None = None,
    features: str | None = None,
    split_column: str = "split",
    split_seed: int = 0,
    nodes_sheet: str | None = None,
    edges_sheet: str | None = None,
    nodes_page: int | None = None,
    edges_page: int | None = None,
    nodes_table: int | None = None,
    edges_table: int | None = None,
    nodes_header_row: int = 1,
    edges_header_row: int = 1,
):
    """Map CSV, TSV, XLSX, or PDF tables into a validated graph dataset."""

    try:
        if not 0 <= split_seed < 2**32:
            raise ValueError("split_seed must be between 0 and 2**32 - 1")

        summary = prepare_csv(
            nodes,
            edges,
            output_dir,
            node_id=node_id,
            label=label,
            source=source,
            target=target,
            features=[s.strip() for s in features.split(",")] if features else None,
            split_column=split_column,
            split_seed=split_seed,
            nodes_options=dict(
                sheet=nodes_sheet, page=nodes_page, table=nodes_table, header_row=nodes_header_row
            ),
            edges_options=dict(
                sheet=edges_sheet, page=edges_page, table=edges_table, header_row=edges_header_row
            ),
        )

        console.print(f"Column mapping: {summary['columns']}", markup=False)

        console.print(f"Ready: {output_dir}")

        console.print(
            f"{summary['nodes']} nodes, {summary['edges']} edges, {summary['classes']} classes"
        )

        console.print(f"Features: {', '.join(summary['features'])}")

        console.print(f"Split: {summary['split']}")

        if summary["ignored_columns"]:
            console.print(f"Unused columns: {', '.join(summary['ignored_columns'])}")

    except (ValueError, OSError) as exc:
        console.print(f"[red]{exc}[/red]")

        raise typer.Exit(1) from exc


@app.command()
def datasets():
    """List built-in datasets and the custom-file option. Does not download data."""

    table = Table(title="Available node-classification datasets")

    for title in ("Use with --dataset", "Name", "Split", "Notes"):
        table.add_column(title)

    for identifier, spec in DATASETS.items():
        table.add_row(identifier, spec.name, spec.split, spec.notes)

    console.print(table)

    console.print("All use full-graph training; large graphs may exceed your machine's memory.")


@app.command("inspect")
def inspect_dataset(
    dataset: str | None = None,
    dataset_path: Annotated[Path | None, typer.Option(exists=True)] = None,
    data_dir: str | None = None,
    split_seed: int | None = None,
    split_index: int | None = None,
    config: Annotated[Path | None, typer.Option(exists=True)] = None,
):
    """Load and check data without training. Built-in datasets download on first use."""

    try:
        cfg = load_config(config)

        for name, value in (
            ("dataset", dataset),
            ("data_dir", data_dir),
            ("dataset_path", str(dataset_path) if dataset_path else None),
            ("split_seed", split_seed),
            ("split_index", split_index),
        ):
            if value is not None:
                setattr(cfg, name, value)

        cfg.validate()

        bundle = load_dataset(
            cfg.dataset,
            cfg.data_dir,
            dataset_path=cfg.dataset_path,
            split_seed=cfg.split_seed,
            split_index=cfg.split_index,
        )

        info = bundle.metadata()

        table = Table(title=f"{dataset_name(cfg.dataset)} â€” validation passed")

        table.add_column("Property")

        table.add_column("Value")

        for key in (
            "num_nodes",
            "num_edges",
            "num_features",
            "num_classes",
            "split",
            "train_nodes",
            "val_nodes",
            "test_nodes",
            "feature_preprocessing",
            "edge_policy",
            "dataset_fingerprint",
        ):
            table.add_row(key, str(info[key]))

        console.print(table)

        parameter_table = Table(title="Model sizes with the selected configuration")

        parameter_table.add_column("Model")

        parameter_table.add_column("Trainable parameters")

        for identifier in MODEL_NAMES:
            if model_spec(identifier)["runner"] != "in_process":
                parameter_table.add_row(model_name(identifier), "Measured in reference worker")
                continue
            candidate = copy.deepcopy(cfg)

            candidate.model.name = identifier

            try:
                candidate.validate()

            except ValueError as exc:
                parameter_table.add_row(model_name(identifier), str(exc))

                continue

            model = build_model(candidate.model, bundle.data.num_node_features, bundle.num_classes)

            count = sum(p.numel() for p in model.parameters() if p.requires_grad)

            parameter_table.add_row(model_name(identifier), f"{count:,}")

            del model

        console.print(parameter_table)

        size = sum(
            getattr(bundle.data, key).numel() * getattr(bundle.data, key).element_size()
            for key in ("x", "edge_index", "y", "train_mask", "val_mask", "test_mask")
        )

        console.print(
            f"Input tensors: {size / 1024**2:.1f} MiB. Training requires additional memory."
        )

        console.print(f"Available execution device: {hardware_info()['device_name']}")

    except (ValueError, OSError) as exc:
        console.print(f"[red]{exc}[/red]")

        raise typer.Exit(1) from exc


@app.command()
def run(
    config: Annotated[Path | None, typer.Option(exists=True)] = None,
    dataset: str | None = None,
    model: str | None = None,
    seed: int | None = None,
    epochs: int | None = None,
    device: str | None = None,
    output_dir: str | None = None,
    dataset_path: Annotated[Path | None, typer.Option(exists=True)] = None,
    split_seed: int | None = None,
    split_index: int | None = None,
):
    """Train one model; explicit CLI options override YAML settings."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        cfg = load_config(config)

        for name, value in (
            ("dataset", dataset),
            ("seed", seed),
            ("device", device),
            ("output_dir", output_dir),
            ("dataset_path", str(dataset_path) if dataset_path else None),
            ("split_seed", split_seed),
            ("split_index", split_index),
        ):
            if value is not None:
                setattr(cfg, name, value)

        if model is not None:
            cfg.model.name = model

        if epochs is not None:
            cfg.training.epochs = epochs

        record = run_benchmark(cfg)

        console.print(
            f"{model_name(record['model'])} on {dataset_name(record['dataset'])}: "
            f"{record['status']} ({record['run_id']})"
        )

        if not is_success(record):
            console.print(failure_reason(record), markup=False)
            raise typer.Exit(1)

    except (ValueError, OSError) as exc:
        console.print(f"[red]{exc}[/red]")

        raise typer.Exit(1) from exc


@app.command()
def matrix(
    config: Annotated[Path | None, typer.Option(exists=True)] = None,
    dataset: str | None = None,
    seeds: str = "42,43,44",
    epochs: int | None = None,
    device: str | None = None,
    output_dir: str | None = None,
    dataset_path: Annotated[Path | None, typer.Option(exists=True)] = None,
    split_seed: int | None = None,
    split_index: int | None = None,
    models: str | None = None,
):
    """Run primary models sequentially; --models selects comma-separated model IDs."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        base = load_config(config)

        if dataset is not None:
            base.dataset = dataset

        if dataset_path is not None:
            base.dataset_path = str(dataset_path)

        if split_seed is not None:
            base.split_seed = split_seed

        if split_index is not None:
            base.split_index = split_index

        if epochs is not None:
            base.training.epochs = epochs

        if device is not None:
            base.device = device

        if output_dir is not None:
            base.output_dir = output_dir

        seed_list = list(dict.fromkeys(int(seed.strip()) for seed in seeds.split(",")))

        jobs = []

        selected_models = (
            [m.strip().lower() for m in models.split(",")] if models else PRIMARY_MODELS
        )
        for model in selected_models:
            for seed in seed_list:
                cfg = copy.deepcopy(base)

                cfg.model.name, cfg.seed = model, seed

                cfg.dataset = canonical_dataset(cfg.dataset)
                jobs.append(cfg)

        from efficientgraphbench.benchmark.matrix import run_matrix

        records = run_matrix(jobs, execute=run_benchmark)
        failed = any(not is_success(row) for row in records)
        skipped_runs = [row for row in records if row.get("status", "").startswith("SKIPPED")]
        skip_path = Path(base.output_dir) / "matrix-skipped.json"
        skip_path.parent.mkdir(parents=True, exist_ok=True)
        skip_path.write_text(json.dumps({"skipped_runs": skipped_runs}, indent=2), encoding="utf-8")

        compare(base.dataset, base.output_dir)

        console.print(f"Report: {write_report(base.dataset, base.output_dir)}")

        if failed:
            raise typer.Exit(1)

    except (ValueError, OSError) as exc:
        console.print(f"[red]{exc}[/red]")

        raise typer.Exit(1) from exc


@app.command()
def compare(dataset: str = "cora", output_dir: str = "experiments", report: bool = False):
    """Show the latest comparison table for this dataset."""

    dataset = canonical_dataset(dataset)

    records = latest_results(read_results(output_dir), dataset)

    if not records:
        console.print("No matching results. Run egbench run or egbench matrix first.")

        return

    failed = sum(not is_success(r) for r in records)

    console.print(f"{len(records)} runs; {failed} failed/OOM (excluded from averages).")

    frame = pd.DataFrame([r for r in records if is_success(r)])

    failures = [r for r in records if not is_success(r)]

    if failures:
        failure_table = Table(title="Failed runs")

        for heading in ("Model", "Status", "Nodes", "Edges", "Reason", "Limit"):
            failure_table.add_column(heading, overflow="fold")

        for row in failures:
            failure_table.add_row(
                model_name(row["model"]),
                status_of(row),
                str(row.get("num_nodes", "unknown")),
                str(row.get("num_edges", "unknown")),
                failure_reason(row),
                str(row.get("implementation_limit") or "—"),
            )

        console.print(failure_table)

    if frame.empty:
        if report:
            console.print(f"Report: {write_report(dataset, output_dir)}")

        return

    for group_id, group in frame.groupby("comparison_group"):
        table = Table(
            title=f"{dataset_name(dataset)} | {group.iloc[0]['device_name']} | {group_id}"
        )

        for title in (
            "Model",
            "Source",
            "Runs",
            "Accuracy mean ± sd",
            "Prep s",
            "Train s",
            "Latency ms",
            "GPU MiB\nalloc / reserved",
        ):
            table.add_column(
                title, min_width=9 if title == "Model" else 1, no_wrap=title == "Model"
            )

        grouped_models = list(group.groupby("model"))

        def is_transformer(item):

            name, rows = item

            metadata = rows.iloc[0].get("model_metadata") or MODEL_METADATA.get(name, {})

            return metadata.get("family") == "Graph Transformer"

        grouped_models.sort(key=lambda item: (is_transformer(item), model_name(item[0])))

        previous_family = None

        for name, rows in grouped_models:
            transformer = is_transformer((name, rows))

            if previous_family is False and transformer:
                table.add_section()

            previous_family = transformer

            sd = rows.accuracy.std(ddof=1) if len(rows) > 1 else 0

            gpu = rows.peak_gpu_memory_mb.mean()

            table.add_row(
                model_name(name),
                {
                    "official_repo": "Ref",
                    "library": "PyG",
                    "adapted": "Adapted",
                    "custom_reimplementation": "Custom",
                }.get(str(rows.iloc[0].get("source_type")), "Legacy"),
                str(len(rows)),
                f"{rows.accuracy.mean():.4f} ± {sd:.4f}",
                f"{rows.preprocessing_time_sec.mean():.3f}"
                if "preprocessing_time_sec" in rows
                else "N/A",
                f"{rows.train_time_sec.mean():.3f}",
                f"{rows.inference_ms.mean():.3f}",
                (f"{gpu:.1f}" if pd.notna(gpu) else "N/A")
                + " / "
                + (
                    f"{rows.peak_gpu_reserved_mib.mean():.1f}"
                    if "peak_gpu_reserved_mib" in rows and rows.peak_gpu_reserved_mib.notna().any()
                    else "N/A"
                ),
            )

        Console(width=max(console.width, 120)).print(table)

    if report:
        console.print(f"Report: {write_report(dataset, output_dir)}")


@app.command()
def setup(model: str):
    """Install and verify an isolated reference environment."""
    from efficientgraphbench.environments import setup_environment

    try:
        console.print_json(data=setup_environment(model))
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        console.print(str(exc), markup=False)
        raise typer.Exit(1) from exc


@app.command()
def environments():
    """Show installed reference environments and pinned commits."""
    from efficientgraphbench.environments.manager import NAMES, inspect_environment

    for name in NAMES:
        spec = inspect_environment(name)
        console.print(f"{model_name(name)}: {spec['status']} | commit: {spec.get('commit')}")


@app.command()
def report(dataset: str = "cora", output_dir: str = "experiments"):
    """Generate a Markdown report from saved benchmark results."""
    console.print(str(write_report(canonical_dataset(dataset), output_dir)))


app.command("scaling")(scale)

if __name__ == "__main__":
    app()
