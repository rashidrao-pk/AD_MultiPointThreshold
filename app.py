"""Small live viewer for training plots."""

from __future__ import annotations

import json
import re
import csv
import time
import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import yaml


ROOT = Path(__file__).resolve().parent
RUNS_ROOT = ROOT / "results" / "runs"
EXPERIMENTS_ROOT = ROOT / "results" / "experiments"
CONFIGS_ROOT = ROOT / "configs"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

PLOT_TYPES = {
    "curves": ("plots/training_curves", "training_curves_epoch_"),
    "latent": ("plots/latent_space", "latent_space_epoch_"),
    "evolution": ("plots/training_evolution", "training_evolution_epoch_"),
    "scores": ("plots/score_distribution", "score_distribution_epoch_"),
    "quality": ("plots/quality_metrics", "quality_metrics_epoch_"),
    "components": ("plots/score_components", "score_components_epoch_"),
    "radius": ("plots/latent_radius", "latent_radius_epoch_"),
    "loss_balance": ("plots/loss_balance", "loss_balance_epoch_"),
    "validation": ("plots/validation_quality", "validation_quality_epoch_"),
}

INFERENCE_PLOT_TYPES = {
    "scores": ("plots", "score_distribution.png"),
    "latent": ("plots", "latent_space.png"),
    "samples": ("plots", "validation_samples.png"),
    "confusion": ("plots", "confusion_matrix.png"),
    "roc": ("plots", "roc_curve.png"),
    "pr": ("plots", "precision_recall_curve.png"),
    "outcomes": ("plots", "outcome_counts.png"),
    "classes": ("plots", "mean_score_by_class.png"),
    "tp": ("plots", "top_tp_gallery.png"),
    "tn": ("plots", "top_tn_gallery.png"),
    "fp": ("plots", "top_fp_gallery.png"),
    "fn": ("plots", "top_fn_gallery.png"),
}

LOSS_HISTORY = "loss_history.csv"


def _json_response(handler, payload, status=200):
    """Send a JSON response."""
    body = json.dumps(payload, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def _sse_headers(handler):
    """Start a Server-Sent Events response."""
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Connection", "keep-alive")
    handler.end_headers()


def _sse_event(handler, event, payload):
    """Send one Server-Sent Events message."""
    body = (
        f"event: {event}\n"
        f"data: {json.dumps(payload, default=str)}\n\n"
    ).encode("utf-8")
    handler.wfile.write(body)
    handler.wfile.flush()


def _safe_run_path(run):
    """Resolve a run path under the project root."""
    run = unquote(str(run or "")).strip()
    if not run:
        raise ValueError("Missing run path.")

    path = Path(run)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()

    if ROOT not in path.parents and path != ROOT:
        raise ValueError("Run path must stay inside the project folder.")
    if not path.exists():
        raise FileNotFoundError(f"Run path does not exist: {path}")
    return path


def _relative_to_root(path):
    """Return a browser-friendly relative path."""
    return path.resolve().relative_to(ROOT).as_posix()


def list_runs():
    """Return known result run folders."""
    if not RUNS_ROOT.exists():
        return []
    runs = [path for path in RUNS_ROOT.iterdir() if path.is_dir()]
    return [
        {
            "name": path.name,
            "path": _relative_to_root(path),
            "mtime": path.stat().st_mtime,
        }
        for path in sorted(runs, key=lambda item: item.stat().st_mtime, reverse=True)
    ]


def list_experiments():
    """Return known inference experiment folders."""
    if not EXPERIMENTS_ROOT.exists():
        return []
    experiments = [path for path in EXPERIMENTS_ROOT.iterdir() if path.is_dir()]
    return [
        {
            "name": path.name,
            "path": _relative_to_root(path),
            "mtime": path.stat().st_mtime,
        }
        for path in sorted(experiments, key=lambda item: item.stat().st_mtime, reverse=True)
    ]


def list_config_files():
    """Return project and saved-run YAML config files."""
    configs = []
    if CONFIGS_ROOT.exists():
        for path in sorted(CONFIGS_ROOT.glob("*.yaml")):
            configs.append(
                {
                    "name": path.name,
                    "path": _relative_to_root(path),
                    "source": "configs",
                    "mtime": path.stat().st_mtime,
                }
            )

    if RUNS_ROOT.exists():
        for path in sorted(RUNS_ROOT.glob("*/config.yaml"), key=lambda item: item.stat().st_mtime, reverse=True):
            configs.append(
                {
                    "name": f"{path.parent.name}/config.yaml",
                    "path": _relative_to_root(path),
                    "source": "runs",
                    "mtime": path.stat().st_mtime,
                }
            )
    return configs


def _safe_config_path(config_path):
    """Resolve a config path under configs or saved training runs."""
    path = _safe_run_path(config_path)
    allowed_configs = CONFIGS_ROOT.resolve()
    allowed_runs = RUNS_ROOT.resolve()
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("Config path must be a YAML file.")
    if (
        (allowed_configs not in path.parents and path != allowed_configs)
        and allowed_runs not in path.parents
    ):
        raise ValueError("Config path must be inside configs/ or results/runs/.")
    return path


def read_config_file(config_path):
    """Read a YAML config file as text and parsed data."""
    path = _safe_config_path(config_path)
    text = path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(text) or {}
    is_run_config = RUNS_ROOT.resolve() in path.parents
    return {
        "name": f"{path.parent.name}/{path.name}" if is_run_config else path.name,
        "path": _relative_to_root(path),
        "source": "runs" if is_run_config else "configs",
        "text": text,
        "parsed": parsed,
        "mtime": path.stat().st_mtime,
    }


def inference_plot_status(run_path):
    """Return available inference plots for an experiment directory."""
    plots = {}
    latest_mtime = 0.0
    for tab, (folder, filename) in INFERENCE_PLOT_TYPES.items():
        path = run_path / folder / filename
        exists = path.exists()
        if exists:
            latest_mtime = max(latest_mtime, path.stat().st_mtime)
        plots[tab] = {
            "exists": exists,
            "path": _relative_to_root(path) if exists else None,
            "mtime": path.stat().st_mtime if exists else None,
        }

    metrics_path = run_path / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        latest_mtime = max(latest_mtime, metrics_path.stat().st_mtime)

    return {
        "run": _relative_to_root(run_path),
        "plots": plots,
        "metrics": metrics,
        "seconds_since_update": time.time() - latest_mtime if latest_mtime else None,
        "server_time": time.time(),
    }


def list_epochs(run_path, tab):
    """Return epochs with available plot images for a tab."""
    if tab not in PLOT_TYPES:
        raise ValueError(f"Unknown tab: {tab}")

    folder, prefix = PLOT_TYPES[tab]
    plot_dir = run_path / folder
    if not plot_dir.exists():
        return []

    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)\.png$")
    epochs = []
    for path in plot_dir.iterdir():
        match = pattern.match(path.name)
        if match:
            epochs.append(int(match.group(1)))
    return sorted(epochs)


def read_loss_history(run_path):
    """Read training loss history records for a run."""
    path = run_path / LOSS_HISTORY
    if not path.exists():
        return []

    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_run_config(run_path):
    """Read the config.yaml saved inside a training run directory."""
    path = run_path / "config.yaml"
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _read_csv_rows(path):
    """Read CSV rows from a file if it exists."""
    if not path.exists():
        return []

    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def find_run_record(run_path):
    """Find the training-run registry row matching a run directory."""
    run_dir = str(run_path.resolve())
    candidates = [
        ROOT / "results" / "training_runs.csv",
        ROOT / "results" / "runs.csv",
    ]
    for path in candidates:
        for row in _read_csv_rows(path):
            row_run_dir = row.get("run_dir") or row.get("path") or ""
            if row_run_dir and str(Path(row_run_dir).resolve()) == run_dir:
                return row
    return {}


def checkpoint_summary(run_path):
    """Return basic checkpoint file locations and modification times for a run."""
    items = {}
    for name in ("model_best.pt", "model_last.pt"):
        path = run_path / name
        if path.exists():
            items[name] = {
                "path": _relative_to_root(path),
                "mtime": path.stat().st_mtime,
                "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
            }
    return items


def run_details(run_path):
    """Build model, data, training, and checkpoint details for a run."""
    config = read_run_config(run_path)
    record = find_run_record(run_path)
    return {
        "config": config,
        "record": record,
        "checkpoints": checkpoint_summary(run_path),
    }


def run_status(run_path):
    """Build a live status payload from run files."""
    epochs = {
        tab: list_epochs(run_path, tab)
        for tab in PLOT_TYPES
    }
    history = read_loss_history(run_path)
    latest_record = history[-1] if history else {}
    loss_path = run_path / LOSS_HISTORY
    latest_mtime = max(
        [loss_path.stat().st_mtime if loss_path.exists() else 0.0]
        + [
            path.stat().st_mtime
            for plot_folder, _ in PLOT_TYPES.values()
            for path in (run_path / plot_folder).glob("*.png")
        ],
        default=0.0,
    )
    seconds_since_update = time.time() - latest_mtime if latest_mtime else None

    return {
        "run": _relative_to_root(run_path),
        "run_details": run_details(run_path),
        "epochs": epochs,
        "latest": {
            tab: values[-1] if values else None
            for tab, values in epochs.items()
        },
        "loss_history": history,
        "latest_metrics": latest_record,
        "history_rows": len(history),
        "seconds_since_update": seconds_since_update,
        "is_live": seconds_since_update is not None and seconds_since_update < 120,
        "server_time": time.time(),
    }


class TrainingViewerHandler(SimpleHTTPRequestHandler):
    """HTTP handler with small JSON APIs for live training plots."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        """Route API requests and static files."""
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.path = "/app/home.html"
            return super().do_GET()
        if parsed.path == "/training":
            self.path = "/app/training.html"
            return super().do_GET()
        if parsed.path == "/inference":
            self.path = "/app/inference.html"
            return super().do_GET()
        if parsed.path == "/config":
            self.path = "/app/configs.html"
            return super().do_GET()
        if parsed.path == "/api/runs":
            return _json_response(self, {"runs": list_runs()})
        if parsed.path == "/api/experiments":
            return _json_response(self, {"experiments": list_experiments()})
        if parsed.path == "/api/configs":
            return _json_response(self, {"configs": list_config_files()})
        if parsed.path == "/api/config":
            return self._handle_config(parsed)
        if parsed.path == "/api/inference_status":
            return self._handle_inference_status(parsed)
        if parsed.path == "/api/epochs":
            return self._handle_epochs(parsed)
        if parsed.path == "/api/status":
            return self._handle_status(parsed)
        if parsed.path == "/api/events":
            return self._handle_events(parsed)
        return super().do_GET()

    def end_headers(self):
        """Disable caching so newly generated images are visible immediately."""
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _handle_epochs(self, parsed):
        """Return available epochs for every plot tab."""
        query = parse_qs(parsed.query)
        try:
            run_path = _safe_run_path(query.get("run", [""])[0])
            tabs = query.get("tab", [])
            selected_tabs = tabs or list(PLOT_TYPES)
            epochs = {
                tab: list_epochs(run_path, tab)
                for tab in selected_tabs
                if tab in PLOT_TYPES
            }
            return _json_response(
                self,
                {
                    "run": _relative_to_root(run_path),
                    "run_details": run_details(run_path),
                    "epochs": epochs,
                    "latest": {
                        tab: values[-1] if values else None
                        for tab, values in epochs.items()
                    },
                },
            )
        except Exception as exc:
            return _json_response(self, {"error": str(exc)}, status=400)

    def _handle_status(self, parsed):
        """Return latest training metrics and plot epochs."""
        query = parse_qs(parsed.query)
        try:
            run_path = _safe_run_path(query.get("run", [""])[0])
            return _json_response(self, run_status(run_path))
        except Exception as exc:
            return _json_response(self, {"error": str(exc)}, status=400)

    def _handle_inference_status(self, parsed):
        """Return latest inference plot and metric availability."""
        query = parse_qs(parsed.query)
        try:
            run_path = _safe_run_path(query.get("run", [""])[0])
            return _json_response(self, inference_plot_status(run_path))
        except Exception as exc:
            return _json_response(self, {"error": str(exc)}, status=400)

    def _handle_config(self, parsed):
        """Return one YAML config file for browser preview."""
        query = parse_qs(parsed.query)
        try:
            return _json_response(self, read_config_file(query.get("path", [""])[0]))
        except Exception as exc:
            return _json_response(self, {"error": str(exc)}, status=400)

    def _handle_events(self, parsed):
        """Stream live training status using Server-Sent Events."""
        query = parse_qs(parsed.query)
        try:
            run_path = _safe_run_path(query.get("run", [""])[0])
        except Exception as exc:
            return _json_response(self, {"error": str(exc)}, status=400)

        _sse_headers(self)
        last_payload = None
        try:
            while True:
                payload = run_status(run_path)
                signature = json.dumps(
                    {
                        "latest": payload["latest"],
                        "latest_metrics": payload["latest_metrics"],
                        "history_rows": payload["history_rows"],
                        "is_live": payload["is_live"],
                    },
                    sort_keys=True,
                    default=str,
                )
                if signature != last_payload:
                    _sse_event(self, "training_status", payload)
                    last_payload = signature
                else:
                    _sse_event(
                        self,
                        "heartbeat",
                        {
                            "server_time": time.time(),
                            "is_live": payload["is_live"],
                            "seconds_since_update": payload["seconds_since_update"],
                        },
                    )
                time.sleep(2)
        except (BrokenPipeError, ConnectionResetError):
            return


def parse_args():
    """Parse viewer server arguments."""
    parser = argparse.ArgumentParser(description="Serve the live training plot viewer.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host interface to bind.")
    parser.add_argument("--port", default=DEFAULT_PORT, type=int, help="Port to bind.")
    return parser.parse_args()


def main():
    """Run the live training plot viewer."""
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), TrainingViewerHandler)
    url = f"http://{args.host}:{args.port}/training"
    print(f"[+] Training viewer: {url}")
    print("[+] Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[+] Stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
