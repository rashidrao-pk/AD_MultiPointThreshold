import yaml
from types import SimpleNamespace
import time
from pathlib import Path

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{
            k: dict_to_namespace(v) for k, v in d.items()
        })
    return d

def read_config(path):
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    return dict_to_namespace(cfg_dict)


def namespace_to_dict(obj):
    if hasattr(obj, "__dict__"):
        return {k: namespace_to_dict(v) for k, v in vars(obj).items()}

    if isinstance(obj, list):
        return [namespace_to_dict(v) for v in obj]

    if isinstance(obj, tuple):
        return tuple(namespace_to_dict(v) for v in obj)

    return obj


def save_config_yaml(config, path):
    with open(path, "w") as f:
        yaml.safe_dump(
            namespace_to_dict(config),
            f,
            sort_keys=False,
            default_flow_style=False,
        )

def make_run_dir(config, suffix=""):
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")

    dataset_name = config.data.name
    category = getattr(config.data, "category", "all")
    scoring_method = config.scoring.method
    threshold_method = config.threshold.method

    run_name = f"{dataset_name}_{category}_{scoring_method}_{threshold_method}_{timestamp}"

    if suffix:
        run_name += f"_{suffix}"

    output_dir = Path(config.output.dir)
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir