import copy
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace

import yaml


# Set up device 
def set_device(config):
    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config.device = str(device)
    else:
        device = torch.device(config.device)
    print(f"[+] Using device: {device}")
    return device




ENV_PATTERN = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")


def deep_merge(base, override):
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if (
            isinstance(value, dict)
            and isinstance(result.get(key), dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def expand_env_value(value):
    if isinstance(value, str):
        def replace(match):
            name, default = match.group(1), match.group(2)
            return os.environ.get(name, default or "")

        previous = None
        expanded = value
        while previous != expanded:
            previous = expanded
            expanded = ENV_PATTERN.sub(replace, expanded)
        return os.path.expanduser(expanded)

    if isinstance(value, dict):
        return {k: expand_env_value(v) for k, v in value.items()}

    if isinstance(value, list):
        return [expand_env_value(v) for v in value]

    return value


def load_yaml_dict(path):
    path = Path(path)
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_local_config_path(config_path, local_config_path=None):
    if local_config_path:
        return Path(local_config_path)

    env_path = os.environ.get("PROJECT_LOCAL_CONFIG")
    if env_path:
        return Path(env_path)

    return Path(config_path).parent / "local.yaml"


def validate_existing_paths(cfg_dict):
    missing = []

    data = cfg_dict.get("data", {})
    if isinstance(data, dict):
        for key in ("dataset_root", "base_dir"):
            value = data.get(key)
            if value and not Path(value).exists():
                missing.append((f"data.{key}", value))

        paths = data.get("paths", {})
        if isinstance(paths, dict):
            for name, value in paths.items():
                if value and not Path(value).exists():
                    missing.append((f"data.paths.{name}", value))

    datasets = cfg_dict.get("datasets", {}).get("available", [])
    if isinstance(datasets, list):
        for dataset in datasets:
            if not isinstance(dataset, dict):
                continue
            name = dataset.get("name", "<unnamed>")
            for key in ("path", "path_train", "path_test"):
                value = dataset.get(key)
                if value and not Path(value).exists():
                    missing.append((f"datasets.available.{name}.{key}", value))

    model = cfg_dict.get("model", {})
    if isinstance(model, dict):
        for key in ("checkpoint_root", "checkpoints_dir"):
            value = model.get(key)
            if value and not Path(value).exists():
                missing.append((f"model.{key}", value))

    if missing:
        details = "\n".join(f"  - {key}: {value}" for key, value in missing)
        raise FileNotFoundError(
            "Configured path(s) do not exist. Set environment variables or "
            "create configs/local.yaml for this machine:\n"
            f"{details}"
        )

import torch

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{
            k: dict_to_namespace(v) for k, v in d.items()
        })
    if isinstance(d, list):
        return [dict_to_namespace(v) for v in d]
    return d

def read_config(path, local_config_path=None, validate_paths=True):
    cfg_dict = load_yaml_dict(path)

    local_path = get_local_config_path(path, local_config_path)
    if local_path.exists():
        cfg_dict = deep_merge(cfg_dict, load_yaml_dict(local_path))

    cfg_dict = expand_env_value(cfg_dict)

    if validate_paths:
        validate_existing_paths(cfg_dict)

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


def resolve_device(requested="auto"):
    """Resolve auto/cuda/mps/cpu into a concrete torch.device."""
    requested = str(requested or "auto").lower()

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        print("[device] CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")

    if requested == "mps":
        mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        if not mps_ok:
            print("[device] MPS requested but unavailable. Falling back to CPU.")
            return torch.device("cpu")

    return torch.device(requested)

def make_run_dir(config, suffix=""):
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")

    dataset_name = config.data.name
    category = getattr(config.data, "category", "all")
    scoring_method = config.scoring.method
    threshold_method = config.threshold.method
    
    run_name = f"{dataset_name}_{category}_{scoring_method}_{threshold_method}"

    if config.threshold.decision_rule is not None:
        threshold_method += f"_{config.threshold.decision_rule}"
        run_name += threshold_method

    if suffix:
        run_name += f"_{suffix}"

    run_name += f"_{timestamp}"

    output_dir = Path(config.output.dir)
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir
