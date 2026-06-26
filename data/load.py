from pathlib import Path

from .mvtec_loading import get_dataloaders_mvtec
from .cobots_loading import get_dataloaders_cobots, canonicalize_area, VALID_COBOTS_AREAS
from .corridor_loading import get_dataloaders_corridor


DATASET_ALIASES = {
    "mvtec": "MVTec",
    "mvtec_ad": "MVTec",
    "mvtec-ad": "MVTec",
    "cobot": "Cobots_Synthetic",
    "cobots": "Cobots_Synthetic",
    "cobots_synthetic": "Cobots_Synthetic",
    "distrimuse_unigra": "Cobots_Synthetic",
    "robotics_hazards": "Robotics_Hazards",
    "hazards": "Robotics_Hazards",
    "corridor": "Robotics_Hazards",
}


def _normalize_name(name):
    key = str(name or "").strip().replace(" ", "_").lower()
    return DATASET_ALIASES.get(key, name)


def infer_dataset_from_structure(data_cfg):
    root = Path(data_cfg.dataset_root)
    category = getattr(data_cfg, "category", None)
    area = canonicalize_area(category) if category is not None else None

    # Cobots:
    # root/train/PLeft/normal
    # root/test/unexpected_person/PLeft/unexpected_person
    if area in VALID_COBOTS_AREAS:
        if (
            (root / "train" / area / "normal").exists()
            or (root / "test" / "unexpected_person" / area).exists()
        ):
            return "Cobots_Synthetic"

    # Hazards:
    # root/train/normal
    # root/test/cable
    if (root / "train" / "normal").exists() and (root / "test").exists():
        return "Robotics_Hazards"

    # MVTec:
    # root/hazelnut/train/good
    # root/hazelnut/test/crack
    if category is not None:
        object_root = root / str(category)
        if (object_root / "train").exists() and (object_root / "test").exists():
            return "MVTec"

    return None


def resolve_dataset_name(data_cfg):
    explicit = _normalize_name(getattr(data_cfg, "name", None))
    inferred = infer_dataset_from_structure(data_cfg)

    if inferred is not None and explicit != inferred:
        print(
            f"[!] Dataset name/config mismatch: data.name={explicit!r}, "
            f"but folder structure looks like {inferred!r}. Using {inferred!r}."
        )
        return inferred

    if inferred is not None:
        return inferred

    return explicit


def load_data(cfg):
    dataset_name = resolve_dataset_name(cfg.data)

    print("[+] Dataset loader selection")
    print(f"    data.name    : {getattr(cfg.data, 'name', None)}")
    print(f"    dataset_root : {getattr(cfg.data, 'dataset_root', None)}")
    print(f"    category/area: {getattr(cfg.data, 'category', None)}")

    if dataset_name == "MVTec":
        return get_dataloaders_mvtec(cfg.data)

    if dataset_name == "Cobots_Synthetic":
        return get_dataloaders_cobots(cfg.data)

    if dataset_name == "Robotics_Hazards":
        return get_dataloaders_corridor(cfg.data)

    raise ValueError(
        f"Unsupported dataset: {dataset_name}. "
        "Supported: MVTec, Cobots_Synthetic, Robotics_Hazards."
    )