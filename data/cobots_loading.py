from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from .transforms import build_transforms


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

VALID_COBOTS_AREAS = ["PLeft", "PRight", "RoboArm", "ConvBelt"]

AREA_ALIASES = {
    "pleft": "PLeft",
    "left": "PLeft",
    "pright": "PRight",
    "right": "PRight",
    "roboarm": "RoboArm",
    "robot": "RoboArm",
    "convbelt": "ConvBelt",
    "conveyor": "ConvBelt",
    "conveyorbelt": "ConvBelt",
}


def canonicalize_area(area):
    """Map Cobots area aliases to canonical area names."""
    key = str(area).replace("_", "").replace("-", "").replace(" ", "").lower()
    return AREA_ALIASES.get(key, area)


def list_images(root):
    """Return sorted image paths under a directory."""
    root = Path(root)
    if not root.exists():
        return []

    return sorted([
        p for p in root.rglob("*")
        if p.suffix.lower() in IMG_EXTS
    ])


class CobotsDataset(Dataset):
    """Dataset wrapper for Cobots image samples and integer labels."""

    def __init__(self, samples, transform=None, classes=None, class_to_idx=None):
        """Store Cobots samples, transforms, labels, and class metadata."""
        self.samples = samples
        self.imgs = samples
        self.targets = [label for _, label in samples]
        self.transform = transform
        self.classes = classes or ["normal", "unexpected_person"]
        self.class_to_idx = class_to_idx or {name: idx for idx, name in enumerate(self.classes)}

    def __len__(self):
        """Return the number of image samples."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Load one RGB image and its label by index."""
        path, label = self.samples[idx]

        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_dataloaders_cobots(cfg):
    """Build train and test dataloaders for a Cobots safety area."""
    dataset_root = Path(cfg.dataset_root)
    area = canonicalize_area(cfg.category)

    if area not in VALID_COBOTS_AREAS:
        raise ValueError(
            f"Invalid Cobots area: {area}. "
            f"Valid areas: {VALID_COBOTS_AREAS}"
        )

    train_normal_root = dataset_root / "train" / area / "normal"

    test_normal_root = (
        dataset_root / "test" / "unexpected_person" / area / "normal"
    )

    test_anomaly_root = (
        dataset_root / "test" / "unexpected_person" / area / "unexpected_person"
    )

    if not train_normal_root.exists():
        raise FileNotFoundError(f"Missing train normal path: {train_normal_root}")

    if not test_normal_root.exists():
        raise FileNotFoundError(f"Missing test normal path: {test_normal_root}")

    train_normal = list_images(train_normal_root)
    test_normal = list_images(test_normal_root)
    test_anomaly = list_images(test_anomaly_root)

    train_samples = [(p, 0) for p in train_normal]

    test_samples = (
        [(p, 0) for p in test_normal] +
        [(p, 1) for p in test_anomaly]
    )

    if len(train_samples) == 0:
        raise RuntimeError(f"No training images found in: {train_normal_root}")

    if len(test_samples) == 0:
        raise RuntimeError("No test images found.")

    train_transform, eval_transform = build_transforms(
        cfg.img_size,
        augmentation=getattr(cfg, "augmentation", "none"),
    )
    classes = ["normal", "unexpected_person"]
    class_to_idx = {"normal": 0, "unexpected_person": 1}

    train_dataset = CobotsDataset(
        train_samples,
        transform=train_transform,
        classes=classes,
        class_to_idx=class_to_idx,
    )
    test_dataset = CobotsDataset(
        test_samples,
        transform=eval_transform,
        classes=classes,
        class_to_idx=class_to_idx,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=getattr(cfg, "pin_memory", False),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=getattr(cfg, "pin_memory", False),
    )

    print("[+] Cobots dataset loaded")
    print(f"    Area: {area}")
    print(f"    Train normal: {len(train_normal)}")
    print(f"    Test normal: {len(test_normal)}")
    print(f"    Test anomaly: {len(test_anomaly)}")

    return train_loader, test_loader, train_dataset, test_dataset
