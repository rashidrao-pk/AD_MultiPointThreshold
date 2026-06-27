from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset, DataLoader

from .transforms import build_transforms


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def list_images(root):
    """Return sorted image paths under a directory."""
    root = Path(root)
    if not root.exists():
        return []

    return sorted([
        p for p in root.rglob("*")
        if p.suffix.lower() in IMG_EXTS
    ])


class CorridorDataset(Dataset):
    """Dataset wrapper for corridor image samples and binary labels."""

    def __init__(self, samples, transform=None, classes=None, class_to_idx=None):
        """Store corridor samples, transforms, labels, and class metadata."""
        self.samples = samples
        self.imgs = samples
        self.targets = [label for _, label in samples]
        self.transform = transform
        self.classes = classes or ["normal", "anomaly"]
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


def get_dataloaders_corridor(cfg):
    """Build train and test dataloaders for the robotics hazards corridor dataset."""
    root = Path(cfg.dataset_root)
    train_transform, eval_transform = build_transforms(
        cfg.img_size,
        augmentation=getattr(cfg, "augmentation", "none"),
    )

    train_normal_root = root / "train" / "normal"
    test_root = root / "test"

    train_normal = list_images(train_normal_root)

    test_samples = []
    for cls_dir in sorted(test_root.iterdir()):
        if cls_dir.is_dir():
            for img_path in list_images(cls_dir):
                test_samples.append((img_path, 1))

    train_samples = [(p, 0) for p in train_normal]

    if len(train_samples) == 0:
        raise RuntimeError(f"No training images found in: {train_normal_root}")

    if len(test_samples) == 0:
        raise RuntimeError(f"No test anomaly images found in: {test_root}")

    classes = ["normal", "anomaly"]
    class_to_idx = {"normal": 0, "anomaly": 1}
    train_dataset = CorridorDataset(
        train_samples,
        transform=train_transform,
        classes=classes,
        class_to_idx=class_to_idx,
    )
    test_dataset = CorridorDataset(
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

    print("[+] Corridor dataset loaded")
    print(f"    Train normal: {len(train_samples)}")
    print(f"    Test anomalies: {len(test_samples)}")

    return train_loader, test_loader, train_dataset, test_dataset
