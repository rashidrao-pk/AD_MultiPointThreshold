from pathlib import Path
import random

from PIL import Image

from torch.utils.data import Dataset, DataLoader

from .transforms import build_transforms


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def list_images(root):
    """Return sorted image paths under a directory."""
    root = Path(root)
    if not root.exists():
        return []

    return sorted([p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS])


def split_normal_images(images, train_fraction=0.8, seed=42):
    """Split normal images into deterministic train and test subsets."""
    images = list(images)
    if not images:
        return [], []

    rng = random.Random(int(seed))
    shuffled = images[:]
    rng.shuffle(shuffled)

    train_count = int(round(len(shuffled) * float(train_fraction)))
    train_count = min(max(train_count, 1), len(shuffled))
    if len(shuffled) > 1:
        train_count = min(train_count, len(shuffled) - 1)

    return sorted(shuffled[:train_count]), sorted(shuffled[train_count:])


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

    normal_train_fraction = float(getattr(cfg, "normal_train_fraction", 0.8))
    normal_split_seed = int(getattr(cfg, "normal_split_seed", 42))
    train_normal_all = list_images(train_normal_root)
    train_normal, test_normal = split_normal_images(
        train_normal_all,
        train_fraction=normal_train_fraction,
        seed=normal_split_seed,
    )

    test_samples = []
    for cls_dir in sorted(test_root.iterdir()):
        if cls_dir.is_dir():
            for img_path in list_images(cls_dir):
                test_samples.append((img_path, 1))
    test_samples = [(p, 0) for p in test_normal] + test_samples

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
    print(
        f"    Normal split: {normal_train_fraction:.2f} train / {1.0 - normal_train_fraction:.2f} test"
    )
    print(f"    Train normal: {len(train_samples)}")
    print(f"    Test normal: {len(test_normal)}")
    print(f"    Test anomalies: {len(test_samples) - len(test_normal)}")

    return train_loader, test_loader, train_dataset, test_dataset
