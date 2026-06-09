from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def get_transforms(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])


def list_images(root):
    root = Path(root)
    if not root.exists():
        return []

    return sorted([
        p for p in root.rglob("*")
        if p.suffix.lower() in IMG_EXTS
    ])


class CorridorDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_dataloaders_corridor(cfg):
    root = Path(cfg.dataset_root)
    transform = get_transforms(cfg.img_size)

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

    train_dataset = CorridorDataset(train_samples, transform=transform)
    test_dataset = CorridorDataset(test_samples, transform=transform)

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