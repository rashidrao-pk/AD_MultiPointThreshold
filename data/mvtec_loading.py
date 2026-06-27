from pathlib import Path
from torchvision import datasets
from torch.utils.data import DataLoader

from .transforms import build_transforms


def get_dataloaders_mvtec(cfg):
    """Build train and test dataloaders for an MVTec category."""
    root = Path(cfg.dataset_root) / cfg.category
    train_transform, eval_transform = build_transforms(
        cfg.img_size,
        augmentation=getattr(cfg, "augmentation", "none"),
    )

    train_dataset = datasets.ImageFolder(
        root=root / "train",
        transform=train_transform
    )

    test_dataset = datasets.ImageFolder(
        root=root / "test",
        transform=eval_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=getattr(cfg, "pin_memory", False),
        drop_last=getattr(cfg, "drop_last", False),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=getattr(cfg, "pin_memory", False),
    )

    return train_loader, test_loader, train_dataset, test_dataset
