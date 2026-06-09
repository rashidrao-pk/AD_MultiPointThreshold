from pathlib import Path
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def get_transforms(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])


def get_dataloaders_mvtec(cfg):
    root = Path(cfg.dataset_root) / cfg.category
    transform = get_transforms(cfg.img_size)

    train_dataset = datasets.ImageFolder(
        root=root / "train",
        transform=transform
    )

    test_dataset = datasets.ImageFolder(
        root=root / "test",
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers
    )

    return train_loader, test_loader, train_dataset, test_dataset