from torchvision import transforms


def build_transforms(img_size, augmentation="none"):
    """Build train and evaluation image transforms for the configured image size."""
    augmentation = str(augmentation or "none").lower()

    train_ops = [transforms.Resize((img_size, img_size))]
    if augmentation in {"light", "custom"}:
        train_ops.extend(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10)],
                    p=0.5,
                ),
                transforms.RandomAffine(
                    degrees=3,
                    translate=(0.02, 0.02),
                    scale=(0.98, 1.02),
                    fill=0,
                ),
            ]
        )
    elif augmentation not in {"none", "false", "0", "min"}:
        raise ValueError(f"Unsupported augmentation: {augmentation}")

    tensor_ops = [
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ]

    train_transform = transforms.Compose(train_ops + tensor_ops)
    eval_transform = transforms.Compose([transforms.Resize((img_size, img_size)), *tensor_ops])
    return train_transform, eval_transform
