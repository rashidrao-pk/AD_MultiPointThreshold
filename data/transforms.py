from torchvision import transforms


def build_transforms(img_size, augmentation="none"):
    """Build train and evaluation image transforms for the configured image size."""
    augmentation = str(augmentation or "none").lower()

    train_ops = [transforms.Resize((img_size, img_size))]
    if augmentation in {"light", "custom"}:
        train_ops.extend(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10)],
                    p=0.5,
                ),
                transforms.RandomAffine(
                    degrees=1,
                    translate=(0.02, 0.02),
                    scale=(0.99, 1.01),
                    fill=0,
                ),
            ]
        )
    elif augmentation == "cad_custom":
        train_ops.extend(
            [
                transforms.RandomAffine(
                    degrees=0.01,
                    translate=(0.01, 0.01),
                    shear=0.1,
                    scale=(0.99, 1.0),
                    fill=(0, 0, 0),
                ),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        )
    elif augmentation in {"cad_max", "strong"}:
        train_ops.extend(
            [
                transforms.RandomAffine(
                    degrees=5,
                    translate=(0.05, 0.05),
                    shear=5,
                    scale=(0.95, 1.0),
                    fill=(0, 0, 0),
                ),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
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
