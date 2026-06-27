import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch


def _json_safe(value):
    """Convert nested checkpoint metadata into JSON-serializable values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, SimpleNamespace):
        return {key: _json_safe(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return {key: _json_safe(item) for key, item in vars(value).items()}
    return str(value)


def _loss_history_to_records(loss_history):
    """Normalize supported loss-history formats to a list of records."""
    if loss_history is None:
        return []
    if isinstance(loss_history, pd.DataFrame):
        return loss_history.to_dict("records")
    if isinstance(loss_history, list):
        return loss_history
    return pd.DataFrame(loss_history).to_dict("records")


def build_dataset_summary(train_dataset=None, val_dataset=None, test_dataset=None, config=None):
    """Collect the dataset facts a future reader needs to understand a checkpoint."""
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_data": _json_safe(getattr(config, "data", None)) if config is not None else None,
    }

    for name, dataset in (
        ("train", train_dataset),
        ("val", val_dataset),
        ("test", test_dataset),
    ):
        if dataset is None:
            continue
        item = {"num_samples": len(dataset)}
        for attr in ("classes", "class_to_idx", "root", "samples"):
            if hasattr(dataset, attr):
                value = getattr(dataset, attr)
                if attr == "samples":
                    item["num_file_samples"] = len(value)
                else:
                    item[attr] = _json_safe(value)
        summary[name] = item

    return summary


def save_checkpoint(
    path,
    encoder,
    decoder,
    discriminator,
    optimizer_enc_dec=None,
    optimizer_dis=None,
    epoch=None,
    loss_history=None,
    config=None,
    dataset_summary=None,
    metrics=None,
    model_name="vaegan",
    notes=None,
    write_sidecars=True,
):
    """Save a VAE-GAN training checkpoint and optional readable sidecar files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    loss_records = _loss_history_to_records(loss_history)
    config_dict = _json_safe(config)
    dataset_summary = _json_safe(dataset_summary)

    checkpoint = {
        "schema_version": 1,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "model_name": model_name,
        "epoch": epoch,
        "epochs_trained": len(loss_records),
        "encoder_state_dict": encoder.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "discriminator_state_dict": discriminator.state_dict() if discriminator is not None else None,
        "optimizer_enc_state_dict": (
            optimizer_enc_dec.state_dict() if optimizer_enc_dec is not None else None
        ),
        "optimizer_dis_state_dict": optimizer_dis.state_dict() if optimizer_dis is not None else None,
        # Backward-compatible key used by the older notebook-derived training script.
        "optimizer_dec_state_dict": optimizer_dis.state_dict() if optimizer_dis is not None else None,
        "loss_history": loss_records,
        "config": config_dict,
        "dataset_summary": dataset_summary,
        "metrics": _json_safe(metrics),
        "notes": notes,
    }

    torch.save(checkpoint, path)

    if write_sidecars:
        stem = path.with_suffix("")
        if config_dict is not None:
            with open(stem.with_name(f"{stem.name}_config.json"), "w", encoding="utf-8") as handle:
                json.dump(config_dict, handle, indent=2)
        if dataset_summary is not None:
            with open(
                stem.with_name(f"{stem.name}_dataset_summary.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(dataset_summary, handle, indent=2)
        if loss_records:
            pd.DataFrame(loss_records).to_csv(
                stem.with_name(f"{stem.name}_loss_history.csv"),
                index=False,
            )

    return path


def load_checkpoint(
    path,
    encoder=None,
    decoder=None,
    discriminator=None,
    optimizer_enc_dec=None,
    optimizer_dis=None,
    device="cpu",
    strict=True,
):
    """Load checkpoint data and optionally restore model/optimizer objects."""
    path = Path(path)
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if encoder is not None:
        encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=strict)
        encoder.to(device)
    if decoder is not None:
        decoder.load_state_dict(checkpoint["decoder_state_dict"], strict=strict)
        decoder.to(device)
    if discriminator is not None and checkpoint.get("discriminator_state_dict") is not None:
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"], strict=strict)
        discriminator.to(device)

    if optimizer_enc_dec is not None and checkpoint.get("optimizer_enc_state_dict") is not None:
        optimizer_enc_dec.load_state_dict(checkpoint["optimizer_enc_state_dict"])
    optimizer_dis_state = checkpoint.get("optimizer_dis_state_dict")
    if optimizer_dis_state is None:
        optimizer_dis_state = checkpoint.get("optimizer_dec_state_dict")
    if optimizer_dis is not None and optimizer_dis_state is not None:
        optimizer_dis.load_state_dict(optimizer_dis_state)

    return checkpoint
