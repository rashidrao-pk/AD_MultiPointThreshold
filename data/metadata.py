from pathlib import Path


def sample_records(dataset, limit=None):
    """Return path/label/class/folder records for datasets exposing .samples."""
    samples = list(getattr(dataset, "samples", []))
    if limit is not None:
        samples = samples[:limit]

    idx_to_class = {
        idx: name for name, idx in getattr(dataset, "class_to_idx", {}).items()
    }

    records = []
    for path, label in samples:
        path = Path(path)
        class_name = idx_to_class.get(int(label), path.parent.name)
        records.append(
            {
                "path": str(path),
                "file_name": path.name,
                "folder_name": path.parent.name,
                "class_idx": int(label),
                "class_name": class_name,
            }
        )
    return records


def dataset_summary(dataset, name):
    records = sample_records(dataset)
    class_counts = {}
    for record in records:
        class_counts[record["class_name"]] = class_counts.get(record["class_name"], 0) + 1

    return {
        "name": name,
        "num_samples": len(dataset),
        "classes": list(getattr(dataset, "classes", [])),
        "class_to_idx": getattr(dataset, "class_to_idx", {}),
        "class_counts": class_counts,
        "sample_records": records[:10],
    }
