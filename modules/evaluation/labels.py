import numpy as np


def prepare_binary_labels(labels, class_to_idx=None, normal_names=("good", "normal"), normal_index=0):
    """Convert raw dataset labels into binary anomaly labels.

    Returns 0 for normal and 1 for anomaly.
    """
    labels = np.asarray(labels).astype(int)

    if class_to_idx:
        for normal_name in normal_names:
            if normal_name in class_to_idx:
                normal_index = class_to_idx[normal_name]
                break

    return (labels != int(normal_index)).astype(int)
