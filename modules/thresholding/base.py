class ThresholdModel:
    """Base interface for threshold models that convert scores to binary labels."""

    # abstract class of thresholding methods
    # whatever the moethod, it must have a predict function
    # to give binary classification {0: normal, 1: anomaly}

    def predict(self, scores):
        """Convert anomaly scores into binary predictions."""
        raise NotImplementedError
