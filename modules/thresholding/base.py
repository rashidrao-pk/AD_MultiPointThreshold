class ThresholdModel:
    # abstract class of thresholding methods
    def __init__(self, threshold):
        self.threshold = threshold

    def predict(self, scores):
        return scores > self.threshold