class ThresholdModel:
    # abstract class of thresholding methods
    # whatever the moethod, it must have a predict function
    # to give binary classification {0: normal, 1: anomaly}

    
    def predict(self, scores):
        raise NotImplementedError