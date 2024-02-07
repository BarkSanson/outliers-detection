from abc import ABC, abstractmethod


class BaseWindowModel(ABC):
    def __init__(self, anomaly_threshold):
        self.anomaly_threshold = anomaly_threshold

    @abstractmethod
    def fit_predict(self, X, y=None):
        pass
