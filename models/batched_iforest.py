import pandas as pd
from sklearn.ensemble import IsolationForest
from .base_window_model import BaseWindowModel


class BatchedIForest(BaseWindowModel):
    def __init__(self, anomaly_threshold, **kwargs):
        super().__init__(anomaly_threshold)
        self._model = IsolationForest(**kwargs)
        self._is_fitted = False

    def fit_predict(self, X, y=None):
        # If the model hasn't been fitted yet, it
        # means it's the first batch we're seeing
        if not self._is_fitted:
            labels = self._model.fit_predict(X)
            self._is_fitted = True

            return labels

        labels = self._model.predict(X)

        if isinstance(labels, pd.DataFrame):
            outlier_number = len(labels[labels == -1])
        else:
            outlier_number = len([label for label in labels if label == -1])

        anomaly_rate = outlier_number / len(labels)
        if anomaly_rate > self.anomaly_threshold:
            labels = self._model.fit_predict(X)

        return labels
