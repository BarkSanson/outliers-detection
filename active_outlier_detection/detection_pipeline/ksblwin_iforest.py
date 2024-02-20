import numpy as np
from sklearn.ensemble import IsolationForest

from active_outlier_detection.concept_drift_detection import KSBLWIN


class KSBLWINIForest:
    def __init__(self,
                 outlier_threshold=0.75,
                 n_estimators=100,
                 contamination="auto",
                 window_size=50,
                 alpha=0.01):

        self.ksblwin = KSBLWIN(window_size, alpha)
        self.iforest = IsolationForest(contamination=contamination, n_estimators=n_estimators)

        self.outlier_threshold = outlier_threshold
        self.is_model_trained = False

    def run_pipe(self, x) -> np.ndarray | None:
        """
        Runs the pipeline of concept drift detection and
        isolation forest scoring.

        :param x: element to be added to the window
        :return: scores if window is full, None otherwise
        """
        has_drift = self.ksblwin.detect_drift(x)

        if not self.is_model_trained and not self.ksblwin.warm:
            # If model is not trained and reference window is not full, do not score
            return None, None

        if not self.is_model_trained and self.ksblwin.warm:
            # If model is not trained and reference window is full, train the model
            ref_win = self.ksblwin.reference_window

            ref_win = np.reshape(ref_win, (-1, 1))
            self.iforest.fit(ref_win)
            self.is_model_trained = True

            scores = np.abs(self.iforest.score_samples(ref_win))
            labels = scores >= self.outlier_threshold
            return scores, labels

        if not has_drift and not self.ksblwin.is_window_full():
            # If window is not full, do not score
            return None, None

        if has_drift:
            # If drift is detected, retrain the model with the new reference window
            ref_win = self.ksblwin.reference_window

            ref_win = np.reshape(ref_win, (-1, 1))
            self.iforest.fit(ref_win)

        # If window is full, score the window
        window = self.ksblwin.current_window
        window = np.reshape(window, (-1, 1))

        scores = np.abs(self.iforest.score_samples(window))
        labels = scores >= self.outlier_threshold

        return scores, labels
