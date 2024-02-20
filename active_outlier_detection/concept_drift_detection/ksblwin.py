import numpy as np
from scipy.stats import ks_2samp

from .base_block_drift_detector import BaseBlockDriftDetector


class KSBLWIN(BaseBlockDriftDetector):
    def __init__(self, window_size=50, alpha=0.01):
        super().__init__(window_size)
        self.alpha = alpha

    def detect_drift(self, x) -> bool:
        self.drift_detected = False
        ref_length = self._reference_window.shape[0]
        if ref_length < self.window_size:
            self._reference_window = np.concatenate([self._reference_window, [x]])
            return self.drift_detected

        if not self.warm:
            self._warm = True

        self._current_window = np.concatenate([self._current_window, [x]])

        curr_length = self._current_window.shape[0]
        if curr_length > self.window_size:
            self._current_window = np.array([x])

        if self.is_window_full():
            self._detect_drift()

        return self.drift_detected

    def _detect_drift(self):
        stat, p_value = ks_2samp(self.reference_window, self.current_window)

        if p_value < self.alpha and stat > 0.1:
            self.drift_detected = True
            self._reference_window = self.current_window
        else:
            self.drift_detected = False
