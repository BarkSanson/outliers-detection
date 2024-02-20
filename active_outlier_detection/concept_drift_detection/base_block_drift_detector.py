from abc import ABC, abstractmethod

import numpy as np


class BaseBlockDriftDetector(ABC):
    def __init__(self, window_size=50):
        self._reference_window = np.array([])
        self._current_window = np.array([])
        self._warm = False  # Warm = True when the reference window is full

        self.window_size = window_size
        self.drift_detected = False

    @property
    def reference_window(self) -> np.array:
        return self._reference_window

    @property
    def current_window(self) -> np.array:
        return self._current_window

    @property
    def warm(self) -> bool:
        return self._warm

    @abstractmethod
    def detect_drift(self, x):
        pass

    def is_window_full(self):
        return len(self.current_window) == self.window_size
