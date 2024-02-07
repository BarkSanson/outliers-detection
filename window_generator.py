import pandas as pd


class WindowGenerator:
    def __init__(self, df, window_length):
        self._df = df
        self._window_length = window_length

    def batch_windows(self):
        for i in range(0, len(self._df), self._window_length):
            yield self._df.iloc[i:i + self._window_length]
