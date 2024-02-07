# from pyod.models.iforest import IForest
# from pyod.models.lof import LOF
# from pyod.models.kde import KDE
import pandas as pd

from .batched_iforest import BatchedIForest


class BatchedTrainer:
    def __init__(self, df, window_generator):
        self.window_generator = window_generator
        self._available_models = {
            "iforest": BatchedIForest,
            # "lof": LOF,
            # "kde": KDE,
        }
        self._df = df

    def fit(self, name, **kwargs):
        if name not in self._available_models:
            raise ValueError(f"Model {name} not available. Available models: {self._available_models.keys()}")

        model = self._available_models[name](**kwargs)

        labels = pd.DataFrame()

        for window in self.window_generator.batch_windows():
            model.fit_predict(window)
            labels = pd.concat([labels, pd.DataFrame(model.fit_predict(window))])

        return model, labels
