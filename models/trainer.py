from pyod.models.iforest import IForest
from pyod.models.lof import LOF


class Trainer:
    def __init__(self, df):
        self._available_models = {
            "iforest": IForest,
            "lof": LOF
        }
        self._df = df

    def fit(self, name, **kwargs):
        if name not in self._available_models:
            raise ValueError(f"Model {name} not available. Available models: {self._available_models.keys()}")

        model = self._available_models[name](**kwargs)

        model.fit(self._df)

        return model, model.labels_
