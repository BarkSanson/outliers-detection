from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


class Trainer:
    def __init__(self, df):
        self.available_models = {
            "iforest": IsolationForest,
            "lof": LocalOutlierFactor
        }
        self.df = df

    def fit_models(self, **kwargs):
        models = {}
        for name, params in kwargs.items():
            model, labels = self._fit(name, **params)
            models[name] = (model, labels)

        return models

    def _fit(self, name, **kwargs):
        if name not in self.available_models:
            raise ValueError(f"Model {name} not available. Available models: {self.available_models.keys()}")

        model = self.available_models[name](**kwargs)

        labels = model.fit_predict(self.df)

        return model, labels


