from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


class Trainer:
    def __init__(self, df):
        self.available_models = {
            "iforest": IsolationForest,
            "lof": LocalOutlierFactor
        }
        self.df = df
        self.trained_models = {}

    def fit(self, name, **kwargs):
        if name not in self.available_models:
            raise ValueError(f"Model {name} not available. Available models: {self.available_models.keys()}")

        if name in self.trained_models:
            return self.trained_models[name]

        model = self.available_models[name](**kwargs)
        self.trained_models[model] = model

        if name == "lof":
            model.fit_predict(self.df)
        else:
            model.fit(self.df)

        return model
