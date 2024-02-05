from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.kde import KDE


class Trainer:
    def __init__(self, df):
        self._available_models = {
            "iforest": IForest,
            "lof": LOF,
            "kde": KDE,
        }
        self._df = df

    def fit(self, name, **kwargs):
        if name not in self._available_models:
            raise ValueError(f"Model {name} not available. Available models: {self._available_models.keys()}")

        model = self._available_models[name](**kwargs)

        model.fit(self._df)

        return model, model.labels_, model.decision_scores_

    def fit_and_save(self, name, serializer, title, **kwargs):
        model, labels, decision_scores = self.fit(name, **kwargs)
        serializer.save_model(model, title)
        return model, labels, decision_scores
