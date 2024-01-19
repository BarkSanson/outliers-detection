from sklearn.model_selection import GridSearchCV


class Tuner:
    def __init__(self, model, df, **kwargs):
        self.model = model
        self.df = df
        self.kwargs = kwargs

        self.tuner = GridSearchCV(self.model, self.kwargs, n_jobs=-1)

    def tune(self):
        self.tuner.fit(self.df)

        return self.tuner.best_params_, self.tuner.best_estimator_
