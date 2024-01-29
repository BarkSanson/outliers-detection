import os
import joblib


class Serializer:
    def __init__(self, models_path):
        self.models_path = models_path

        os.makedirs(models_path, exist_ok=True)

    def load_model(self, name):
        return joblib.load(os.path.join(self.models_path, name))

    def save_model(self, model, name):
        joblib.dump(model, os.path.join(self.models_path, name))
