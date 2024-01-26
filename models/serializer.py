import joblib


class Serializer(object):
    def __init__(self, model_path):
        self.model_path = model_path

    def load_model(self):
        return joblib.load(self.model_path)

    def save_model(self, model):
        joblib.dump(model, self.model_path)
