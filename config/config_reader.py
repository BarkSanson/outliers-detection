import json

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    name: str
    params: dict[str, any]


class ConfigReader:
    def __init__(self, path):
        self.path = path

    def read(self) -> list[ModelConfig]:
        models = []
        with open(self.path) as f:
            models_json = json.load(f)

            for model in models_json:
                models.append(ModelConfig(model['name'], model['params']))

        return models
