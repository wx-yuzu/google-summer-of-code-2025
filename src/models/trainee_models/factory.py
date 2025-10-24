from pathlib import Path

from ultralytics import YOLO

from models.trainee_models.base import TraineeModel

TRAINEE_REGISTRY = {}


def register_trainee(name):
    def decorator(cls):
        TRAINEE_REGISTRY[name] = cls
        return cls

    return decorator


def create_trainee(model_name: str, **kwargs) -> TraineeModel:
    return TRAINEE_REGISTRY[model_name](**kwargs)


@register_trainee("yolo11")
class Yolo11nTraineeModel(TraineeModel):
    def __init__(self, weights_path: Path):
        self.model = YOLO(str(weights_path))
        self.input_image_size = (640, 640)

    def train(self, **kwargs):
        self.model.train(**kwargs)
