from abc import ABC, abstractmethod


class TraineeModel(ABC):
    @abstractmethod
    def train(self) -> None:
        pass
