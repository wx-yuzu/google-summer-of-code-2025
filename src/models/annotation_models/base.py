from abc import ABC, abstractmethod
from typing import Dict


class AnnotationModel(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def annotate(self, image_path: str) -> Dict:
        pass


class OpenVocabularyAnnotationModel(ABC):
    def __init__(self, ontology: Dict[str, str]):
        self.ontology = ontology

    @abstractmethod
    def annotate(self, image_path: str) -> Dict:
        pass
