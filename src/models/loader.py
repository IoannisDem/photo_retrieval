import abc
from src.models import loader
from pathlib import Path


class BaseModelLoader(abc.ABC):
    @abc.abstractmethod
    def load(self, config: loader.ModelConfig):
        pass
