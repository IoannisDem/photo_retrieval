import abc
from src.models import loader


class BaseModelLoader(abc.ABC):
    @abc.abstractmethod
    def load(self, config: loader.ModelConfig):
        pass
