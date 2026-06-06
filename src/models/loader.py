import abc


class BaseModelLoader(abc.ABC):
    @abc.abstractmethod
    def load(self):
        pass
