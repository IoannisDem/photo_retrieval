from __future__ import annotations

import abc
from typing import TypeVar, Generic, Any

RawInput = TypeVar("RawInput")
TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


class PipelineMissingModel(Exception):
    pass


class Processor(abc.ABC, Generic[RawInput, TInput]):
    @abc.abstractmethod
    def process(self, data: RawInput) -> TInput:
        pass


class SimpleProcessor(Processor[Any, Any]):
    def process(self, data: Any) -> Any:
        return data


class Model(abc.ABC, Generic[TInput, TOutput]):
    @abc.abstractmethod
    def predict(self, data: TInput) -> TOutput:
        pass


class DefaultPipeline(Generic[RawInput, TOutput]):
    def __init__(
        self,
        model: Model[TInput, TOutput],
        processor: Processor[RawInput, TInput],
    ) -> None:
        self._model = model
        self._processor = processor

    def predict(self, data: RawInput) -> TOutput:
        return self._model.predict(self._processor.process(data))


class PipelineBuilder(Generic[RawInput, TOutput]):
    def __init__(self) -> None:
        self._model: Model | None = None
        self._processor: Processor | None = None

    def with_model(self, model: Model) -> PipelineBuilder[RawInput, TOutput]:
        self._model = model
        return self

    def with_processor(
        self, processor: Processor
    ) -> PipelineBuilder[RawInput, TOutput]:
        self._processor = processor
        return self

    @property
    def model(self) -> Model:
        if self._model is None:
            raise PipelineMissingModel(
                "A model must be set before building the pipeline"
            )
        return self._model

    @property
    def processor(self) -> Processor:
        if self._processor is None:
            self._processor = SimpleProcessor()
        return self._processor

    def build(self) -> DefaultPipeline[RawInput, TOutput]:
        return DefaultPipeline(model=self.model, processor=self.processor)
