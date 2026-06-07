from __future__ import annotations

import dataclasses

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from transformers.modeling_outputs import BaseModelOutputWithPooling

from src.models import model_predictor_base


@dataclasses.dataclass
class CLIPRawData:
    images: list[Image.Image] | None = None
    texts: list[str] | None = None


@dataclasses.dataclass
class CLIPInput:
    images: dict[str, torch.Tensor] | None = None
    texts: dict[str, torch.Tensor] | None = None


@dataclasses.dataclass
class CLIPOutput:
    image_output: BaseModelOutputWithPooling | None = None
    text_output: BaseModelOutputWithPooling | None = None


class ProcessorCLIP(model_predictor_base.Processor[CLIPRawData, CLIPInput]):
    def __init__(self, processor: CLIPProcessor, device: str = "cpu") -> None:
        self._processor = processor
        self._device = device

    def process(self, data: CLIPRawData) -> CLIPInput:
        images_inputs = (
            self._process_images(data.images) if data.images is not None else None
        )
        texts_inputs = (
            self._process_texts(data.texts) if data.texts is not None else None
        )
        return CLIPInput(images=images_inputs, texts=texts_inputs)

    def _process_images(self, images: list[Image.Image]) -> dict[str, torch.Tensor]:
        image_inputs = self._processor(images=images, return_tensors="pt")
        return {k: v.to(self._device) for k, v in image_inputs.items()}

    def _process_texts(self, texts: list[str]) -> dict[str, torch.Tensor]:
        text_inputs = self._processor(text=texts, return_tensors="pt", padding=True)
        return {k: v.to(self._device) for k, v in text_inputs.items()}


class ModelCLIP(model_predictor_base.Model[CLIPInput, CLIPOutput]):
    def __init__(self, model: CLIPModel) -> None:
        self._model = model

    def predict(self, data: CLIPInput) -> CLIPOutput:
        image_output = (
            self._image_embeddings(data.images) if data.images is not None else None
        )
        text_output = (
            self._text_embeddings(data.texts) if data.texts is not None else None
        )
        return CLIPOutput(image_output=image_output, text_output=text_output)

    def _image_embeddings(self, image_data: dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            output = self._model.vision_model(pixel_values=image_data["pixel_values"])
            return self._model.visual_projection(output.pooler_output)

    def _text_embeddings(self, text_data: dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            output = self._model.text_model(
                input_ids=text_data["input_ids"],
                attention_mask=text_data["attention_mask"],
            )
            return self._model.text_projection(output.pooler_output)
