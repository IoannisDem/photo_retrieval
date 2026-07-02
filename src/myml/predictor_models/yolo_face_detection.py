from __future__ import annotations

import dataclasses

import torch
from PIL import Image
from pydantic import BaseModel
from ultralytics import YOLO

from myml import model_predictor_base, utils


@dataclasses.dataclass
class FaceRawData:
    images: list[Image.Image]


class FaceRequest(BaseModel):
    images: list[str]


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


class FaceResultOutput(BaseModel):
    boxes: list[list[BoundingBox]]


@dataclasses.dataclass
class FaceOutput:
    boxes: list[torch.Tensor]


def face_request_conversion(face_raw_data: FaceRawData) -> FaceRequest:
    image_data = [utils.encode_image(image) for image in face_raw_data.images]
    return FaceRequest(images=image_data)


def to_face_output(result: FaceResultOutput) -> FaceOutput:
    boxes = [
        torch.tensor([[b.x1, b.y1, b.x2, b.y2, b.confidence] for b in image_boxes])
        for image_boxes in result.boxes
    ]
    return FaceOutput(boxes=boxes)


class ModelFace(model_predictor_base.Model[FaceRawData, FaceOutput]):
    def __init__(self, model: YOLO, conf: float = 0.5, iou: float = 0.45) -> None:
        self._model = model
        self._conf = conf
        self._iou = iou

    def predict(self, data: FaceRawData) -> FaceOutput:
        results = self._model.predict(
            data.images, conf=self._conf, iou=self._iou, verbose=False
        )
        boxes = [
            torch.cat([r.boxes.xyxy, r.boxes.conf.unsqueeze(1)], dim=1) for r in results
        ]
        return FaceOutput(boxes=boxes)
