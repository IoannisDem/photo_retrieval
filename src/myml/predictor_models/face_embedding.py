from __future__ import annotations

import dataclasses

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from pydantic import BaseModel

from myml import model_predictor_base, utils


@dataclasses.dataclass
class FaceEmbeddingRawData:
    images: list[list[Image.Image]]


class FaceEmbeddingRequest(BaseModel):
    images: list[list[str]]


class EmbeddingResultOutput(BaseModel):
    embeddings: list[list[list[float]]]


@dataclasses.dataclass
class FaceEmbeddingInput:
    images: list[np.ndarray]


@dataclasses.dataclass
class FaceEmbeddingOutput:
    embeddings: list[torch.Tensor]


def face_embedding_request_conversion(
    raw_data: FaceEmbeddingRawData,
) -> FaceEmbeddingRequest:
    image_data = [
        [utils.encode_image(face) for face in faces] for faces in raw_data.images
    ]
    return FaceEmbeddingRequest(images=image_data)


def to_face_embedding_output(result: EmbeddingResultOutput) -> FaceEmbeddingOutput:
    embeddings = [
        torch.tensor(image_embeddings) for image_embeddings in result.embeddings
    ]
    return FaceEmbeddingOutput(embeddings=embeddings)


def crop_faces(
    images: list[Image.Image], boxes_per_image: list[torch.Tensor]
) -> list[list[Image.Image]]:
    return [
        [image.crop((x1, y1, x2, y2)) for x1, y1, x2, y2, _ in boxes.tolist()]
        for image, boxes in zip(images, boxes_per_image)
    ]


class ProcessorFaceEmbedding(
    model_predictor_base.Processor[FaceEmbeddingRawData, FaceEmbeddingInput]
):
    def __init__(self):
        self.INPUT_SIZE = (112, 112)
        self.MEAN = 127.5
        self.STD = 128.0

    def process(self, data: FaceEmbeddingRawData) -> FaceEmbeddingInput:
        images = [self._process_faces(faces) for faces in data.images]
        return FaceEmbeddingInput(images=images)

    def _process_faces(self, faces: list[Image.Image]) -> np.ndarray:
        if not faces:
            return np.empty((0, *self.INPUT_SIZE, 3), dtype=np.float32)

        blobs = [self._preprocess_face(face) for face in faces]
        return np.stack(blobs)

    def _preprocess_face(self, face: Image.Image) -> np.ndarray:
        img = face.convert("RGB").resize(self.INPUT_SIZE)
        arr = (np.array(img).astype(np.float32) - self.MEAN) / self.STD
        return arr


class ModelFaceEmbedding(
    model_predictor_base.Model[FaceEmbeddingInput, FaceEmbeddingOutput]
):
    def __init__(self, session: ort.InferenceSession) -> None:
        self._session = session
        self._input_name = session.get_inputs()[0].name
        self._output_name = session.get_outputs()[0].name

    def predict(self, data: FaceEmbeddingInput) -> FaceEmbeddingOutput:
        embeddings = [self._embed_batch(batch) for batch in data.images]
        return FaceEmbeddingOutput(embeddings=embeddings)

    def _embed_batch(self, batch: np.ndarray) -> torch.Tensor:
        if batch.shape[0] == 0:
            return torch.empty(0, 512)

        output = self._session.run([self._output_name], {self._input_name: batch})[0]
        return torch.tensor(output)
