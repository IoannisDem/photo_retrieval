from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from PIL import Image

from credentials import AWSCredentials, PostgresCredentials
from load_prefs import LocalModelPreferences, Preferences, S3Preferences
from pipeline.components import PipelineComponents
from pipeline import ingestion


class FakeDatabase:
    def __init__(self, image_is_new: bool = True):
        self.image_is_new = image_is_new
        self.face_inserts: list[dict] = []

    def fetch_all(self, query, params):
        if self.image_is_new:
            return [{"image_id": params["image_id"]}]
        return []

    def insert(self, query, params):
        self.face_inserts.append(params)
        return 1


class FakeObjectStorage:
    def __init__(self):
        self.writes: list[tuple[str, bytes]] = []

    def write(self, key: str, data: bytes):
        self.writes.append((key, data))


def make_pipeline(image_is_new: bool = True) -> PipelineComponents:
    face_boxes = torch.tensor(
        [
            [1.0, 2.0, 8.0, 9.0, 0.95],
            [10.0, 11.0, 18.0, 19.0, 0.90],
        ]
    )
    face_detector = Mock()
    face_detector.predict.return_value = SimpleNamespace(boxes=[face_boxes])

    face_model = Mock()
    face_model.predict.return_value = SimpleNamespace(embeddings=[torch.ones(2, 512)])

    clip_model = Mock()
    clip_model.predict.return_value = SimpleNamespace(image_output=torch.ones(1, 512))

    preferences = Preferences(
        aws=AWSCredentials(
            access_key="dummy",
            secret_key="dummy",
            role_arns="dummy",
        ),
        postgres=PostgresCredentials(
            host="localhost",
            port=5432,
            database="postgres",
            user="postgres",
            password="dummy",
        ),
        s3=S3Preferences(bucket="bucket", prefix="images"),
        local_models=LocalModelPreferences(
            face_detection_weights=Path("face.pt"),
            face_embedding_weights=Path("face.onnx"),
            clip_model_name="clip",
            device="cpu",
        ),
    )
    return PipelineComponents(
        preferences=preferences,
        database=FakeDatabase(image_is_new),
        object_storage=FakeObjectStorage(),
        face_detection=face_detector,
        face_embedding=face_model,
        clip=clip_model,
    )


def create_image(path: Path) -> bytes:
    image = Image.new("RGB", (32, 32), color="white")
    image.save(path, format="PNG")
    return path.read_bytes()


def test_rejects_missing_folder(tmp_path):
    with pytest.raises(NotADirectoryError):
        ingestion.ingest_folder(tmp_path / "missing", make_pipeline())


class TestIngestImage:
    def test_returns_none_for_duplicate_image_uri(self, tmp_path):
        image_path = tmp_path / "photo.png"
        create_image(image_path)

        result = ingestion.ingest_image(image_path, tmp_path, make_pipeline(False))

        assert result is None

    def test_inserts_one_face_record_per_detection(self, tmp_path):
        image_path = tmp_path / "photo.png"
        create_image(image_path)
        pipeline = make_pipeline()

        ingestion.ingest_image(image_path, tmp_path, pipeline)

        assert len(pipeline.database.face_inserts) == 2
        assert len(pipeline.object_storage.writes) == 1
