from __future__ import annotations

from pathlib import Path

import onnxruntime as ort
import torch
from transformers import CLIPModel, CLIPProcessor
from ultralytics import YOLO

from load_prefs import Preferences
from myml import model_predictor_base
from myml.predictor_models import clip_predictors, face_embedding, yolo_face_detection
from pipeline.components import PipelineComponents
from pipeline.retrieval import (
    QueryEncoder,
    RetrievalPipeline,
    VectorRetriever,
)
from storage import vector_search
from storage.postgres_client_wrapper import (
    PostgresClientWrapper,
    build_pg_connection,
)
from storage.s3_client_wrapper import AWSS3ClientWrapper, build_s3_client


def _resolve_device(configured_device: str) -> str:
    if configured_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return configured_device


def _require_model_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Model weights not found: {path}")


def build_pipeline(preferences: Preferences) -> PipelineComponents:
    """Build all model and storage dependencies without performing any actions."""
    model_settings = preferences.local_models
    device = _resolve_device(model_settings.device)

    _require_model_file(model_settings.face_detection_weights)
    _require_model_file(model_settings.face_embedding_weights)

    yolo_model = YOLO(str(model_settings.face_detection_weights))
    yolo_model.to(device)
    face_detection_pipeline = (
        model_predictor_base.PipelineBuilder()
        .with_model(yolo_face_detection.ModelFace(yolo_model))
        .build()
    )

    face_session = ort.InferenceSession(str(model_settings.face_embedding_weights))
    face_embedding_pipeline = (
        model_predictor_base.PipelineBuilder()
        .with_model(face_embedding.ModelFaceEmbedding(face_session))
        .with_processor(face_embedding.ProcessorFaceEmbedding())
        .build()
    )

    clip_model = CLIPModel.from_pretrained(model_settings.clip_model_name)
    clip_processor = CLIPProcessor.from_pretrained(model_settings.clip_model_name)
    clip_model.to(device)
    clip_model.eval()
    clip_pipeline = (
        model_predictor_base.PipelineBuilder()
        .with_model(clip_predictors.ModelCLIP(clip_model))
        .with_processor(
            clip_predictors.ProcessorCLIP(
                processor=clip_processor,
                device=device,
            )
        )
        .build()
    )

    database = PostgresClientWrapper(build_pg_connection(preferences.postgres))
    object_storage = AWSS3ClientWrapper(
        build_s3_client(preferences.aws),
        bucket=preferences.s3.bucket,
        prefix=preferences.s3.prefix,
    )
    retrieval_pipeline = RetrievalPipeline(
        query_encoder=QueryEncoder(
            clip=clip_pipeline,
            face_detector=face_detection_pipeline,
            face_embedder=face_embedding_pipeline,
        ),
        vector_retriever=VectorRetriever(vector_search.VectorSearch(database)),
    )

    return PipelineComponents(
        preferences=preferences,
        database=database,
        object_storage=object_storage,
        face_detection=face_detection_pipeline,
        face_embedding=face_embedding_pipeline,
        clip=clip_pipeline,
        retrieval=retrieval_pipeline,
    )
