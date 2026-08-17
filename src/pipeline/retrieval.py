from __future__ import annotations

import dataclasses
from typing import Any

from PIL import Image

from myml.predictor_models import clip_predictors, face_embedding, yolo_face_detection
from storage import vector_search


@dataclasses.dataclass(frozen=True)
class QueryEmbeddings:
    image: list[float]
    text: list[float]
    faces: list[list[float]]


class QueryEncoder:
    """Converts a query image and prompt into all retrieval embeddings."""

    def __init__(self, clip, face_detector, face_embedder) -> None:
        self._clip = clip
        self._face_detector = face_detector
        self._face_embedder = face_embedder

    def encode(self, query_image: Image.Image, prompt: str) -> QueryEmbeddings:
        image = query_image.convert("RGB")
        clip_result = self._clip.predict(
            clip_predictors.CLIPRawData(images=[image], texts=[prompt])
        )

        face_result = self._face_detector.predict(
            yolo_face_detection.FaceRawData(images=[image])
        )
        face_boxes = face_result.boxes[0]
        face_crops = face_embedding.crop_faces([image], [face_boxes])
        face_result_embeddings = self._face_embedder.predict(
            face_embedding.FaceEmbeddingRawData(images=face_crops)
        )

        return QueryEmbeddings(
            image=clip_result.image_output[0].detach().cpu().tolist(),
            text=clip_result.text_output[0].detach().cpu().tolist(),
            faces=[
                embedding.detach().cpu().tolist()
                for embedding in face_result_embeddings.embeddings[0]
            ],
        )


class VectorRetriever:
    """Searches and ranks stored vectors using already-encoded queries."""

    def __init__(self, search_backend: vector_search.VectorSearch) -> None:
        self._search_backend = search_backend

    def retrieve(
        self,
        embeddings: QueryEmbeddings,
        top_k: int = 5,
        candidate_pool_size: int = 50,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if candidate_pool_size <= 0:
            raise ValueError("candidate_pool_size must be positive")

        ranked_lists = [
            self._search_backend.search_images_by_embedding(
                embeddings.image, candidate_pool_size
            ),
            self._search_backend.search_images_by_text(
                embeddings.text, candidate_pool_size
            ),
        ]
        ranked_lists.extend(
            self._search_backend.search_faces_by_embedding(
                face_embedding, candidate_pool_size
            )
            for face_embedding in embeddings.faces
        )

        metadata = {
            result["image_id"]: result.get("image_uri")
            for ranked_list in ranked_lists
            for result in ranked_list
            if result.get("image_uri") is not None
        }
        fused_results = vector_search.reciprocal_rank_fusion(
            ranked_lists, key="image_id"
        )

        return [
            {
                "image_id": image_id,
                "image_uri": metadata.get(image_id),
                "score": score,
            }
            for image_id, score in fused_results[:top_k]
        ]


class RetrievalPipeline:
    """Coordinates query encoding and vector retrieval."""

    def __init__(
        self,
        query_encoder: QueryEncoder,
        vector_retriever: VectorRetriever,
    ) -> None:
        self._query_encoder = query_encoder
        self._vector_retriever = vector_retriever

    def retrieve(
        self,
        query_image: Image.Image,
        prompt: str,
        top_k: int = 5,
        candidate_pool_size: int = 50,
    ) -> list[dict[str, Any]]:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        embeddings = self._query_encoder.encode(query_image, prompt)
        return self._vector_retriever.retrieve(
            embeddings,
            top_k=top_k,
            candidate_pool_size=candidate_pool_size,
        )
