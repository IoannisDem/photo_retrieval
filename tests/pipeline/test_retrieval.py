from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import torch
from PIL import Image

from pipeline.retrieval import (
    QueryEmbeddings,
    QueryEncoder,
    RetrievalPipeline,
    VectorRetriever,
)


def test_encodes_image_text_and_face_embeddings():
    image = Image.new("RGB", (32, 32), color="white")
    clip = Mock()
    clip.predict.return_value = SimpleNamespace(
        image_output=torch.ones(1, 3),
        text_output=torch.ones(1, 3) * 2,
    )

    face_detector = Mock()
    face_detector.predict.return_value = SimpleNamespace(
        boxes=[torch.tensor([[1.0, 1.0, 8.0, 8.0, 0.9]])]
    )

    face_embedder = Mock()
    face_embedder.predict.return_value = SimpleNamespace(
        embeddings=[torch.ones(1, 3) * 3]
    )

    result = QueryEncoder(clip, face_detector, face_embedder).encode(image, "a person")

    assert result == QueryEmbeddings(
        image=[1.0, 1.0, 1.0],
        text=[2.0, 2.0, 2.0],
        faces=[[3.0, 3.0, 3.0]],
    )


class FakeVectorSearch:
    def __init__(self, image_results, text_results, face_results):
        self.image_results = image_results
        self.text_results = text_results
        self.face_results = face_results

    def search_images_by_embedding(self, query_embedding, top_k):
        return self.image_results

    def search_images_by_text(self, query_embedding, top_k):
        return self.text_results

    def search_faces_by_embedding(self, query_embedding, top_k):
        return self.face_results


class TestVectorRetriever:
    def test_fuses_image_text_and_face_results(self):
        image_results = [
            {"image_id": "image-1", "image_uri": "s3://bucket/image-1.jpg"}
        ]
        text_results = [{"image_id": "image-2", "image_uri": "s3://bucket/image-2.jpg"}]
        face_results = [
            {
                "image_id": "image-1",
                "image_uri": "s3://bucket/image-1.jpg",
                "face_id": "face-1",
            }
        ]

        vector_search = FakeVectorSearch(image_results, text_results, face_results)

        result = VectorRetriever(vector_search).retrieve(
            QueryEmbeddings(image=[1.0], text=[2.0], faces=[[3.0]]),
            top_k=2,
            candidate_pool_size=10,
        )

        assert result == [
            {
                "image_id": "image-1",
                "image_uri": "s3://bucket/image-1.jpg",
                "score": 2 / 61,
            },
            {
                "image_id": "image-2",
                "image_uri": "s3://bucket/image-2.jpg",
                "score": 1 / 61,
            },
        ]


def test_composes_encoder_and_retriever():
    query_embeddings = QueryEmbeddings(image=[1.0], text=[2.0], faces=[])
    expected_results = [{"image_id": "image-1", "score": 1.0}]
    query_encoder = Mock()
    query_encoder.encode.return_value = query_embeddings
    vector_retriever = Mock()
    vector_retriever.retrieve.return_value = expected_results

    result = RetrievalPipeline(query_encoder, vector_retriever).retrieve(
        Image.new("RGB", (2, 2)), "a person"
    )

    assert result == expected_results
