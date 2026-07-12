from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence
from unittest import mock

import pytest

from queries import SEARCH_FACES_BY_EMBEDDING, SEARCH_IMAGES_BY_TEXT
from storage import vector_search


def make_mock_db(image_results, face_results):
    db = mock.Mock()

    def fetch_all_side_effect(query, params=None):
        if query == SEARCH_IMAGES_BY_TEXT:
            return image_results
        if query == SEARCH_FACES_BY_EMBEDDING:
            return face_results

    db.fetch_all = mock.Mock(side_effect=fetch_all_side_effect)
    return db


@dataclass(frozen=True)
class TestCase:
    image_results: list[dict[str, Any]]
    face_results: list[dict[str, Any]]
    text_embedding: Sequence[float]
    face_embedding: Sequence[float]
    expected: list[tuple[str, float]]
    top_k: int = 10
    candidate_pool_size: int = 50


class TestFindBestImages:
    TESTCASES = [
        pytest.param(
            TestCase(
                image_results=[
                    {"image_id": "A"},
                    {"image_id": "B"},
                    {"image_id": "C"},
                ],
                face_results=[
                    {"image_id": "B"},
                    {"image_id": "D"},
                    {"image_id": "A"},
                ],
                text_embedding=[0.1, 0.2],
                face_embedding=[0.3, 0.4],
                expected=[
                    ("B", 1 / 62 + 1 / 61),
                    ("A", 1 / 61 + 1 / 63),
                    ("D", 1 / 62),
                    ("C", 1 / 63),
                ],
            ),
            id="fuses_rankings",
        ),
        pytest.param(
            TestCase(
                image_results=[],
                face_results=[],
                text_embedding=[0.1],
                face_embedding=[0.2],
                expected=[],
            ),
            id="empty_results",
        ),
        pytest.param(
            TestCase(
                image_results=[
                    {"image_id": "A"},
                ],
                face_results=[
                    {"image_id": "B"},
                ],
                text_embedding=[0.1],
                face_embedding=[0.2],
                expected=[
                    ("A", 1 / 61),
                    ("B", 1 / 61),
                ],
            ),
            id="no_overlap",
        ),
        pytest.param(
            TestCase(
                image_results=[{"image_id": str(i)} for i in range(5)],
                face_results=[{"image_id": str(i)} for i in range(5)],
                text_embedding=[0.1],
                face_embedding=[0.2],
                expected=[
                    ("0", 2 / 61),
                    ("1", 2 / 62),
                ],
                top_k=2,
            ),
            id="respects_top_k",
        ),
    ]

    @pytest.fixture(params=TESTCASES)
    def testcase(self, request):
        return request.param

    @pytest.fixture
    def db(self, testcase):
        return make_mock_db(
            testcase.image_results,
            testcase.face_results,
        )

    @pytest.fixture
    def expected(self, testcase):
        return testcase.expected

    def test_find_best_images(self, db, expected, testcase):
        observed = vector_search.find_best_images(
            db=db,
            text_embedding=testcase.text_embedding,
            face_embedding=testcase.face_embedding,
            top_k=testcase.top_k,
            candidate_pool_size=testcase.candidate_pool_size,
        )
        assert observed == expected
