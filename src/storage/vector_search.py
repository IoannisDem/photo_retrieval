from __future__ import annotations

from typing import Sequence

from storage.postgres_client_wrapper import PostgresClientWrapper
from queries import SEARCH_FACES_BY_EMBEDDING, SEARCH_IMAGES_BY_TEXT


def _search_images_by_text(
    db: PostgresClientWrapper, query_embedding: Sequence[float], top_k: int = 50
) -> list[dict]:
    return db.fetch_all(
        SEARCH_IMAGES_BY_TEXT,
        {"query_embedding": list(query_embedding), "top_k": top_k},
    )


def _search_faces_by_embedding(
    db: PostgresClientWrapper, query_embedding: Sequence[float], top_k: int = 50
) -> list[dict]:
    return db.fetch_all(
        SEARCH_FACES_BY_EMBEDDING,
        {"query_embedding": list(query_embedding), "top_k": top_k},
    )


def _reciprocal_rank_fusion(
    ranked_lists: list[list[dict]], key: str = "image_id", k: int = 60
) -> list[tuple[str, float]]:
    """
    score(d) = sum over lists L containing d of  1 / (k + rank_L(d))
    typically k is set to 60
    https://medium.com/@devalshah1619/mathematical-intuition-behind-reciprocal-rank-fusion-rrf-explained-in-2-mins-002df0cc5e2a
    """
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            item_id = item[key]
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


def find_best_images(
    db: PostgresClientWrapper,
    text_embedding: Sequence[float],
    face_embedding: Sequence[float],
    top_k: int = 10,
    candidate_pool_size: int = 50,
) -> list[tuple[str, float]]:
    image_results = _search_images_by_text(
        db, text_embedding, top_k=candidate_pool_size
    )
    face_results = _search_faces_by_embedding(
        db, face_embedding, top_k=candidate_pool_size
    )

    fused = _reciprocal_rank_fusion([image_results, face_results], key="image_id")
    return fused[:top_k]
