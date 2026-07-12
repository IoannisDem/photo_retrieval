INSERT_IMAGE = """
INSERT INTO images (image_id, image_uri, clip_embedding)
VALUES (%(image_id)s, %(image_uri)s, %(clip_embedding)s)
RETURNING image_id;
"""

INSERT_FACE = """
INSERT INTO faces (face_id, image_id, face_embedding, bbox)
VALUES (%(face_id)s, %(image_id)s, %(face_embedding)s, %(bbox)s)
RETURNING face_id;
"""

SEARCH_IMAGES_BY_TEXT = """
SELECT
    image_id,
    image_uri,
    1 - (clip_embedding <=> %(query_embedding)s::vector) AS similarity
FROM images
ORDER BY clip_embedding <=> %(query_embedding)s::vector
LIMIT %(top_k)s;
"""

SEARCH_FACES_BY_EMBEDDING = """
SELECT
    image_id,
    face_id,
    1 - (face_embedding <=> %(query_embedding)s::vector) AS similarity
FROM faces
ORDER BY face_embedding <=> %(query_embedding)s::vector
LIMIT %(top_k)s;
"""
