CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE images (
    image_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    image_uri       TEXT NOT NULL UNIQUE,
    clip_embedding  vector(512),                
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE faces (
    face_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    image_id        UUID NOT NULL REFERENCES images(image_id) ON DELETE CASCADE,
    face_embedding  vector(512),                  
    bbox            JSONB,                      
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_faces_image_id ON faces (image_id);


CREATE INDEX idx_images_clip_embedding
    ON images USING hnsw (clip_embedding vector_cosine_ops);

CREATE INDEX idx_faces_face_embedding
    ON faces USING hnsw (face_embedding vector_cosine_ops);