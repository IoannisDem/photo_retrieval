from __future__ import annotations

import io
import json
import uuid
from pathlib import Path

from PIL import Image

from myml.predictor_models import clip_predictors, face_embedding, yolo_face_detection
from pipeline.components import PipelineComponents
from queries import INSERT_FACE, INSERT_IMAGE


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def ingest_folder(
    folder: str | Path,
    pipeline: PipelineComponents,
) -> list[str]:
    """Ingest supported images from a folder and return inserted image IDs."""
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Image folder not found: {folder}")

    inserted_image_ids: list[str] = []
    for image_path in sorted(folder.rglob("*")):
        if (
            image_path.is_file()
            and image_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ):
            image_id = ingest_image(image_path, folder, pipeline)
            if image_id is not None:
                inserted_image_ids.append(image_id)

    return inserted_image_ids


def ingest_image(
    image_path: Path,
    root_folder: Path,
    pipeline: PipelineComponents,
) -> str | None:
    image_bytes = image_path.read_bytes()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_key = image_path.relative_to(root_folder).as_posix()
    image_uri = _build_image_uri(pipeline, image_key)

    clip_result = pipeline.clip.predict(clip_predictors.CLIPRawData(images=[image]))
    clip_embedding = clip_result.image_output[0].detach().cpu().tolist()

    face_result = pipeline.face_detection.predict(
        yolo_face_detection.FaceRawData(images=[image])
    )
    face_boxes = face_result.boxes[0]
    face_crops = face_embedding.crop_faces([image], [face_boxes])
    face_result_embeddings = pipeline.face_embedding.predict(
        face_embedding.FaceEmbeddingRawData(images=face_crops)
    )

    pipeline.object_storage.write(image_key, image_bytes)

    image_id = str(uuid.uuid4())
    inserted = pipeline.database.fetch_all(
        INSERT_IMAGE,
        {
            "image_id": image_id,
            "image_uri": image_uri,
            "clip_embedding": clip_embedding,
        },
    )
    if not inserted:
        return None

    for box, embedding in zip(
        face_boxes.tolist(), face_result_embeddings.embeddings[0]
    ):
        pipeline.database.insert(
            INSERT_FACE,
            {
                "face_id": str(uuid.uuid4()),
                "image_id": image_id,
                "face_embedding": embedding.detach().cpu().tolist(),
                "bbox": json.dumps(box),
            },
        )

    return image_id


def _build_image_uri(pipeline: PipelineComponents, image_key: str) -> str:
    prefix = pipeline.preferences.s3.prefix.strip("/")
    full_key = f"{prefix}/{image_key}" if prefix else image_key
    return f"s3://{pipeline.preferences.s3.bucket}/{full_key}"
