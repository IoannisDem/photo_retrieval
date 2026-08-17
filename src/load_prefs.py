from __future__ import annotations

import dataclasses
from pathlib import Path

from credentials import AWSCredentials, PostgresCredentials
from myml.utils import load_yaml


@dataclasses.dataclass(frozen=True)
class S3Preferences:
    bucket: str
    prefix: str


@dataclasses.dataclass(frozen=True)
class LocalModelPreferences:
    face_detection_weights: Path
    face_embedding_weights: Path
    clip_model_name: str
    device: str


@dataclasses.dataclass(frozen=True)
class Preferences:
    aws: AWSCredentials
    postgres: PostgresCredentials
    s3: S3Preferences
    local_models: LocalModelPreferences


def load_preferences(path: str | Path) -> Preferences:
    """Load a YAML configuration file into typed preferences."""
    path = Path(path)
    raw_preferences = load_yaml(path)
    model_preferences = raw_preferences["local-models"]

    return Preferences(
        aws=AWSCredentials(**raw_preferences["aws"]),
        postgres=PostgresCredentials(**raw_preferences["postgres"]),
        s3=S3Preferences(**raw_preferences["s3"]),
        local_models=LocalModelPreferences(
            face_detection_weights=path.parent
            / model_preferences["face_detection_weights"],
            face_embedding_weights=path.parent
            / model_preferences["face_embedding_weights"],
            clip_model_name=model_preferences["clip_model_name"],
            device=model_preferences["device"],
        ),
    )
