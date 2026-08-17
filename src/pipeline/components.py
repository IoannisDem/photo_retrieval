from __future__ import annotations

import dataclasses
from typing import Any

from load_prefs import Preferences
from storage.postgres_client_wrapper import PostgresClientWrapper
from storage.s3_client_wrapper import AWSS3ClientWrapper


@dataclasses.dataclass
class PipelineComponents:
    preferences: Preferences
    database: PostgresClientWrapper
    object_storage: AWSS3ClientWrapper
    face_detection: Any
    face_embedding: Any
    clip: Any
    retrieval: Any | None = None
