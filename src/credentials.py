from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class AWSCredentials:
    access_key: str
    secret_key: str
    role_arns: str
    region: str = "us-east-1"


@dataclasses.dataclass
class PostgresCredentials:
    host: str
    port: int
    database: str
    user: str
    password: str
