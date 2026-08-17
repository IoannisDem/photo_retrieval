from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

import credentials

logger = logging.getLogger(__name__)


def build_pg_connection(pg_cred: credentials.PostgresCredentials) -> psycopg.Connection:
    conn = psycopg.connect(
        host=pg_cred.host,
        port=pg_cred.port,
        dbname=pg_cred.database,
        user=pg_cred.user,
        password=pg_cred.password,
        autocommit=True,
    )
    register_vector(conn)
    return conn


class PostgresClientWrapper:
    def __init__(self, conn: psycopg.Connection):
        self._conn = conn

    def fetch_all(
        self, query: str, params: Mapping[str, Any] | Sequence[Any] | None = None
    ) -> list[dict[str, Any]]:
        try:
            with self._conn.cursor(row_factory=dict_row) as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
            logger.debug("fetch_all -> %d rows", len(rows))
            return rows
        except psycopg.Error:
            logger.error("Failed executing query: %s", query)
            raise

    def insert(
        self, query: str, params: Mapping[str, Any] | Sequence[Any] | None = None
    ) -> int:
        """Runs an INSERT statement. Returns affected row count."""
        try:
            with self._conn.cursor() as cur:
                cur.execute(query, params)
                affected = cur.rowcount
            logger.debug("insert -> %d row(s) affected", affected)
            return affected
        except psycopg.Error:
            logger.error("Failed executing query: %s", query)
            raise
