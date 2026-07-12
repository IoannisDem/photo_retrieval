import pytest
from pytest_postgresql import factories

from storage.postgres_client_wrapper import PostgresClientWrapper

postgresql_proc = factories.postgresql_proc()
postgresql = factories.postgresql("postgresql_proc")


class TestPostgresClientWrapper:
    @pytest.fixture
    def conn(self, postgresql):
        return postgresql

    @pytest.fixture
    def wrapper(self, conn):
        return PostgresClientWrapper(conn)

    @pytest.fixture(autouse=True)
    def _create_table(self, conn):
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE users (
                    id INT PRIMARY KEY,
                    email TEXT NOT NULL
                )
                """
            )
        conn.commit()

    def test_insert_adds_row(self, wrapper, conn):
        affected = wrapper.insert(
            "INSERT INTO users (id, email) VALUES (%(id)s, %(email)s)",
            {"id": 1, "email": "a@b.com"},
        )
        conn.commit()

        assert affected == 1

        with conn.cursor() as cur:
            cur.execute("SELECT id, email FROM users WHERE id = 1")
            row = cur.fetchone()
        assert row == (1, "a@b.com")

    def test_fetch_all_returns_inserted_rows(self, wrapper, conn):
        wrapper.insert(
            "INSERT INTO users (id, email) VALUES (%(id)s, %(email)s)",
            {"id": 1, "email": "a@b.com"},
        )
        wrapper.insert(
            "INSERT INTO users (id, email) VALUES (%(id)s, %(email)s)",
            {"id": 2, "email": "c@d.com"},
        )
        conn.commit()

        rows = wrapper.fetch_all("SELECT * FROM users ORDER BY id")

        assert rows == [
            {"id": 1, "email": "a@b.com"},
            {"id": 2, "email": "c@d.com"},
        ]

    def test_fetch_all_returns_empty_list_when_no_match(self, wrapper):
        rows = wrapper.fetch_all("SELECT * FROM users WHERE id = 999")
        assert rows == []

    def test_insert_duplicate_primary_key_raises(self, wrapper, conn):
        wrapper.insert(
            "INSERT INTO users (id, email) VALUES (%(id)s, %(email)s)",
            {"id": 1, "email": "a@b.com"},
        )
        conn.commit()

        with pytest.raises(Exception):  # psycopg.errors.UniqueViolation
            wrapper.insert(
                "INSERT INTO users (id, email) VALUES (%(id)s, %(email)s)",
                {"id": 1, "email": "dup@b.com"},
            )
        conn.rollback()
