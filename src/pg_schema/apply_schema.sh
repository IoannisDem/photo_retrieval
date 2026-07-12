#!/usr/bin/env bash
set -euo pipefail

PGHOST="${PGHOST:-localhost}"
PGPORT="${PGPORT:-5432}"
PGUSER="${PGUSER:-postgres}"
PGPASSWORD="${PGPASSWORD:-postgres}"
PGDATABASE="${PGDATABASE:-postgres}"
DOCKER_CONTAINER="${DOCKER_CONTAINER:-}"

SCHEMA_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/schema.sql"

if [[ ! -f "$SCHEMA_FILE" ]]; then
    echo "schema.sql not found next to this script at: $SCHEMA_FILE" >&2
    exit 1
fi

if [[ -n "$DOCKER_CONTAINER" ]]; then
    echo "Applying schema.sql to container '$DOCKER_CONTAINER' database '$PGDATABASE'..."
    docker exec -i \
        -e PGPASSWORD="$PGPASSWORD" \
        "$DOCKER_CONTAINER" \
        psql -U "$PGUSER" -d "$PGDATABASE" -v ON_ERROR_STOP=1 < "$SCHEMA_FILE"
else
    # Postgres port is exposed to the host; connect directly with psql.
    echo "Applying schema.sql to $PGHOST:$PGPORT/$PGDATABASE..."
    PGPASSWORD="$PGPASSWORD" psql \
        -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" \
        -v ON_ERROR_STOP=1 -f "$SCHEMA_FILE"
fi

echo "Schema applied successfully."