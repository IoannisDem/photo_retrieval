import enum
from pathlib import Path
import yaml
import json
from typing import Any


class FileType(enum.StrEnum):
    YAML = "yaml"
    JSON = "json"


def get_file_type(file_name: str | Path) -> FileType:
    path = Path(file_name)
    suffix = path.suffix.lower().lstrip(".")

    if suffix in {"yaml", "yml"}:
        return FileType.YAML
    elif suffix == "json":
        return FileType.JSON
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    try:
        with open(path, "r") as f:
            return json.load(f)

    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {path}")

    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in file: {path}")


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    except FileNotFoundError:
        raise FileNotFoundError(f"YAML file not found: {path}")

    except yaml.YAMLError:
        raise ValueError(f"Invalid YAML format in file: {path}")
