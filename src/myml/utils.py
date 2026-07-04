import enum
from pathlib import Path
import yaml
import json
from typing import Any
import logging
import base64
from PIL import Image
from io import BytesIO
import dataclasses
from typing import Literal

logger = logging.getLogger(__name__)


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


@dataclasses.dataclass
class ImageData:
    content: Image.Image
    format: Literal["JPEG", "PNG"]


def encode_image(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format=image.format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def decode_image_list(images_b64: list[str]) -> list[Image.Image]:
    logger.info(f"[DECODE] Decoding {len(images_b64)} images")

    images = []
    for img_b64 in images_b64:
        img_bytes = base64.b64decode(img_b64)
        image = Image.open(BytesIO(img_bytes)).convert("RGB")
        images.append(image)

    return images


def decode_text_list(texts: list[str]) -> list[str]:

    return list(texts)
