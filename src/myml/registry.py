import dataclasses
import enum
from pathlib import Path
from myml import utils
from typing import TypeAlias


DEFAULT_REGISTRY_PATH = Path("")


class Provider(enum.StrEnum):
    LOCAL = "local"
    ULTRALYTICS = "ultralytics"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"


class Version(enum.StrEnum):
    V1 = "1"


@dataclasses.dataclass
class ModelSpec:
    key_name: str
    name: str
    provider: Provider
    version: Version | None = None


class InvalidModelRegistryPath(Exception):
    pass


class InvalidRegistryFormat(Exception):
    pass


class ModelRegistry:
    def __init__(self, specs: list[ModelSpec]):
        self._indexed_spec = {s.key_name: s for s in specs}

    @property
    def name_list(self) -> list[str]:
        return list(self._indexed_spec.keys())

    def get_model_spec(self, name: str) -> ModelSpec:
        spec = self._indexed_spec[name]

        if spec:
            return spec
        else:
            raise ValueError(f"ModelSpec with name '{name}' not found")


def data_extraction(path: str | Path):
    file_type = utils.get_file_type(path)
    match file_type:
        case utils.FileType.YAML:
            return utils.load_yaml(path)
        case utils.FileType.JSON:
            return utils.load_json(path)


RawModelSpec: TypeAlias = dict[str, str]


def _extract_model_spec(content: dict[str, RawModelSpec]) -> list[ModelSpec]:
    model_spec_list: list[ModelSpec] = []
    for key, value in content.items():
        key_name = key
        model_name = value["name"]
        provider = Provider(value["provider"])
        version = Version(value["version"]) if value.get("version") else None
        model_spec_list.append(ModelSpec(key_name, model_name, provider, version))
    return model_spec_list


def build_model_registry(model_registry_path: str | Path) -> ModelRegistry:
    content = data_extraction(model_registry_path)
    model_specs = _extract_model_spec(content)
    model_registry = ModelRegistry(model_specs)
    return model_registry
