import dataclasses
import enum
from pathlib import Path
import utils
from typing import TypeAlias
from functools import cached_property


DEFAULT_REGISTRY_PATH = Path("")


class Provider(enum.StrEnum):
    LOCAL = "local"
    ULTRALYTICS = "ultralytics"
    OPENAI = "openai"


class Version(enum.StrEnum):
    V1 = "1"


@dataclasses.dataclass
class ModelSpec:
    name: str
    provider: Provider
    version: Version


class InvalidModelRegistryPath(Exception):
    pass


class InvalidRegistryFormat(Exception):
    pass


class ModelRegistry:
    def __init__(self, specs: list[ModelSpec]):
        self._indexed_spec = {s.name: s for s in specs}

    @property
    def name_list(self) -> list[str]:
        return [model_spec.name for model_spec in self._specs]

    def get_model_spec(self, name: str) -> ModelSpec:
        spec = self._indexed_spec[name]

        if spec:
            return spec
        else:
            raise ValueError(f"ModelSpec with name '{name}' not found")


RawModelSpec: TypeAlias = dict[str, str]
RegistryData: TypeAlias = dict[str, RawModelSpec]


class ModelRegistryBuilder:
    def __init__(self) -> None:
        self._registry_path: Path | None = None

    def build(self):
        model_specs = self._load_model_specs()
        return ModelRegistry(model_specs)

    def with_registry_path(self, registry_path: Path) -> "ModelRegistryBuilder":
        self._registry_path = registry_path
        return self

    @property
    def registry_path(self) -> Path:
        if self._registry_path:
            return self._registry_path
        else:
            return DEFAULT_REGISTRY_PATH

    def _load_model_specs(self):
        file_type = utils.get_file_type(self.registry_path)
        if file_type == utils.FileType.JSON:
            registry_data = utils.load_json(self.registry_path)
        elif file_type == utils.FileType.YAML:
            registry_data = utils.load_yaml(self.registry_path)
        else:
            msg = f"This is not a suitable model registry path, it can only be a yaml or json: {self.registry_path}"
            raise InvalidModelRegistryPath(msg)
        return self._convert_model_specs(registry_data)

    def _convert_model_specs(self, registry_data: RegistryData) -> list[ModelSpec]:
        try:
            return [
                ModelSpec(
                    name=key,
                    provider=Provider(value["provider"]),
                    version=Version(value["version"]),
                )
                for key, value in registry_data.items()
            ]

        except KeyError as e:
            missing_key = e.args[0]
            raise ValueError(
                f"Model registry entry is missing required field '{missing_key}'. "
                f"Each model must contain 'provider' and 'version'."
            )
