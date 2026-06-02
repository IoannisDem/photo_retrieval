import pytest
from src import utils


@pytest.mark.parametrize(
    "file_name, expected",
    [
        pytest.param("foo.yaml", utils.FileType.YAML, id="yaml_file"),
        pytest.param("foo.yml", utils.FileType.YAML, id="yml_file"),
        pytest.param("foo.json", utils.FileType.JSON, id="json_file"),
    ],
)
def test_valid_get_file_type(file_name, expected):
    observed = utils.get_file_type(file_name)
    assert observed == expected


@pytest.mark.parametrize("file_name", ["foo", "foo.kl", "foo.jpg", "foo.ylm"])
def test_invalid_get_file_type(file_name):
    with pytest.raises(ValueError):
        assert utils.get_file_type(file_name)
