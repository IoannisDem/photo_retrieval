import pytest
from src.models import registry


class TestModelRegistry:
    TESTCASES = (
        [
            {
                "clip": registry.ModelSpec("clip", "clip-v1", registry.Provider.OPENAI),
                "yolo": registry.ModelSpec(
                    "yolo", "face-yolo", registry.Provider.ULTRALYTICS
                ),
                "vit": registry.ModelSpec("vit", "custom-vit", registry.Provider.LOCAL),
            },
            ["clip", "yolo", "vit"],
        ],
        [{}, []],
    )

    @pytest.fixture(params=TESTCASES)
    def testcases(self, request):
        return request.param

    @pytest.fixture
    def expected_model_spec(self, testcases):
        return testcases[0]

    @pytest.fixture
    def expected_name_list(self, testcases):
        return testcases[1]

    @pytest.fixture
    def model_registry(self, testcases):
        return registry.ModelRegistry(list(testcases[0].values()))

    def test_name_list(self, model_registry, expected_name_list):
        assert model_registry.name_list == expected_name_list

    def test_get_model_spec(self, model_registry, expected_model_spec):
        for name in model_registry.name_list:
            assert model_registry.get_model_spec(name) == expected_model_spec[name]
