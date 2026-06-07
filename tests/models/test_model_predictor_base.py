import pytest
from src.models import model_predictor_base


class ModelMock(model_predictor_base.Model):
    def predict(self, data):
        return data


@pytest.fixture
def mock_model():
    return ModelMock()


@pytest.fixture
def simple_pipeline(mock_model):
    pipeline_builder = model_predictor_base.PipelineBuilder()
    pipeline_builder.with_model(mock_model)
    return pipeline_builder.build()


@pytest.mark.parametrize(
    "input_data, expected",
    [pytest.param(1, 1), pytest.param(2, 2), pytest.param(10, 10)],
)
def test_simple_pipeline_builder(input_data, expected, simple_pipeline):
    observed = simple_pipeline.predict(input_data)
    assert observed == expected


class HalfProcessor(model_predictor_base.Processor):
    def process(self, data):
        return data // 2


@pytest.fixture
def mock_processor():
    return HalfProcessor()


@pytest.fixture
def composite_pipeline(mock_model, mock_processor):
    pipeline_builder = model_predictor_base.PipelineBuilder()
    pipeline_builder.with_model(mock_model)
    pipeline_builder.with_processor(mock_processor)
    return pipeline_builder.build()


@pytest.mark.parametrize(
    "input_data, expected",
    [pytest.param(1, 0), pytest.param(2, 1), pytest.param(10, 5)],
)
def test_composite_pipeline_builder(input_data, expected, composite_pipeline):
    observed = composite_pipeline.predict(input_data)
    assert observed == expected


def test_missing_pipeline_model():
    pipeline_builder = model_predictor_base.PipelineBuilder()
    with pytest.raises(model_predictor_base.PipelineMissingModel):
        pipeline_builder.build()
