from model_endpoint_code.clip_model.code import inference
from myml import model_predictors
from unittest import mock
import pytest
import torch
from PIL import Image


def equal_image_check(imga: Image.Image, imgb: Image.Image) -> bool:
    return imga.tobytes() == imgb.tobytes()


@pytest.fixture
def clip_raw_mock():
    sample_image = Image.new("RGB", (224, 224))
    sample_image.format = "JPEG"
    raw_input = model_predictors.CLIPRawData(
        images=[sample_image],
        texts=["a dog", "a cat"],
    )
    return raw_input


@pytest.fixture
def clip_raw_pydantic_mock(clip_raw_mock):
    request_input = model_predictors.clip_request_conversion(clip_raw_mock)
    return request_input


@pytest.fixture
def clip_raw_request_mock(clip_raw_pydantic_mock):
    return clip_raw_pydantic_mock.model_dump_json()


@pytest.fixture
def clip_output_mock():
    return model_predictors.CLIPOutput(
        image_output=torch.rand(1, 512),
        text_output=torch.rand(1, 512),
    )


@pytest.fixture
def model_fn_mock(clip_output_mock):
    model = mock.Mock()
    model.predict.return_value = clip_output_mock
    return model


@pytest.fixture
def expected_input_fn(clip_raw_mock):
    expected = {
        "images": clip_raw_mock.images,
        "texts": clip_raw_mock.texts,
    }
    return expected


def test_input_fn(expected_input_fn, clip_raw_request_mock):
    observed = inference.input_fn(clip_raw_request_mock, None)
    assert expected_input_fn["texts"] == observed["texts"]
    assert all(
        equal_image_check(imga, imgb)
        for imga, imgb in zip(expected_input_fn["images"], observed["images"])
    )


def test_predict_fn(expected_input_fn, model_fn_mock, clip_output_mock):
    observed = inference.predict_fn(expected_input_fn, model_fn_mock)
    assert clip_output_mock == observed
