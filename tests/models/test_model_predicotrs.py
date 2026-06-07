from __future__ import annotations

from unittest import mock

import pytest
import torch
from PIL import Image

from src.models import model_predictors


@pytest.fixture
def processor_mock():
    processor = mock.Mock()
    processor.return_value = {
        "pixel_values": torch.zeros(1, 3, 224, 224),
        "input_ids": torch.zeros(1, 10, dtype=torch.long),
        "attention_mask": torch.ones(1, 10, dtype=torch.long),
    }
    return processor


@pytest.fixture
def model_mock():
    model = mock.Mock()
    model.vision_model.return_value = mock.Mock(pooler_output=torch.zeros(1, 768))
    model.text_model.return_value = mock.Mock(pooler_output=torch.zeros(1, 512))
    model.visual_projection.return_value = torch.zeros(1, 512)
    model.text_projection.return_value = torch.zeros(1, 512)
    return model


_IMAGE = Image.new("RGB", (224, 224))
_TEXTS = ["a dog", "a cat"]
_IMAGE_TENSOR = {
    "pixel_values": torch.zeros(1, 3, 224, 224),
    "input_ids": torch.zeros(1, 10, dtype=torch.long),
    "attention_mask": torch.ones(1, 10, dtype=torch.long),
}

_TEXT_TENSOR = {
    "pixel_values": torch.zeros(1, 3, 224, 224),  # optional depending on your mock
    "input_ids": torch.zeros(1, 10, dtype=torch.long),
    "attention_mask": torch.ones(1, 10, dtype=torch.long),
}


def assert_tensor_dict(observed_tensor, expected_tensor):
    assertion_results = []
    for k in expected_tensor:
        assertion_results.append(
            torch.all(observed_tensor[k] == expected_tensor[k]).item()
        )

    return all(assertion_results)


@pytest.mark.parametrize(
    "raw_data, expected",
    [
        (
            model_predictors.CLIPRawData(images=[_IMAGE], texts=None),
            model_predictors.CLIPInput(images=_IMAGE_TENSOR, texts=None),
        ),
        (
            model_predictors.CLIPRawData(images=None, texts=_TEXTS),
            model_predictors.CLIPInput(images=None, texts=_TEXT_TENSOR),
        ),
        (
            model_predictors.CLIPRawData(images=[_IMAGE], texts=_TEXTS),
            model_predictors.CLIPInput(images=_IMAGE_TENSOR, texts=_TEXT_TENSOR),
        ),
        (
            model_predictors.CLIPRawData(images=None, texts=None),
            model_predictors.CLIPInput(images=None, texts=None),
        ),
    ],
    ids=["images-no_texts", "no_images-texts", "images-texts", "no_images-no_texts"],
)
class TestProcessorCLIP:
    def assert_output(self, observed, expected):
        images_assert = False
        texts_assert = False
        if observed.images and expected.images:
            images_assert = assert_tensor_dict(observed.images, expected.images)
        elif observed.images is None and expected.images is None:
            images_assert = True
        if observed.texts and expected.texts:
            texts_assert = assert_tensor_dict(observed.texts, expected.texts)
        elif observed.texts is None and expected.texts is None:
            texts_assert = True
        return all([images_assert, texts_assert])

    def test_process(self, processor_mock, raw_data, expected):
        processor = model_predictors.ProcessorCLIP(
            processor=processor_mock, device="cpu"
        )
        observed = processor.process(raw_data)
        assert self.assert_output(observed, expected)
