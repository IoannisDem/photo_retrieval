from model_endpoint_code.clip_model.code import inference
from myml.predictor_models import clip_predictors
import pytest
from PIL import Image


def equal_image_check(imga: Image.Image, imgb: Image.Image) -> bool:
    return imga.tobytes() == imgb.tobytes()


def build_clip_raw_data(raw_image_size_list, raw_text_list):
    sample_images = []
    for raw_image_size in raw_image_size_list:
        img = Image.new("RGB", raw_image_size)
        img.format = "JPEG"
        sample_images.append(img)

    return clip_predictors.CLIPRawData(
        images=sample_images,
        texts=raw_text_list,
    )


def build_clip_raw_pydantic(clip_raw_data):
    return clip_predictors.clip_request_conversion(clip_raw_data)


class TestInputFn:
    TEST_CASES = (
        pytest.param(([(224, 224)], ["a dog", "a cat"]), id="single_image_pair"),
        pytest.param(([(64, 64)], ["a small image"]), id="small_image_single_text"),
        pytest.param(
            ([(224, 224), (64, 64)], ["a dog", "a cat", "a bird"]),
            id="multi_image_multi_text",
        ),
    )

    @pytest.fixture(params=TEST_CASES)
    def test_case(self, request):
        return request.param

    @pytest.fixture
    def clip_raw_data_mock(self, test_case):
        raw_image_size_list, raw_text_list = test_case
        return build_clip_raw_data(raw_image_size_list, raw_text_list)

    @pytest.fixture
    def clip_raw_request_mock(self, clip_raw_data_mock):
        clip_raw_pydantic = build_clip_raw_pydantic(clip_raw_data_mock)
        return clip_raw_pydantic.model_dump_json()

    @pytest.fixture
    def expected(self, clip_raw_data_mock):
        return {
            "images": clip_raw_data_mock.images,
            "texts": clip_raw_data_mock.texts,
        }

    def test_input_fn(self, expected, clip_raw_request_mock):
        observed = inference.input_fn(clip_raw_request_mock, None)

        assert expected["texts"] == observed["texts"]
        assert all(
            equal_image_check(imga, imgb)
            for imga, imgb in zip(expected["images"], observed["images"])
        )
