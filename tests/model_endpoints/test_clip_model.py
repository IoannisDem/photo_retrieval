import pytest
from model_endpoint_code.clip_model.code import inference
from myml import model_predictor_base, model_predictors


def test_m():
    assert inference.model_fn("") == model_predictors.CLIPRawData()
