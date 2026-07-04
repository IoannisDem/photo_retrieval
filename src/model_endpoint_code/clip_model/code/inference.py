import torch
import logging

from transformers import CLIPModel, CLIPProcessor

from myml import model_predictor_base, utils
from myml import registry
import os
import json

from myml.predictor_models import clip_predictors

logger = logging.getLogger(__name__)


def model_fn(model_dir):
    model_config_path = os.path.join(model_dir, "configs.yaml")
    model_registry = registry.build_model_registry(model_config_path)
    model_name = model_registry.get_model_spec("clip").name
    clip = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip.to(device)
    clip.eval()
    pipeline = (
        model_predictor_base.PipelineBuilder()
        .with_model(clip_predictors.ModelCLIP(model=clip))
        .with_processor(
            clip_predictors.ProcessorCLIP(processor=processor, device=device)
        )
        .build()
    )
    return pipeline


def input_fn(request_body, request_content_type):
    if isinstance(request_body, (bytes, bytearray)):
        request_body = request_body.decode("utf-8")

    payload = json.loads(request_body)

    images = payload.get("images")
    texts = payload.get("texts")

    if images is not None:
        images = utils.decode_image_list(images)

    return {
        "images": images,
        "texts": texts,
    }


def predict_fn(input_data, model):
    raw_input = clip_predictors.CLIPRawData(
        images=input_data["images"],
        texts=input_data["texts"],
    )
    results = model.predict(raw_input)
    return results


def output_fn(prediction, response_content_type):
    result = clip_predictors.CLIPResultOutput(
        image_embeddings=prediction.image_output.tolist()
        if prediction.image_output is not None
        else None,
        text_embeddings=prediction.text_output.tolist()
        if prediction.text_output is not None
        else None,
    )
    return result.model_dump_json()
