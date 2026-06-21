import json
import base64
import torch
import numpy as np

from io import BytesIO
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from myml import model_predictor_base, model_predictors


def model_fn(model_dir):
    # model_name = "openai/clip-vit-base-patch32"

    # clip_model = CLIPModel.from_pretrained(model_name)
    # processor = CLIPProcessor.from_pretrained(model_name)

    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # clip_model.to(device)
    # clip_model.eval()

    # pipeline = (
    #     model_predictor_base.PipelineBuilder()
    #     .with_model(model_predictors.ModelCLIP(model=clip_model))
    #     .with_processor(
    #         model_predictors.ProcessorCLIP(
    #             processor=processor,
    #             device=device
    #         )
    #     )
    #     .build()
    # )

    return model_predictors.CLIPRawData()


def input_fn(request_body, content_type):
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")

    payload = json.loads(request_body)

    texts = payload.get("texts")
    images_b64 = payload.get("images")

    images = None

    if images_b64:
        images = [
            Image.open(BytesIO(base64.b64decode(img))).convert("RGB")
            for img in images_b64
        ]

    return model_predictors.CLIPRawData(images=images, texts=texts)


def predict_fn(input_data, pipeline):
    return pipeline.predict(images=input_data.images, texts=input_data.texts)


def output_fn(prediction, accept):
    if accept != "application/json":
        raise ValueError(f"Unsupported accept type: {accept}")

    def extract(x):
        if x is None:
            return None

        if hasattr(x, "pooler_output") and x.pooler_output is not None:
            return x.pooler_output.detach().cpu().tolist()

        if torch.is_tensor(x):
            return x.detach().cpu().tolist()

        if isinstance(x, np.ndarray):
            return x.tolist()

        return x

    return json.dumps(
        {
            "image_embedding": extract(prediction.image_output),
            "text_embedding": extract(prediction.text_output),
        }
    )
