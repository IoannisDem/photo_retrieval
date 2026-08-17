from dotenv import load_dotenv
import os
import dataclasses
from sagemaker.core.helper.session_helper import Session
from sagemaker.serve import ModelBuilder
from sagemaker.core import image_uris
import logging
from credentials import AWSCredentials, PostgresCredentials

logger = logging.getLogger(__name__)
load_dotenv()


def get_credentials(local: bool = False) -> AWSCredentials:
    if local:
        return AWSCredentials(
            access_key=os.getenv("AWS_ACCESS_KEY_ID"),
            secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            role_arns=os.getenv("ROLE"),
        )


@dataclasses.dataclass
class ModelImageURI:
    framework: str
    version: str
    py_version: str
    image_scope: str


@dataclasses.dataclass
class ModelDeploymentConfigs:
    image_uri: ModelImageURI
    s3_model_artifacts: str
    instance_type: str
    endpoint_name: str


class FailedEndpointDeployment(Exception):
    pass


def deploy_model(
    model_deployment_configs: ModelDeploymentConfigs, role_arn: str, session: Session
) -> None:

    image_uri = image_uris.retrieve(
        framework=model_deployment_configs.image_uri.framework,
        region=session.boto_region_name,
        version=model_deployment_configs.image_uri.version,
        py_version=model_deployment_configs.image_uri.py_version,
        image_scope=model_deployment_configs.image_uri.image_scope,
        instance_type=model_deployment_configs.instance_type,
    )

    model_builder = ModelBuilder(
        image_uri=image_uri,
        s3_model_data_url=model_deployment_configs.s3_model_artifacts,
        role_arn=role_arn,
        sagemaker_session=session,
        instance_type=model_deployment_configs.instance_type,
        dependencies={},
    )

    model_builder.build()

    predictor = model_builder.deploy(
        initial_instance_count=1,
        instance_type=model_deployment_configs.instance_type,
        endpoint_name=model_deployment_configs.endpoint_name,
    )

    if predictor:
        logger.info(
            f"The {model_deployment_configs.endpoint_name} endpoint has been successfully deployed"
        )
    else:
        msg = f"The {model_deployment_configs.endpoint_name} failed to deploy"
        raise FailedEndpointDeployment(msg)
