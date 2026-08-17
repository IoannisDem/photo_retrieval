from __future__ import annotations

import boto3
from botocore.client import BaseClient
from botocore.exceptions import ClientError
import credentials
import logging

logger = logging.getLogger(__name__)


def build_s3_client(aws_cred: credentials.AWSCredentials) -> BaseClient:
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=aws_cred.access_key,
        aws_secret_access_key=aws_cred.secret_key,
        region_name=aws_cred.region,
    )
    return s3_client


class AWSS3ClientWrapper:
    def __init__(self, s3_client: BaseClient, bucket: str, prefix: str):
        self._s3_client = s3_client
        self._bucket = bucket
        self._prefix = prefix.rstrip("/")

    def write(self, key: str, data: bytes) -> None:
        full_key = self._key(key)
        try:
            self._s3_client.put_object(Bucket=self._bucket, Key=full_key, Body=data)
            logger.debug("Wrote object to s3://%s/%s", self._bucket, full_key)
        except ClientError:
            logger.error("Failed to write object to s3://%s/%s", self._bucket, full_key)
            raise

    def read(self, key: str) -> bytes:
        full_key = self._key(key)
        try:
            response = self._s3_client.get_object(Bucket=self._bucket, Key=full_key)
            data = response["Body"].read()
            logger.debug("Read object from s3://%s/%s", self._bucket, full_key)
            return data
        except ClientError as e:
            code = e.response["Error"]["Code"]
            logger.error(
                "Failed to read object from s3://%s/%s: %s",
                self._bucket,
                full_key,
                code,
            )
            raise

    def delete(self, key: str) -> None:
        full_key = self._key(key)
        try:
            self._s3_client.delete_object(Bucket=self._bucket, Key=full_key)
            logger.debug("Deleted object at s3://%s/%s", self._bucket, full_key)
        except ClientError:
            logger.error(
                "Failed to delete object at s3://%s/%s", self._bucket, full_key
            )
            raise

    def list(self, sub_prefix: str = "") -> list[str]:
        full_prefix = self._key(sub_prefix) if sub_prefix else self._prefix + "/"
        keys: list[str] = []
        try:
            paginator = self._s3_client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self._bucket, Prefix=full_prefix):
                keys.extend(obj["Key"] for obj in page.get("Contents", []))
            logger.debug(
                "Listed %d objects under s3://%s/%s",
                len(keys),
                self._bucket,
                full_prefix,
            )
            return keys
        except ClientError:
            logger.error(
                "Failed to list objects under s3://%s/%s", self._bucket, full_prefix
            )
            raise

    def _key(self, key: str) -> str:
        key = key.lstrip("/")
        return f"{self._prefix}/{key}" if self._prefix else key
