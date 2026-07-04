import pytest
import boto3
from moto import mock_aws

from storage import storage_base


class TestS3ClientWrapper:
    @pytest.fixture
    def s3_boto(self):
        s3 = boto3.client("s3", region_name="us-east-1")
        return s3

    @pytest.fixture
    def bucket(self):
        return "foo"

    @pytest.fixture
    def prefix(self):
        return "bar"

    @pytest.fixture
    def s3_client_wrapper(self, s3_boto, bucket, prefix):
        s3_client = storage_base.AWSS3ClientWrapper(s3_boto, bucket, prefix)
        return s3_client

    @mock_aws
    def test_write(self, s3_boto, s3_client_wrapper, bucket, prefix):
        key = "test_object.txt"
        body = "testing"
        s3_boto.create_bucket(Bucket=bucket)

        s3_client_wrapper.write(key, body.encode("utf-8"))

        response = s3_boto.get_object(Bucket=bucket, Key=f"{prefix}/{key}")
        assert response["Body"].read() == body.encode("utf-8")

    @mock_aws
    def test_write_uses_prefixed_key(self, s3_boto, s3_client_wrapper, bucket, prefix):
        key = "test_object.txt"
        expected = f"{prefix}/{key}"

        s3_boto.create_bucket(Bucket=bucket)

        s3_client_wrapper.write(key, b"data")

        listing = s3_boto.list_objects_v2(Bucket=bucket)
        observed_keys = [obj["Key"] for obj in listing["Contents"]]

        assert expected in observed_keys

    @mock_aws
    def test_read_returns_written_bytes(
        self, s3_boto, s3_client_wrapper, bucket, prefix
    ):
        key = "test_object.txt"
        expected = b"hello world"

        s3_boto.create_bucket(Bucket=bucket)
        s3_client_wrapper.write(key, expected)

        observed = s3_client_wrapper.read(key)

        assert observed == expected
