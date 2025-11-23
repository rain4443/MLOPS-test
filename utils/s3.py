import boto3, os
from botocore.config import Config as BotoConfig
from .config import AWS_REGION, S3_BUCKET

_s3 = boto3.client("s3", region_name=AWS_REGION,
                   config=BotoConfig(retries={"max_attempts": 5, "mode": "standard"}))

def s3_key_exists(key: str) -> bool:
    try:
        _s3.head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except Exception:
        return False

def upload_file(local_path: str, key: str):
    _s3.upload_file(local_path, S3_BUCKET, key)

def download_file(key: str, local_path: str):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    _s3.download_file(S3_BUCKET, key, local_path)
