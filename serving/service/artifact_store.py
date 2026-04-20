import os
import json
from pathlib import Path
import boto3


class MinioStore:
    def __init__(self):
        self.client = boto3.client(
            "s3",
            endpoint_url=os.getenv("MINIO_ENDPOINT"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        self.bucket = os.getenv("MINIO_BUCKET", "models")

    def get_json(self, key: str):
        obj = self.client.get_object(Bucket=self.bucket, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))

    def download(self, key: str, local_path: str):
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, key, local_path)
        return local_path
