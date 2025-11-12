import boto3
from typing import Final, List, Dict, Any

class S3Helper:
    _CONTENT_TYPE: Final = "application/json"
    _CACHE_CONTROL: Final = "no-cache"

    def __init__(self, bucket: str):
        self._s3: Final = boto3.client("s3")
        self.bucket: str = bucket

    def get(self, key: str) -> List[Dict[str, Any]]:
        s3_response = self._s3.get_object(
            Bucket = self.bucket,
            Key = key
        )
        return s3_response["Body"].read().decode("utf-8")
    
    def put(self, key: str, content: List[Dict[str, Any]]):
        self._s3.put_object(
            Bucket = self.bucket,
            Key = key,
            Body = content,
            ContentType= self._CONTENT_TYPE,
            CacheControl = self._CACHE_CONTROL
        )
