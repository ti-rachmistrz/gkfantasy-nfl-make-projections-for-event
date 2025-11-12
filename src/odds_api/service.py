import boto3
from botocore.exceptions import ClientError
import requests
import json
from typing import List, Dict, Any
from .response import OddsApiResponse

class OddsApiService:
    def __init__(self, base_url: str, secret_id: str):
        self.base_url: str = base_url
        self._api_key: str = self._get_api_key(secret_id)

    def get_event_odds(self, event_id: str, markets: str, bookmakers: str) -> OddsApiResponse:
        response = requests.get(
            url = f"{self.base_url}{event_id}/odds",
            params = {
                "apiKey"            : self._api_key,
                "regions"           : "us",
                "markets"           : markets,
                "oddsFormat"        : "decimal",
                "bookmakers"        : bookmakers,
                "includeLinks"      : "false",
                "includeSids"       : "false",
                "includeBetLimits"  : "false"
            },
            timeout = 10
        )

        response.raise_for_status()
        return OddsApiResponse(response.json())
    
    def _get_api_key(self, secret_id: str) -> str:
        secrets_client = boto3.client("secretsmanager")
        try:
            secret_response = secrets_client.get_secret_value(SecretId=secret_id)
            key = json.loads(secret_response['SecretString'])['key']
        except ClientError as e:
            raise RuntimeError(f"Failed to retrieve api key: {e.response['Error']['Code']}") from e
        
        return key
    
