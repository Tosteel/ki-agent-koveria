from __future__ import annotations

from typing import Any, Dict

import requests


class SearchService:
    def __init__(self, search_base_url: str, api_key: str):
        self.base = search_base_url.rstrip("/")
        self.api_key = api_key

    def _headers(self) -> Dict[str, str]:
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def generate_json_request_body(self, user_prompt: str) -> Dict[str, Any]:
        """
        POST /generate_json_request_body
        payload:
          {"user_prompt": "..."}
        """
        url = f"{self.base}/generate_json_request_body"
        payload = {"user_prompt": user_prompt}
        r = requests.post(url, json=payload, headers=self._headers(), timeout=60)
        r.raise_for_status()
        return r.json()

    def generate_json(self, prompt: str, table: Dict[str, Any], batches: int = 1) -> Dict[str, Any]:
        """
        POST /generate_json
        payload:
          {
            "prompt": "...",
            "table": {"delimiter": ";", "columns": [{"name":"...","description":"..."}]},
            "batches": 1
          }
        """
        url = f"{self.base}/generate_json"
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "table": table,
            "batches": int(batches),
        }
        r = requests.post(url, json=payload, headers=self._headers(), timeout=120)
        r.raise_for_status()
        return r.json()

    def search_generate_json(self, user_prompt: str) -> Dict[str, Any]:
        """
        POST /search/generate_json
        payload:
          {"user_prompt": "..."}
        """
        url = f"{self.base}/search/generate_json"
        payload = {"user_prompt": user_prompt}
        r = requests.post(url, json=payload, headers=self._headers(), timeout=120)
        r.raise_for_status()
        return r.json()
