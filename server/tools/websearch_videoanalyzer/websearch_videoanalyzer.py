from __future__ import annotations

from typing import Any, Dict

import requests


class VideoAnalyzerService:
    def __init__(self, base_url: str, api_key: str):
        self.base = base_url.rstrip("/")
        self.api_key = api_key

    def _headers(self) -> Dict[str, str]:
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def analyze_json_sync_from_prompt(self, prompt: str) -> Dict[str, Any]:
        url = f"{self.base}/video/analyze_json_sync_from_prompt"
        payload = {"prompt": prompt}
        r = requests.post(url, json=payload, headers=self._headers(), timeout=120)
        r.raise_for_status()
        return r.json()
