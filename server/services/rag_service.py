# server/services/rag_service.py (im Agent-Projekt)
from typing import Any, Dict, Optional
import requests

class RagService:
    def __init__(self, rag_base_url: str, api_key: str):
        self.base = rag_base_url.rstrip("/")
        self.api_key = api_key

    def query(self, query: str, top_k: int = 5, classification: Optional[str] = None) -> Dict[str, Any]:
        url = f"{self.base}/rag/query"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload: Dict[str, Any] = {"query": query, "top_k": int(top_k)}
        if classification:
            payload["classification"] = classification
        r = requests.post(url, json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        return r.json()

    def categories(self, include_empty: bool = False) -> Dict[str, Any]:
        url = f"{self.base}/rag/categories"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        params = {"include_empty": "true"} if include_empty else None
        r = requests.get(url, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json()
