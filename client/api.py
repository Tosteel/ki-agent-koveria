import json
from typing import Any, Dict, Optional, List
import requests

class Assistant:
    def __init__(self, name: str, explanation: str, system_prompt: str):
        self.name = name
        self.explanation = explanation
        self.system_prompt = system_prompt

    @classmethod
    def from_dict(cls, d: dict) -> "Assistant":
        return cls(
            name=d.get("name", "").strip(),
            explanation=d.get("explanation", "").strip(),
            system_prompt=d.get("system_prompt", "").strip(),
        )

    def to_payload(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "explanation": self.explanation,
            "system_prompt": self.system_prompt,
        }

class ClientConfig:
    def __init__(self, server_rag: str, api_key: str, server_chat: Optional[str] = None, assistants: Optional[List[Assistant]] = None):
        self.server_rag = server_rag.rstrip("/")
        self.api_key = api_key
        self.server_chat = (server_chat or "http://127.0.0.1:8010").rstrip("/")
        self.assistants: List[Assistant] = assistants or []

    @classmethod
    def from_file(cls, path: str) -> "ClientConfig":
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        assistants = [Assistant.from_dict(a) for a in (cfg.get("assistant") or [])]
        return cls(
            server_rag=cfg["server_rag"],
            api_key=cfg["api_key"],
            server_chat=cfg.get("server_chat"),
            assistants=assistants,
        )

def get_categories(cfg: ClientConfig, include_empty: bool = False) -> Dict[str, Any]:
    url = f"{cfg.server_rag}/rag/categories"
    params = {"include_empty": str(include_empty).lower()} if include_empty else None
    headers = {"Authorization": f"Bearer {cfg.api_key}"}
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()

def rag_query(cfg: ClientConfig, query: str, top_k: int = 3, classification: Optional[str] = None) -> Dict[str, Any]:
    url = f"{cfg.server_rag}/rag/query"
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {"query": query, "top_k": int(top_k)}
    if classification and classification.strip():
        payload["classification"] = classification.strip()
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()

def generate_answer(cfg: ClientConfig, prompt: str, rag_response: Dict[str, Any], assistant: Optional[Assistant]) -> Dict[str, Any]:
    """
    Ruft den lokalen Chat-Server auf und liefert {answer, usage, ...} zurück.
    Schickt den API-Key als Bearer-Token mit.
    """
    url = f"{cfg.server_chat}/chat/generate"
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    payload = {"prompt": prompt, "rag": rag_response}
    if assistant:
        payload["assistant"] = assistant.to_payload()
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()
