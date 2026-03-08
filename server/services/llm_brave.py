#https://api-dashboard.search.brave.com/documentation/services/answers

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests


BRAVE_URL = (
    os.getenv("BRAVE_URL", "https://api.search.brave.com/res/v1").strip()
    or "https://api.search.brave.com/res/v1"
)


def _chat_completions_url(base_or_endpoint: str) -> str:
    u = str(base_or_endpoint or "").strip().rstrip("/")
    if u.endswith("/chat/completions"):
        return u
    return f"{u}/chat/completions"


class LlmBrave:
    def __init__(self):
        # tolerate typo from env request + standard key name
        self.api_key = (os.getenv("BRAVE_API_KEY", "").strip())
        self.model = os.getenv("BRAVE_MODEL", "brave").strip() or "brave"

    def enabled(self) -> bool:
        return bool(self.api_key)

    def chat_completions(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
        language: Optional[str] = None,
        country: Optional[str] = None,
        enable_research: Optional[bool] = None,
        enable_citations: Optional[bool] = None,
        enable_entities: Optional[bool] = None,
        timeout_s: int = 60,
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("BRAVE_API_EY/BRAVE_API_KEY not set")

        payload: Dict[str, Any] = {
            "model": str(model or self.model),
            "stream": bool(stream),
            "messages": [
                {"role": str(m.get("role") or "user"), "content": str(m.get("content") or "")}
                for m in (messages or [])
            ],
        }
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        if top_p is not None:
            payload["top_p"] = float(top_p)
        if isinstance(response_format, dict) and response_format:
            payload["response_format"] = response_format
        if language:
            payload["language"] = str(language).strip()
        if country:
            payload["country"] = str(country).strip()
        if enable_research is not None:
            payload["enable_research"] = bool(enable_research)
        if enable_citations is not None:
            payload["enable_citations"] = bool(enable_citations)
        if enable_entities is not None:
            payload["enable_entities"] = bool(enable_entities)

        if stream:
            r = requests.post(
                _chat_completions_url(BRAVE_URL),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=timeout_s,
                stream=True,
            )
            if r.status_code >= 400:
                raise RuntimeError(f"Brave HTTP {r.status_code}: {r.text}")
            parts: List[str] = []
            usage: Dict[str, Any] = {}
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                s = str(line).strip()
                if not s.startswith("data:"):
                    continue
                data = s[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    chunk = json.loads(data)
                except Exception:
                    continue
                if isinstance(chunk.get("usage"), dict):
                    usage = chunk["usage"]
                try:
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    c = delta.get("content")
                    if isinstance(c, str) and c:
                        parts.append(c)
                except Exception:
                    continue
            return {
                "choices": [{"message": {"content": "".join(parts)}}],
                "usage": usage or None,
            }

        r = requests.post(
            _chat_completions_url(BRAVE_URL),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout_s,
        )
        if r.status_code >= 400:
            raise RuntimeError(f"Brave HTTP {r.status_code}: {r.text}")
        return r.json()

    @staticmethod
    def extract_text(completion: Dict[str, Any]) -> str:
        try:
            content = completion["choices"][0]["message"]["content"]
        except Exception:
            return ""
        if isinstance(content, str):
            return content.strip()
        return str(content).strip()
