from __future__ import annotations

import os
import requests
from typing import Any, Dict, List, Optional

OPENAI_URL = "https://api.openai.com/v1/responses"

class LlmRuntime:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

    def enabled(self) -> bool:
        return bool(self.api_key)

    def _call(self, input_messages: List[Dict[str, Any]], text_format: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        payload: Dict[str, Any] = {
            "model": self.model,
            "input": input_messages,
        }
        if text_format is not None:
            payload["text"] = {"format": text_format}

        r = requests.post(
            OPENAI_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        r.raise_for_status()
        return r.json()

    def plan_steps(self, *, goal: str, tool_schema: Dict[str, Any]) -> Dict[str, Any]:
        # Structured Output (json_schema strict). :contentReference[oaicite:2]{index=2}
        resp = self._call(
            input_messages=[
                {"role": "system", "content": "You are a planner. Produce ONLY JSON matching the schema."},
                {"role": "user", "content": f"Goal: {goal}\nCreate an execution plan using the available tools."},
            ],
            text_format={
                "type": "json_schema",
                "strict": True,
                "schema": tool_schema,
            },
        )

        # Responses API gibt die strukturierte Ausgabe in output_text / output ab; pragmatisch:
        # Wir lesen das JSON aus resp["output"][...]["content"][...]["text"].
        # (SDK würde das schöner machen, aber hier ohne SDK.)
        out = ""
        for item in resp.get("output", []):
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    out += c.get("text", "")
        import json
        return json.loads(out)

    def final_answer(self, *, goal: str, tool_outputs: List[Dict[str, Any]]) -> str:
        resp = self._call(
            input_messages=[
                {"role": "system", "content": "You are an assistant. Use the tool outputs to answer the goal succinctly."},
                {"role": "user", "content": f"Goal: {goal}\nTool outputs:\n{tool_outputs}"},
            ],
        )
        # output_text extrahieren
        out = ""
        for item in resp.get("output", []):
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    out += c.get("text", "")
        return out.strip()
