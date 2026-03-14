from __future__ import annotations

import os
import json
import requests
from typing import Any, Dict, List, Optional

from .agent_prompts import (
    get_clarification_system_prompt,
    get_final_system_prompt,
    get_planner_system_prompt,
)

OPENAI_URL = os.getenv("OPENAI_URL", "https://api.openai.com/v1/responses").strip() or "https://api.openai.com/v1/responses"


class LlmOpenai:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

    def enabled(self) -> bool:
        return bool(self.api_key)

    @staticmethod
    def _to_responses_input(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert simple {role, content:str} messages into Responses API input items with content parts.
        """
        out: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if content is None:
                content = ""
            # Responses API: content as array of parts
            out.append(
                {
                    "role": role,
                    "content": [{"type": "input_text", "text": str(content)}],
                }
            )
        return out

    def _call(
        self,
        input_messages: List[Dict[str, Any]],
        text_format: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        payload: Dict[str, Any] = {
            "model": self.model,
            "input": self._to_responses_input(input_messages),
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

        if r.status_code >= 400:
            # Body im Exception-Text – den siehst du dann sicher im uvicorn Trace
            raise RuntimeError(f"OpenAI HTTP {r.status_code}: {r.text}")

        return r.json()

        # Hilft massiv beim Debuggen von 400ern
        if r.status_code >= 400:
            print("OPENAI ERROR STATUS:", r.status_code)
            print("OPENAI ERROR BODY:", r.text)

        r.raise_for_status()
        return r.json()

    def plan_steps(self, *, goal: str, tool_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        tool_schema kommt bei dir als:
          {"name": "tool_plan", "schema": {...}}
        und muss für OpenAI json_schema so gemappt werden:
          {"type":"json_schema","name": "...","schema": {...},"strict": True}
        """
        json_schema_format = {
            "type": "json_schema",
            "name": tool_schema.get("name", "tool_plan"),
            "schema": tool_schema.get("schema", tool_schema),  # fallback
            "strict": False,
        }

        resp = self._call(
            input_messages=[
                {
                    "role": "system",
                    "content": get_planner_system_prompt("openai"),
                },
                {
                    "role": "user",
                    "content": f"Goal: {goal}\nCreate an execution plan using the available tools.",
                },
            ],
            text_format=json_schema_format,
        )

        # output_text extrahieren
        out = ""
        for item in resp.get("output", []):
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    out += c.get("text", "")
        return json.loads(out)

    def clarify_goal(self, *, goal: str) -> Dict[str, Any]:
        schema = {
            "type": "json_schema",
            "name": "clarification_gate",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "status": {"type": "string", "enum": ["ready", "needs_info"]},
                    "goal_summary": {"type": "string"},
                    "normalized_goal": {"type": "string"},
                    "missing_fields": {"type": "array", "items": {"type": "string"}},
                    "questions": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["status", "goal_summary", "normalized_goal", "missing_fields", "questions"],
            },
            "strict": False,
        }
        resp = self._call(
            input_messages=[
                {
                    "role": "system",
                    "content": get_clarification_system_prompt("openai"),
                },
                {"role": "user", "content": f"Anfrage: {goal}"},
            ],
            text_format=schema,
        )
        out = ""
        for item in resp.get("output", []):
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    out += c.get("text", "")
        try:
            parsed = json.loads(out or "{}")
        except Exception:
            parsed = {}
        status = parsed.get("status")
        if status not in {"ready", "needs_info"}:
            status = "ready"
        return {
            "status": status,
            "goal_summary": str(parsed.get("goal_summary") or ""),
            "normalized_goal": str(parsed.get("normalized_goal") or goal),
            "missing_fields": list(parsed.get("missing_fields") or []),
            "questions": list(parsed.get("questions") or []),
        }

    def final_answer(self, *, goal: str, tool_outputs: List[Dict[str, Any]]) -> str:
        resp = self._call(
            input_messages=[
                {
                    "role": "system",
                    "content": get_final_system_prompt("openai"),
                },
                {
                    "role": "user",
                    "content": f"Goal: {goal}\nTool outputs:\n{tool_outputs}",
                },
            ],
        )

        out = ""
        for item in resp.get("output", []):
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    out += c.get("text", "")
        return out.strip()
