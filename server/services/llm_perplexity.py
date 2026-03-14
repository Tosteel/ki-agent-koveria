from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests

from .agent_prompts import (
    get_clarification_system_prompt,
    get_final_system_prompt,
    get_planner_system_prompt,
)

PERPLEXITY_URL = (
    os.getenv("PERPLEXITY_URL", "https://api.perplexity.ai/chat/completions").strip()
    or "https://api.perplexity.ai/chat/completions"
)


class LlmPerplexity:
    def __init__(self):
        self.api_key = os.getenv("PERPLEXITY_API_KEY", "").strip()
        self.model = os.getenv("PERPLEXITY_MODEL", "sonar-pro").strip()

    def enabled(self) -> bool:
        return bool(self.api_key)

    def _call(
        self,
        input_messages: List[Dict[str, Any]],
        response_format: Optional[Dict[str, Any]] = None,
        text_format: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("PERPLEXITY_API_KEY not set")

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": str(m.get("role") or "user"),
                    "content": str(m.get("content") or ""),
                }
                for m in input_messages
            ],
        }
        # Keep compatibility with call sites that use OpenAI-Responses style arg `text_format`.
        # Perplexity expects response_format in chat-completions shape:
        # - {"type":"text"}
        # - {"type":"json_schema","json_schema":{"name":"...","schema":{...}}}
        # - {"type":"regex","regex":"..."}
        rf = response_format if response_format is not None else text_format
        normalized_rf = self._normalize_response_format(rf)
        if normalized_rf is not None:
            payload["response_format"] = normalized_rf

        r = requests.post(
            PERPLEXITY_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        if r.status_code >= 400:
            raise RuntimeError(f"Perplexity HTTP {r.status_code}: {r.text}")
        data = r.json()
        # Provide OpenAI-Responses-like compatibility for existing parsers expecting `output -> output_text`.
        txt = self._extract_text(data)
        if "output" not in data:
            data["output"] = [{"content": [{"type": "output_text", "text": txt}]}]
        return data

    @staticmethod
    def _extract_text(resp: Dict[str, Any]) -> str:
        try:
            content = resp["choices"][0]["message"]["content"]
        except Exception:
            return ""

        if isinstance(content, str):
            return content.strip()
        if isinstance(content, (int, float)):
            return str(content).strip()
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    s = item.strip()
                    if s:
                        parts.append(s)
                    continue
                if isinstance(item, dict):
                    # Common block styles: {"type":"text","text":"..."} or {"text":"..."}
                    txt = item.get("text")
                    if isinstance(txt, str) and txt.strip():
                        parts.append(txt.strip())
                        continue
                    # Defensive fallback for dict-like blocks.
                    for v in item.values():
                        if isinstance(v, str) and v.strip():
                            parts.append(v.strip())
                            break
            return "\n".join(parts).strip()
        if isinstance(content, dict):
            txt = content.get("text")
            if isinstance(txt, str):
                return txt.strip()
            try:
                return json.dumps(content, ensure_ascii=False).strip()
            except Exception:
                return ""
        return str(content).strip()

    @staticmethod
    def _normalize_response_format(fmt: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not isinstance(fmt, dict) or not fmt:
            return None

        t = str(fmt.get("type") or "").strip().lower()
        if not t:
            return None

        if t == "text":
            return {"type": "text"}

        if t == "regex":
            rx = str(fmt.get("regex") or "").strip()
            if rx:
                return {"type": "regex", "regex": rx}
            return {"type": "text"}

        # OpenAI compatibility shim: many call sites use json_object.
        if t == "json_object":
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": {"type": "object", "additionalProperties": True},
                },
            }

        if t == "json_schema":
            # Already in Perplexity-compatible shape.
            js = fmt.get("json_schema")
            if isinstance(js, dict):
                name = str(js.get("name") or "response").strip() or "response"
                schema = js.get("schema")
                if isinstance(schema, dict):
                    return {"type": "json_schema", "json_schema": {"name": name, "schema": schema}}
                # Fallback when json_schema exists but schema is missing/invalid.
                return {
                    "type": "json_schema",
                    "json_schema": {
                        "name": name,
                        "schema": {"type": "object", "additionalProperties": True},
                    },
                }

            # OpenAI Responses-style shape: {"type":"json_schema","name":"...","schema":{...}}
            schema = fmt.get("schema")
            if isinstance(schema, dict):
                name = str(fmt.get("name") or "response").strip() or "response"
                return {"type": "json_schema", "json_schema": {"name": name, "schema": schema}}

            # Last-resort fallback.
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": {"type": "object", "additionalProperties": True},
                },
            }

        return None

    @staticmethod
    def _parse_json_strictish(text: str) -> Dict[str, Any]:
        if not text:
            return {}
        t = text.strip()
        if t.startswith("```"):
            t = t.strip("`").strip()
            if "\n" in t:
                first, rest = t.split("\n", 1)
                if first.strip().lower() in {"json", "javascript"}:
                    t = rest.strip()
        try:
            obj = json.loads(t)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            pass
        start = t.find("{")
        end = t.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(t[start : end + 1])
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}
        return {}

    def plan_steps(self, *, goal: str, tool_schema: Dict[str, Any]) -> Dict[str, Any]:
        planner_system = get_planner_system_prompt("openai")
        schema = tool_schema.get("schema", tool_schema)
        schema_name = tool_schema.get("name", "tool_plan")

        user = (
            f"Goal: {goal}\n"
            "Create an execution plan using available tools.\n"
            "Return JSON with top-level key 'steps' and follow this schema:\n"
            f"{json.dumps({'name': schema_name, 'schema': schema}, ensure_ascii=False)}"
        )

        # Perplexity JSON-mode support can vary by model; fallback to plain text parsing.
        try:
            resp = self._call(
                input_messages=[
                    {"role": "system", "content": planner_system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
            )
            parsed = self._parse_json_strictish(self._extract_text(resp))
            steps = parsed.get("steps")
            if isinstance(steps, list):
                return {"steps": steps}
        except Exception:
            pass

        resp2 = self._call(
            input_messages=[
                {"role": "system", "content": planner_system},
                {"role": "user", "content": user},
            ],
        )
        parsed2 = self._parse_json_strictish(self._extract_text(resp2))
        steps2 = parsed2.get("steps")
        return {"steps": steps2 if isinstance(steps2, list) else []}

    def clarify_goal(self, *, goal: str) -> Dict[str, Any]:
        schema_hint = {
            "status": "ready|needs_info",
            "goal_summary": "string",
            "normalized_goal": "string",
            "missing_fields": ["string"],
            "questions": ["string"],
        }
        user = (
            f"Anfrage: {goal}\n"
            "Return JSON only with keys: status, goal_summary, normalized_goal, missing_fields, questions.\n"
            f"Schema hint: {json.dumps(schema_hint, ensure_ascii=False)}"
        )

        try:
            resp = self._call(
                input_messages=[
                    {"role": "system", "content": get_clarification_system_prompt("openai")},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
            )
            parsed = self._parse_json_strictish(self._extract_text(resp))
        except Exception:
            parsed = {}

        status = str(parsed.get("status") or "").strip()
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
                {"role": "system", "content": get_final_system_prompt("openai")},
                {"role": "user", "content": f"Goal: {goal}\nTool outputs:\n{tool_outputs}"},
            ],
        )
        return self._extract_text(resp).strip()
