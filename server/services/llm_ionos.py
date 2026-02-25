from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from .agent_prompts import (
    get_clarification_system_prompt,
    get_final_system_prompt,
    get_planner_system_prompt,
)

def _env_int(name: str, default: int) -> int:
    v = os.getenv(name, "").strip()
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name, "").strip()
    if not v:
        return default
    try:
        return float(v)
    except ValueError:
        return default


@dataclass(frozen=True)
class IonosConfig:
    api_base: str
    api_key: str
    model: str
    max_tokens: int
    temperature: float
    top_p: float
    timeout_s: int = 120
    retries: int = 2


class IonosLLM:
    """
    IONOS Inference (OpenAI-compatible) client via /v1/chat/completions.

    Env:
      IONOS_API_BASE=https://openai.inference.de-txl.ionos.com/v1
      IONOS_API_KEY=...
      IONOS_MODEL=meta-llama/Llama-3.3-70B-Instruct
      MAX_TOKENS=400
      TEMPERATURE=0
      TOP_P=0.1
    """

    def __init__(self, cfg: Optional[IonosConfig] = None):
        if cfg is None:
            cfg = IonosConfig(
                api_base=os.getenv("IONOS_API_BASE", "https://openai.inference.de-txl.ionos.com/v1").rstrip("/"),
                api_key=os.getenv("IONOS_API_KEY", "").strip(),
                model=os.getenv("IONOS_MODEL", "meta-llama/Llama-3.3-70B-Instruct").strip(),
                max_tokens=_env_int("MAX_TOKENS", 400),
                temperature=_env_float("TEMPERATURE", 0.0),
                top_p=_env_float("TOP_P", 0.1),
                timeout_s=_env_int("IONOS_TIMEOUT_S", 120),
                retries=_env_int("IONOS_RETRIES", 2),
            )
        self.cfg = cfg

        self.session = requests.Session()
        if self.cfg.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.cfg.api_key}"})
        self.session.headers.update({"Content-Type": "application/json"})

    def enabled(self) -> bool:
        return bool(self.cfg.api_key and self.cfg.api_base and self.cfg.model)

    def chat_completions(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        timeout_s: Optional[int] = None,
        retries: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.enabled():
            raise RuntimeError("IONOS is not configured (missing IONOS_API_KEY / IONOS_API_BASE / IONOS_MODEL).")

        url = f"{self.cfg.api_base}/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "max_tokens": int(max_tokens if max_tokens is not None else self.cfg.max_tokens),
            "temperature": float(temperature if temperature is not None else self.cfg.temperature),
            "top_p": float(top_p if top_p is not None else self.cfg.top_p),
        }

        # IONOS: response_format unterstützt json_object und json_schema (Structured Outputs)
        if response_format is not None:
            payload["response_format"] = response_format

        timeout = int(timeout_s if timeout_s is not None else self.cfg.timeout_s)
        n_retries = int(retries if retries is not None else self.cfg.retries)

        last_err: Optional[Exception] = None
        for attempt in range(n_retries + 1):
            try:
                resp = self.session.post(url, json=payload, timeout=timeout)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                last_err = e
                time.sleep(0.8 * (attempt + 1))

        raise RuntimeError(f"IONOS chat request failed: {last_err}")

    @staticmethod
    def extract_text(completion: Dict[str, Any]) -> str:
        try:
            return completion["choices"][0]["message"]["content"].strip()
        except Exception:
            return ""

    @staticmethod
    def extract_usage(completion: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            return completion.get("usage")
        except Exception:
            return None

    # ---------------------------------------------------------------------
    # Agent runtime: plan_steps + final_answer
    # ---------------------------------------------------------------------

    def plan_steps(self, *, goal: str, tool_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Structured Output via response_format=json_schema.
        tool_schema erwartet Format:
          {"name": "...", "schema": {...}}  oder direkt {"schema": {...}}.
        """
        schema_obj = tool_schema.get("schema", tool_schema)
        schema_obj = _sanitize_schema_for_ionos(schema_obj)
        schema_obj = _make_ionos_planner_schema(schema_obj)
        schema_name = tool_schema.get("name", "tool_plan")

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": schema_obj,
                "strict": False,
            },
        }

        planner_system = get_planner_system_prompt("ionos")

        # Primär: json_schema erzwingen
        try:
            completion = self.chat_completions(
                messages=[
                    {"role": "system", "content": planner_system},
                    {"role": "user", "content": f"Goal: {goal}"},
                ],
                response_format=response_format,
            )

            text = self.extract_text(completion)
            parsed = _parse_json_strictish(text)
            steps = parsed.get("steps") or []
            if isinstance(steps, list) and steps:
                return {"steps": steps}
        except Exception as e:
            print("\n===== IONOS PLANNER FALLBACK =====")
            print("json_schema planning failed; switching to json_object fallback")
            print(f"error={e}")
            print("===================================\n")
            # Fallback below (json_object) instead of bubbling up as 500
            pass

        # Fallback: wenigstens JSON erzwingen (falls json_schema/oneOf nicht sauber unterstützt wird)
        completion2 = self.chat_completions(
            messages=[
                {"role": "system", "content": planner_system},
                {"role": "user", "content": f"Goal: {goal}"},
            ],
            response_format={"type": "json_object"},
        )
        text2 = self.extract_text(completion2)
        parsed2 = _parse_json_strictish(text2)
        steps2 = parsed2.get("steps") or []
        return {"steps": steps2 if isinstance(steps2, list) else []}

    def final_answer(self, *, goal: str, tool_outputs: List[Dict[str, Any]]) -> str:
        system = get_final_system_prompt("ionos")
        user = (
            f"Ziel: {goal}\n\n"
            f"Tool-Outputs (JSON):\n{json.dumps(tool_outputs, ensure_ascii=False)}\n\n"
            "Erstelle eine kurze Zusammenfassung."
        )
        completion = self.chat_completions(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return self.extract_text(completion)

    def clarify_goal(self, *, goal: str) -> Dict[str, Any]:
        schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "clarification_gate",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "status": {"type": "string", "enum": ["ready", "needs_info"]},
                        "normalized_goal": {"type": "string"},
                        "missing_fields": {"type": "array", "items": {"type": "string"}},
                        "questions": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["status", "normalized_goal", "missing_fields", "questions"],
                },
                "strict": True,
            },
        }
        completion = self.chat_completions(
            messages=[
                {
                    "role": "system",
                    "content": get_clarification_system_prompt("ionos"),
                },
                {"role": "user", "content": f"Anfrage: {goal}"},
            ],
            response_format=schema,
        )
        parsed = _parse_json_strictish(self.extract_text(completion))
        status = parsed.get("status")
        if status not in {"ready", "needs_info"}:
            status = "ready"
        return {
            "status": status,
            "normalized_goal": str(parsed.get("normalized_goal") or goal),
            "missing_fields": list(parsed.get("missing_fields") or []),
            "questions": list(parsed.get("questions") or []),
        }


def _parse_json_strictish(text: str) -> Dict[str, Any]:
    """
    Robust gegen:
    - führendes/trailing Text
    - JSON in einem Codeblock
    - einfache Formatfehler durch zusätzliche Zeichen
    """
    if not text:
        return {}

    t = text.strip()

    # Codeblock entfernen
    if t.startswith("```"):
        t = t.strip("`").strip()
        # oft steht 'json' in der ersten Zeile
        if "\n" in t:
            first, rest = t.split("\n", 1)
            if first.strip().lower() in ("json", "javascript"):
                t = rest.strip()

    # Direkt versuchen
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    # Fallback: erstes {...} herausschneiden
    start = t.find("{")
    end = t.rfind("}")
    if start >= 0 and end > start:
        snippet = t[start : end + 1]
        try:
            obj = json.loads(snippet)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    return {}


def _sanitize_schema_for_ionos(schema: Any) -> Any:
    """
    IONOS' OpenAI-compatible endpoint can be strict about JSON-Schema size/keywords.
    Keep semantic structure, drop verbose/meta keys.
    """
    if isinstance(schema, dict):
        out: Dict[str, Any] = {}
        for k, v in schema.items():
            if k in {"description", "title", "examples", "default"}:
                continue
            out[k] = _sanitize_schema_for_ionos(v)
        return out
    if isinstance(schema, list):
        return [_sanitize_schema_for_ionos(x) for x in schema]
    return schema


def _make_ionos_planner_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert verbose oneOf planner schemas to a compact IONOS-friendly variant:
    steps[].tool as enum + free-form args object.
    """
    try:
        one_of = (
            schema.get("properties", {})
            .get("steps", {})
            .get("items", {})
            .get("oneOf", [])
        )
        tools: List[str] = []
        for item in one_of:
            if not isinstance(item, dict):
                continue
            name = (
                item.get("properties", {})
                .get("tool", {})
                .get("const")
            )
            if isinstance(name, str) and name.strip():
                tools.append(name.strip())
        tools = sorted(set(tools))
        if not tools:
            return schema

        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "tool": {"type": "string", "enum": tools},
                            "args": {"type": "object", "additionalProperties": True},
                        },
                        "required": ["tool", "args"],
                    },
                }
            },
            "required": ["steps"],
        }
    except Exception:
        return schema
