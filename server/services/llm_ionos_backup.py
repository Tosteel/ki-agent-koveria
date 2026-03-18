from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests


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
        # Header wird pro Request ergänzt; aber Session-Defaults helfen.
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
            response_format: Optional[Dict[str, Any]] = None,  # <-- NEU
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

        if response_format is not None:
            payload[
                "response_format"] = response_format  # <-- NEU (IONOS unterstützt das) :contentReference[oaicite:1]{index=1}

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
                # einfacher Backoff
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
    # "Agent runtime"-nahe Hilfsfunktionen: plan_steps + final_answer
    # ---------------------------------------------------------------------

    def plan_steps(self, *, goal: str, tool_schema: Dict[str, Any]) -> Dict[str, Any]:
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": tool_schema.get("name", "tool_plan"),
                "schema": tool_schema.get("schema", tool_schema),
                "strict": True,
            },
        }

        completion = self.chat_completions(
            messages=[
                {"role": "system", "content": "You are a planner. Output ONLY JSON that matches the schema."},
                {"role": "user", "content": f"Goal: {goal}"},
            ],
            response_format=response_format,
        )

        text = self.extract_text(completion)
        parsed = _parse_json_strictish(text)
        steps = parsed.get("steps") or []
        if not isinstance(steps, list):
            steps = []

        return {"steps": steps}

    def _plan_steps_with_allowed_tools(
            self,
            *,
            goal: str,
            allowed_tools: List[str],
    ) -> List[Dict[str, Any]]:
        system = (
            "Du bist ein Planner.\n"
            "Erzeuge einen Ausführungsplan als reines JSON und nichts anderes.\n\n"

            "Allgemeine Regeln:\n"
            "- Antworte AUSSCHLIESSLICH mit gültigem JSON.\n"
            "- Halte dich strikt an die vorgegebenen Tool-Namen.\n"
            "- Verwende keine zusätzlichen Felder.\n\n"

            "Tool-Struktur:\n"
            '{ "steps": [ { "tool": "<tool-name>", "args": { ... } } ] }\n\n'

            "Tool-Auswahl-Regeln:\n"
            "- Wenn Fakten/Informationen gesucht werden (z.B. Preise, Definitionen, Inhalte aus Dokumenten): nutze zuerst rag_knowledgebase.\n"
            "- Nutze file_read nur, wenn der Nutzer explizit eine konkrete Datei genannt hat ODER wenn ein vorheriger Schritt den Dateinamen geliefert hat.\n"
            "- Wenn rag_knowledgebase Ergebnisse liefert: nutze die Snippets für die Antwort oder schreibe eine Zusammenfassung in eine Datei.\n"
            "- Wenn der Nutzer keine konkrete Datei nennt: verwende KEIN file_read. In diesem Fall MUSS rag_knowledgebase genutzt werden.\n"
            "- Formuliere rag_knowledgebase-Queries als Suchanfrage (keine SQL). Beispiel: 'Deckenabhängung Preis €/m²'.\n"
            "- Für Export: nutze pdf_export.\n"


            "Pfad-Regeln (sehr wichtig):\n"
            "- Alle Pfade sind RELATIV zum Work-Verzeichnis des Nutzers.\n"
            "- Verwende KEIN Präfix wie 'user/work', 'work/' oder '/'.\n"
            "- Beispiel korrekt: 'decke.txt', 'exports/plan.pdf'\n"
            "- Beispiel falsch: 'user/work/decke.txt'\n\n"

            f"Erlaubte Tools: {', '.join(allowed_tools)}\n"
        )

        user = f"Ziel: {goal}"

        completion = self.chat_completions(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )

        text = self.extract_text(completion)
        parsed = _parse_json_strictish(text)

        steps = parsed.get("steps", [])
        if not isinstance(steps, list):
            return []

        # Sicherheitsfilter
        cleaned = []
        for s in steps:
            if (
                    isinstance(s, dict)
                    and s.get("tool") in allowed_tools
                    and isinstance(s.get("args", {}), dict)
            ):
                cleaned.append(s)

        return cleaned

    def final_answer(self, *, goal: str, tool_outputs: List[Dict[str, Any]]) -> str:
        """
        Erzeugt eine finale Antwort auf Basis der Tool-Outputs.
        """
        system = (
            "Du bist ein Assistent. Antworte sachlich und knapp.\n"
            "Nutze ausschließlich die Tool-Outputs. Erfinde nichts.\n"
            "Wenn Daten fehlen: benenne das klar.\n"
        )
        user = (
            f"Ziel: {goal}\n\n"
            f"Tool-Outputs (JSON):\n{json.dumps(tool_outputs, ensure_ascii=False)}\n\n"
            "Erstelle eine kurze Zusammenfassung."
        )
        completion = self.chat_completions(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return self.extract_text(completion)


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

    # Entferne ```...``` falls vorhanden
    if t.startswith("```"):
        # entferne erste Zeile ``` oder ```json
        lines = t.splitlines()
        if len(lines) >= 2:
            # drop first and last fence if present
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            t = "\n".join(lines).strip()

    # Versuch 1: direkt
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    # Versuch 2: erstes {...} extrahieren
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = t[start: end + 1]
        try:
            obj = json.loads(snippet)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    return {}
