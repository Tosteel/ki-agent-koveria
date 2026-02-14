from __future__ import annotations

import os
from typing import Dict, Optional


_PROMPTS: Dict[str, Dict[str, Dict[str, str]]] = {
    "default": {
        "ionos": {
            "planner_system": (
                "You are a planner. Output ONLY valid JSON with a top-level key 'steps'.\n"
                "read_file ist verboten, wenn der Nutzer keine Datei benennt. In diesem Fall MUSS rag_knowledgebase genutzt werden.\n"
                "Formuliere für rag_knowledgebase-Queries die Suchbegriffe (keine SQL).\n"
                "Wenn der Nutzer eine Zusammenfassung/kompakte Ausgabe verlangt, nutze llm_summarize nach rag_knowledgebase und vor pdf_export.\n"
                "Wenn der Nutzer einen kohärenten, gut lesbaren Fließtext aus Bausteinen verlangt, nutze llm_compose nach rag_knowledgebase und vor pdf_export.\n"
                "Für Präsentations-Export nutze ppt_export (statt pdf_export).\n"
                "Für Websuche/Internet-Recherche aus einem User-Prompt nutze search_multitable.\n"
                "Nachfolgende Schritte erhalten automatisch den Payload des vorherigen Schritts als zusätzliche Args.\n"
                "Plane daher so, dass Ergebnisfelder (z.B. text) von Schritt N direkt von Schritt N+1 genutzt werden können.\n"
                "Platzhalter nur als {steps[0].text}, {{steps.1.result.text}} oder {last.text}; niemals mit führendem $.\n"
            ),
            "final_system": (
                "Du bist ein Assistent. Antworte sachlich und knapp.\n"
                "Nutze ausschließlich die Tool-Outputs. Erfinde nichts.\n"
                "Wenn Daten fehlen: benenne das klar.\n"
            ),
        },
        "openai": {
            "planner_system": (
                "You are a planner. Produce ONLY JSON matching the schema. "
                "Later steps automatically receive the payload/result fields from the previous step as additional arguments. "
                "Use search_multitable for internet/web research from a user prompt."
            ),
            "final_system": "You are an assistant. Use the tool outputs to answer the goal succinctly.",
        },
    },
}


def _provider_key(provider: str) -> str:
    p = (provider or "").strip().lower()
    if p in {"ionos", "openai"}:
        return p
    return "ionos"


def _resolve_variant(provider: str, variant: Optional[str]) -> str:
    p = _provider_key(provider).upper()
    requested = (variant or "").strip()
    if not requested:
        requested = os.getenv(f"{p}_AGENT_PROMPT_VARIANT", "").strip()
    if not requested:
        requested = os.getenv("AGENT_PROMPT_VARIANT", "default").strip()
    if requested in _PROMPTS:
        return requested
    return "default"


def get_planner_system_prompt(provider: str, variant: Optional[str] = None) -> str:
    p = _provider_key(provider)
    v = _resolve_variant(provider, variant)
    return _PROMPTS[v][p]["planner_system"]


def get_final_system_prompt(provider: str, variant: Optional[str] = None) -> str:
    p = _provider_key(provider)
    v = _resolve_variant(provider, variant)
    return _PROMPTS[v][p]["final_system"]
