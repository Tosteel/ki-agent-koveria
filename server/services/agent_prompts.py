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
                "Bei lockeren Begrüßungen/Smalltalk ohne konkrete Aufgabe nutze llm_smalltalk (statt rag_knowledgebase).\n"
                "Wenn der Nutzer explizit nach einer Zusammenfassung des bisherigen Dialogs fragt, nutze NICHT llm_smalltalk.\n"
                "Nutze dafür llm_compose mit dem bereitgestellten Dialogkontext.\n"
                "Wenn der Nutzer eine Zusammenfassung/kompakte Ausgabe verlangt, nutze llm_summarize nach rag_knowledgebase und vor pdf_export.\n"
                "Wenn der Nutzer einen kohärenten, gut lesbaren Fließtext aus Bausteinen verlangt, nutze llm_compose nach rag_knowledgebase und vor pdf_export.\n"
                "Für Präsentations-Export nutze ppt_export (statt pdf_export).\n"
                "Für Websuche/Internet-Recherche aus einem User-Prompt nutze search_multitable.\n"
                "Für Produktsuche auf eBay nutze search_ebay.\n"
                "Für gezielte Inhalts-Suche auf einer konkreten Website-URL nutze view_website.\n"
                "Wenn Klicks/Navigation über mehrere Unterseiten nötig sind, nutze browse_website.\n"
                "Wenn der Nutzer nach Fähigkeiten/Tools des Agenten fragt, nutze list_skills.\n"
                "Nachfolgende Schritte erhalten automatisch den Payload des vorherigen Schritts als zusätzliche Args.\n"
                "Plane daher so, dass Ergebnisfelder (z.B. text) von Schritt N direkt von Schritt N+1 genutzt werden können.\n"
                "Platzhalter nur als {steps[0].text}, {{steps.1.result.text}} oder {last.text}; niemals mit führendem $.\n"
            ),
            "final_system": (
                "Du bist ein Assistent. Antworte sachlich und knapp.\n"
                "Nutze ausschließlich die Tool-Outputs. Erfinde nichts.\n"
                "Wenn Daten fehlen: benenne das klar.\n"
            ),
            "clarification_system": (
                "Du bist ein Clarification-Gate für Tool-Ausführung. "
                "WICHTIG: Bei Such-/Recherche-/Wissensfragen (z.B. 'suche', 'recherchiere', 'in meinem Wissen') "
                "immer status=ready, auch wenn Details fehlen. Dann best-effort ausführen. "
                "Rückfragen (status=needs_info) nur wenn KEIN Tool sinnvoll startbar ist, weil zwingende Pflichtfelder fehlen. "
                "Beispiele für needs_info: E-Mail ohne Empfänger, browse_website ohne URL. "
                "Beispiele für ready: Wissenssuche, Webrecherche, allgemeine Analyse mit unvollständigen Filtern."
            ),
        },
        "openai": {
            "planner_system": (
                "You are a planner. Produce ONLY JSON matching the schema. "
                "Later steps automatically receive the payload/result fields from the previous step as additional arguments. "
                "Use llm_smalltalk for greetings/casual smalltalk without a concrete task. "
                "If the user asks for a summary of the dialogue so far, do not use llm_smalltalk; use llm_compose with provided dialogue context. "
                "Use search_multitable for internet/web research from a user prompt. "
                "Use search_ebay for product search on eBay. "
                "Use view_website for targeted content lookup on a specific website URL. "
                "Use browse_website when click/navigation across subpages is required. "
                "Use list_skills when the user asks about agent capabilities/tools."
            ),
            "final_system": "You are an assistant. Use the tool outputs to answer the goal succinctly.",
            "clarification_system": (
                "You are a clarification gate for tool execution. "
                "IMPORTANT: For search/research/knowledge requests (e.g. 'search', 'research', 'in my knowledge'), "
                "always return status=ready even if details are missing, then proceed best-effort. "
                "Return status=needs_info only when no tool can be started due to mandatory missing fields. "
                "Examples for needs_info: email without recipient, browse_website without URL. "
                "Examples for ready: knowledge retrieval, web research, analysis with incomplete filters."
            ),
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


def get_clarification_system_prompt(provider: str, variant: Optional[str] = None) -> str:
    p = _provider_key(provider)
    v = _resolve_variant(provider, variant)
    return _PROMPTS[v][p]["clarification_system"]
