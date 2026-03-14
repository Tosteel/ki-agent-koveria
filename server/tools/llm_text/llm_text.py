from __future__ import annotations

from typing import Any, Dict, Optional

from server.services.llm_ionos import IonosLLM


def _clip(text: str, max_chars: int) -> str:
    out = str(text or "").strip()
    if len(out) > max_chars:
        out = out[: max(0, max_chars - 1)].rstrip() + "…"
    return out


def _fallback_compose(text: str, max_chars: int) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return "Keine Daten zur Ausformulierung vorhanden."
    return _clip("\n".join(lines), max_chars)


def _fallback_summary(text: str, max_chars: int) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return "Keine Daten zur Zusammenfassung vorhanden."
    return _clip(" ".join(lines[:8]), max_chars)


def _fallback_chat(message: str) -> str:
    msg = (message or "").strip()
    if not msg:
        return "Hallo! Wie kann ich dir helfen?"
    if any(w in msg.lower() for w in ("hallo", "hi", "hey", "guten tag", "moin")):
        return "Hallo! Wie kann ich dir helfen?"
    return "Verstanden. Wie kann ich dir weiterhelfen?"


def llm_text_compose(
    text: str,
    *,
    goal: str = "",
    instruction: str = "",
    max_chars: int = 3000,
    llm: Optional[IonosLLM] = None,
) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        empty = "Keine Daten zur Ausformulierung vorhanden."
        return {
            "text": empty,
            "composed_text": empty,
            "fallback_used": True,
            "model": "",
            "usage": None,
        }

    client = llm or IonosLLM()
    if not client.enabled():
        composed = _fallback_compose(raw, max_chars)
        return {
            "text": composed,
            "composed_text": composed,
            "fallback_used": True,
            "model": "",
            "usage": None,
        }

    system = (
        "Du formulierst strukturierte Textbausteine in einen kohärenten, gut lesbaren Fließtext um.\n"
        "Regeln:\n"
        "- Das Nutzerziel ist verbindlich und hat Vorrang.\n"
        "- Nutze ausschließlich Informationen aus dem Input.\n"
        "- Keine Halluzinationen.\n"
        "- Bei Web-/Suchergebnissen darfst du nur Inhalte aus konkreten Matches/Snippets zusammenfassen.\n"
        "- Wenn keine belastbaren Matches/Snippets vorliegen (z.B. Matches: 0 oder nur Navigations-/Boilerplate-Text), schreibe explizit: 'Keine belastbaren Treffer gefunden.'\n"
        "- In diesem Fall keine allgemeinen Hintergrundinformationen aus Vorwissen ergänzen.\n"
        "- Keine Meta-Hinweise zum Prozess ausgeben (z.B. keine Sätze über PDF-Erstellung oder Anfragehinweise).\n"
        "- Schreibe niemals über Systemfähigkeiten oder Tool-Limits.\n"
        "- Liefere ausschließlich die inhaltliche Zusammenfassung der bereitgestellten Daten.\n"
        "- Erhalte Fakten, Zahlen, Namen und Quellenhinweise präzise.\n"
        "- Wenn das Ziel 'mit Quelle' verlangt, integriere Quellen direkt im Text.\n"
        "- Schreibe natürlich, klar und professionell auf Deutsch.\n"
    )
    if instruction.strip():
        system += f"Zusatzanweisung: {instruction.strip()}\n"

    user = (
        f"Ursprüngliches Nutzerziel (bindend):\n{goal.strip() or '(nicht angegeben)'}\n\n"
        f"Forme die folgenden Textbausteine in einen kohärenten, gut lesbaren Text um "
        f"(max. {int(max_chars)} Zeichen):\n\n"
        f"{raw}"
    )

    completion = client.chat_completions(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=1200,
        temperature=0.0,
        top_p=0.1,
    )
    composed = (client.extract_text(completion) or "").strip()
    if not composed:
        composed = _fallback_compose(raw, max_chars)
        fallback_used = True
    else:
        fallback_used = False

    composed = _clip(composed, max_chars)
    return {
        "text": composed,
        "composed_text": composed,
        "fallback_used": fallback_used,
        "model": getattr(client.cfg, "model", ""),
        "usage": client.extract_usage(completion),
    }


def llm_text_summarize(
    text: str,
    *,
    goal: str = "",
    instruction: str = "",
    max_chars: int = 1200,
    llm: Optional[IonosLLM] = None,
) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        empty = "Keine Daten zur Zusammenfassung vorhanden."
        return {
            "summary": empty,
            "text": empty,
            "fallback_used": True,
            "model": "",
            "usage": None,
        }

    client = llm or IonosLLM()
    if not client.enabled():
        summary = _fallback_summary(raw, max_chars)
        return {
            "summary": summary,
            "text": summary,
            "fallback_used": True,
            "model": "",
            "usage": None,
        }

    system = (
        "Du fasst Such- und Retrieval-Ergebnisse präzise zusammen.\n"
        "Regeln:\n"
        "- Das Nutzerziel ist verbindlich und hat Vorrang.\n"
        "- Nur Informationen aus dem Input nutzen.\n"
        "- Keine Halluzinationen.\n"
        "- Keine Meta-Hinweise zum Prozess ausgeben.\n"
        "- Preise/Beträge, Mengen und Quellen priorisieren.\n"
        "- Wenn im Ziel 'mit Quelle' oder 'inkl. Quelle' steht: jede Kernaussage mit Quelle ausgeben.\n"
        "- Falls eine Quelle gefordert ist, aber fehlt: 'Quelle nicht gefunden' klar benennen.\n"
        "- Antworte auf Deutsch, kompakt, klar und umsetzbar.\n"
    )
    if instruction.strip():
        system += f"Zusatzanweisung: {instruction.strip()}\n"

    user = (
        f"Ursprüngliches Nutzerziel (bindend):\n{goal.strip() or '(nicht angegeben)'}\n\n"
        f"Fasse folgenden Text zusammen (max. {int(max_chars)} Zeichen):\n\n"
        f"{raw}"
    )

    completion = client.chat_completions(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=700,
        temperature=0.0,
        top_p=0.1,
    )
    summary = (client.extract_text(completion) or "").strip()
    if not summary:
        summary = _fallback_summary(raw, max_chars)
        fallback_used = True
    else:
        fallback_used = False

    summary = _clip(summary, max_chars)
    return {
        "summary": summary,
        "text": summary,
        "fallback_used": fallback_used,
        "model": getattr(client.cfg, "model", ""),
        "usage": client.extract_usage(completion),
    }


def llm_text_chat(
    *,
    message: str,
    tone: str = "freundlich",
    max_chars: int = 280,
    llm: Optional[IonosLLM] = None,
) -> Dict[str, Any]:
    raw = (message or "").strip()
    if not raw:
        out = "Hallo! Wie kann ich dir helfen?"
        return {"reply": out, "text": out, "fallback_used": True, "model": "", "usage": None}

    client = llm or IonosLLM()
    if not client.enabled():
        out = _clip(_fallback_chat(raw), max_chars)
        return {"reply": out, "text": out, "fallback_used": True, "model": "", "usage": None}

    system = (
        "Du bist ein Assistent für kurze, lockere Unterhaltung. "
        "Antworte direkt an den Nutzer, natürlich und freundlich. "
        "Keine Meta-Beschreibungen über Rollen. "
        "Wenn keine konkrete Aufgabe vorliegt, stelle höchstens eine kurze Rückfrage. "
        "Antworte auf Deutsch und knapp."
    )
    user = (
        f"Ton: {tone.strip() or 'freundlich'}\n"
        f"Nutzer-Nachricht: {raw}\n"
        f"Formuliere eine kurze Antwort (max. {int(max_chars)} Zeichen)."
    )

    completion = client.chat_completions(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=220,
        temperature=0.4,
        top_p=0.9,
    )
    reply = (client.extract_text(completion) or "").strip()
    if not reply:
        reply = _fallback_chat(raw)
        fallback_used = True
    else:
        fallback_used = False

    reply = _clip(reply, max_chars)
    return {
        "reply": reply,
        "text": reply,
        "fallback_used": fallback_used,
        "model": getattr(client.cfg, "model", ""),
        "usage": client.extract_usage(completion),
    }

