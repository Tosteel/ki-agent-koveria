from __future__ import annotations

from typing import Any, Dict, Optional

from server.services.llm_ionos import IonosLLM


def _fallback_summary(text: str, max_chars: int) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return "Keine Daten zur Zusammenfassung vorhanden."
    out = " ".join(lines[:8]).strip()
    if len(out) > max_chars:
        out = out[: max(0, max_chars - 1)].rstrip() + "…"
    return out


def llm_summarize_text(
    text: str,
    *,
    goal: str = "",
    instruction: str = "",
    max_chars: int = 1200,
    llm: Optional[IonosLLM] = None,
) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {
            "summary": "Keine Daten zur Zusammenfassung vorhanden.",
            "text": "Keine Daten zur Zusammenfassung vorhanden.",
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

    if len(summary) > max_chars:
        summary = summary[: max(0, max_chars - 1)].rstrip() + "…"

    return {
        "summary": summary,
        "text": summary,
        "fallback_used": fallback_used,
        "model": getattr(client.cfg, "model", ""),
        "usage": client.extract_usage(completion),
    }
