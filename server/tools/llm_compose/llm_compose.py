from __future__ import annotations

from typing import Any, Dict, Optional

from server.services.llm_ionos import IonosLLM


def _fallback_compose(text: str, max_chars: int) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return "Keine Daten zur Ausformulierung vorhanden."
    out = "\n".join(lines).strip()
    if len(out) > max_chars:
        out = out[: max(0, max_chars - 1)].rstrip() + "…"
    return out


def llm_compose_text(
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
            "fallback_used": True,
            "model": "",
            "usage": None,
        }

    client = llm or IonosLLM()
    if not client.enabled():
        composed = _fallback_compose(raw, max_chars)
        return {
            "text": composed,
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
        "- Schreibe niemals über Systemfähigkeiten oder Tool-Limits (z.B. niemals 'ich kann keine PDF erstellen' oder 'ich kann keine E-Mail senden').\n"
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

    if len(composed) > max_chars:
        composed = composed[: max(0, max_chars - 1)].rstrip() + "…"

    return {
        "text": composed,
        "fallback_used": fallback_used,
        "model": getattr(client.cfg, "model", ""),
        "usage": client.extract_usage(completion),
    }
