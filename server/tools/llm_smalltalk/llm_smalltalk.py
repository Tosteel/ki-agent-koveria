from __future__ import annotations

from typing import Any, Dict, Optional

from server.services.llm_ionos import IonosLLM


def _fallback_reply(message: str) -> str:
    msg = (message or "").strip()
    if not msg:
        return "Hallo! Wie kann ich dir helfen?"
    if any(w in msg.lower() for w in ("hallo", "hi", "hey", "guten tag", "moin")):
        return "Hallo! Wie kann ich dir helfen?"
    return "Verstanden. Wie kann ich dir weiterhelfen?"


def llm_smalltalk(
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
        out = _fallback_reply(raw)
        if len(out) > max_chars:
            out = out[: max(0, max_chars - 1)].rstrip() + "…"
        return {"reply": out, "text": out, "fallback_used": True, "model": "", "usage": None}

    system = (
        "Du bist ein Assistent für kurze, lockere Unterhaltung. "
        "Antworte direkt an den Nutzer, natürlich und freundlich. "
        "Keine Meta-Beschreibungen über 'der Nutzer' oder 'der Assistent'. "
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
        reply = _fallback_reply(raw)
        fallback_used = True
    else:
        fallback_used = False

    if len(reply) > max_chars:
        reply = reply[: max(0, max_chars - 1)].rstrip() + "…"

    return {
        "reply": reply,
        "text": reply,
        "fallback_used": fallback_used,
        "model": getattr(client.cfg, "model", ""),
        "usage": client.extract_usage(completion),
    }

