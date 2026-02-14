from __future__ import annotations

from typing import Any, Dict


def run_tool(*, text: str, extra: str = "") -> Dict[str, Any]:
    """
    Replace this with the real tool logic.
    Keep return values JSON-serializable.
    """
    out = (text or "").strip()
    if extra.strip():
        out = f"{out} | extra={extra.strip()}"
    return {
        "text": out,
        "ok": True,
    }
