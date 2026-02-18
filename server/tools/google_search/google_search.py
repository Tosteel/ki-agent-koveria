from __future__ import annotations

import os
from typing import Any, Dict, List

import requests
from fastapi import HTTPException


def _to_int(value: Any) -> int | None:
    try:
        return int(str(value))
    except Exception:
        return None


def _extract_items(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        return []

    items: List[Dict[str, Any]] = []
    for entry in raw_items:
        if not isinstance(entry, dict):
            continue
        items.append(
            {
                "title": str(entry.get("title") or ""),
                "link": str(entry.get("link") or ""),
                "snippet": str(entry.get("snippet") or ""),
                "display_link": str(entry.get("displayLink") or ""),
            }
        )
    return items


def _result_to_text(query: str, items: List[Dict[str, Any]]) -> str:
    lines = [f"Google Query: {query}", f"Treffer: {len(items)}", ""]
    if not items:
        lines.append("Keine Treffer gefunden.")
        return "\n".join(lines).strip()

    for idx, item in enumerate(items, start=1):
        title = str(item.get("title") or "").strip()
        link = str(item.get("link") or "").strip()
        snippet = str(item.get("snippet") or "").strip()
        lines.append(f"[{idx}] {title}")
        if link:
            lines.append(f"URL: {link}")
        if snippet:
            lines.append(f"Snippet: {snippet}")
        lines.append("")
    return "\n".join(lines).strip()


def search_google_custom(
    *,
    query: str,
    num: int = 5,
    start: int = 1,
    gl: str | None = None,
    hl: str | None = None,
    safe: str | None = None,
    site_search: str | None = None,
) -> Dict[str, Any]:
    api_key = os.getenv("GOOGLE_CSE_API_KEY", "").strip()
    cx = os.getenv("GOOGLE_CSE_CX", "").strip()

    if not api_key:
        raise HTTPException(status_code=500, detail="GOOGLE_CSE_API_KEY is missing")
    if not cx:
        raise HTTPException(status_code=500, detail="GOOGLE_CSE_CX is missing")

    params: Dict[str, Any] = {
        "key": api_key,
        "cx": cx,
        "q": query,
        "num": max(1, min(int(num), 10)),
        "start": max(1, min(int(start), 91)),
    }
    if gl:
        params["gl"] = gl
    if hl:
        params["hl"] = hl
    if safe:
        params["safe"] = safe
    if site_search:
        params["siteSearch"] = site_search

    try:
        resp = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Google Custom Search request failed: {exc}") from exc

    items = _extract_items(payload if isinstance(payload, dict) else {})
    total_results = None
    if isinstance(payload, dict):
        search_info = payload.get("searchInformation")
        if isinstance(search_info, dict):
            total_results = _to_int(search_info.get("totalResults"))

    return {
        "query": query,
        "count": len(items),
        "total_results": total_results,
        "items": items,
        "text": _result_to_text(query, items),
    }

