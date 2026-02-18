from __future__ import annotations

import os
from typing import Any, Dict, List

import requests
from fastapi import HTTPException


def _pick_str(entry: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        val = entry.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _pick_float(entry: Dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        val = entry.get(key)
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            try:
                return float(val)
            except ValueError:
                continue
    return None


def _extract_items(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Native LangSearch shape:
    # {"code":200,"data":{"webPages":{"value":[...]}}}
    data = payload.get("data")
    if isinstance(data, dict):
        web_pages = data.get("webPages")
        if isinstance(web_pages, dict):
            value = web_pages.get("value")
            if isinstance(value, list):
                out: List[Dict[str, Any]] = []
                for entry in value:
                    if not isinstance(entry, dict):
                        continue
                    out.append(
                        {
                            "title": _pick_str(entry, "name", "title"),
                            "url": _pick_str(entry, "url", "link", "href"),
                            "snippet": _pick_str(entry, "snippet", "summary", "text"),
                            "score": _pick_float(entry, "score", "relevance"),
                            "raw": dict(entry),
                        }
                    )
                return out

    for key in ("results", "items", "data", "hits"):
        raw = payload.get(key)
        if isinstance(raw, list):
            out: List[Dict[str, Any]] = []
            for entry in raw:
                if not isinstance(entry, dict):
                    continue
                out.append(
                    {
                        "title": _pick_str(entry, "title", "name"),
                        "url": _pick_str(entry, "url", "link", "href"),
                        "snippet": _pick_str(entry, "snippet", "summary", "content", "text"),
                        "score": _pick_float(entry, "score", "relevance"),
                        "raw": dict(entry),
                    }
                )
            return out
    return []


def _extract_summary(payload: Dict[str, Any]) -> str:
    data = payload.get("data")
    if isinstance(data, dict):
        for key in ("summary", "answer", "text"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    for key in ("summary", "answer", "text"):
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _result_to_text(query: str, items: List[Dict[str, Any]], summary_text: str) -> str:
    lines = [f"LangSearch Query: {query}", f"Treffer: {len(items)}", ""]
    if summary_text:
        lines.append("Summary:")
        lines.append(summary_text)
        lines.append("")
    if not items:
        lines.append("Keine Treffer gefunden.")
        return "\n".join(lines).strip()

    for i, item in enumerate(items, start=1):
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        snippet = str(item.get("snippet") or "").strip()
        lines.append(f"[{i}] {title}")
        if url:
            lines.append(f"URL: {url}")
        if snippet:
            lines.append(f"Snippet: {snippet}")
        lines.append("")
    return "\n".join(lines).strip()


def search_langsearch(
    *,
    query: str,
    count: int = 5,
    summary: bool = True,
    freshness: str | None = None,
) -> Dict[str, Any]:
    base = os.getenv("LANGSEARCH_API_BASE", "https://api.langsearch.com").strip().rstrip("/")
    api_key = os.getenv("LANGSEARCH_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=500, detail="LANGSEARCH_API_KEY is missing")

    payload: Dict[str, Any] = {
        "query": query,
        "count": max(1, min(int(count), 20)),
        "summary": bool(summary),
    }
    if freshness:
        payload["freshness"] = freshness

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    try:
        resp = requests.post(
            f"{base}/v1/web-search",
            json=payload,
            headers=headers,
            timeout=45,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"LangSearch request failed: {exc}") from exc

    if not isinstance(data, dict):
        data = {}
    items = _extract_items(data)
    summary_text = _extract_summary(data)
    return {
        "query": query,
        "count": len(items),
        "items": items,
        "summary_text": summary_text,
        "text": _result_to_text(query, items, summary_text),
    }
