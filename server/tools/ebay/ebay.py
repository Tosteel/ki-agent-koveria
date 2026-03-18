from __future__ import annotations

import os
from typing import Any, Dict, List

import requests
from fastapi import HTTPException


def _extract_search_items(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    root = payload.get("findItemsByKeywordsResponse")
    if not isinstance(root, list) or not root:
        return []

    search_result = root[0].get("searchResult")
    if not isinstance(search_result, list) or not search_result:
        return []

    items = search_result[0].get("item")
    if not isinstance(items, list):
        return []

    out: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue

        title = ""
        view_url = ""
        item_id = ""
        price = ""
        currency = ""
        location = ""

        t = item.get("title")
        if isinstance(t, list) and t:
            title = str(t[0])

        u = item.get("viewItemURL")
        if isinstance(u, list) and u:
            view_url = str(u[0])

        i = item.get("itemId")
        if isinstance(i, list) and i:
            item_id = str(i[0])

        l = item.get("location")
        if isinstance(l, list) and l:
            location = str(l[0])

        selling_status = item.get("sellingStatus")
        if isinstance(selling_status, list) and selling_status:
            current_price = selling_status[0].get("currentPrice")
            if isinstance(current_price, list) and current_price:
                cp0 = current_price[0]
                if isinstance(cp0, dict):
                    price = str(cp0.get("__value__") or "")
                    currency = str(cp0.get("@currencyId") or "")

        out.append(
            {
                "item_id": item_id,
                "title": title,
                "price": price,
                "currency": currency,
                "location": location,
                "url": view_url,
            }
        )

    return out


def ebay_search(*, query: str, limit: int = 10, sort_order: str = "BestMatch") -> Dict[str, Any]:
    app_id = os.getenv("EBAY_APP_ID", "").strip()
    if not app_id:
        raise HTTPException(status_code=500, detail="EBAY_APP_ID is missing")

    limit = max(1, min(int(limit), 50))

    params = {
        "OPERATION-NAME": "findItemsByKeywords",
        "SERVICE-VERSION": "1.13.0",
        "SECURITY-APPNAME": app_id,
        "RESPONSE-DATA-FORMAT": "JSON",
        "REST-PAYLOAD": "",
        "keywords": query,
        "paginationInput.entriesPerPage": str(limit),
        "sortOrder": sort_order,
    }

    try:
        resp = requests.get(
            "https://svcs.ebay.com/services/search/FindingService/v1",
            params=params,
            timeout=20,
        )
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"eBay request failed: {exc}") from exc

    items = _extract_search_items(payload)
    return {
        "query": query,
        "count": len(items),
        "items": items,
    }
