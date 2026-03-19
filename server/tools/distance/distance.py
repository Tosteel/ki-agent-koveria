from __future__ import annotations

import math
from typing import Dict, Tuple

import requests
from fastapi import HTTPException


_CITY_FALLBACK: Dict[str, Tuple[float, float]] = {
    "pforzheim": (48.892186, 8.694628),
    "stuttgart": (48.775846, 9.182932),
    "karlsruhe": (49.006890, 8.403653),
    "heilbronn": (49.142692, 9.210879),
    "muenchen": (48.137154, 11.576124),
    "munich": (48.137154, 11.576124),
    "frankfurt": (50.110924, 8.682127),
    "berlin": (52.520008, 13.404954),
}


def _normalize_key(value: str) -> str:
    return (
        str(value or "")
        .strip()
        .lower()
        .replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("ß", "ss")
    )


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def _geocode_nominatim(query: str) -> Tuple[float, float] | None:
    q = str(query or "").strip()
    if not q:
        return None
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": q, "format": "json", "limit": 1},
            headers={"User-Agent": "ki-agent-koveria/1.0"},
            timeout=10,
        )
    except requests.RequestException:
        return None
    if resp.status_code >= 400:
        return None
    try:
        data = resp.json()
    except Exception:
        return None
    if not isinstance(data, list) or not data:
        return None
    first = data[0] if isinstance(data[0], dict) else {}
    try:
        lat = float(first.get("lat"))
        lon = float(first.get("lon"))
    except Exception:
        return None
    return lat, lon


def _geocode(query: str) -> Tuple[float, float]:
    geo = _geocode_nominatim(query)
    if geo is not None:
        return geo

    key = _normalize_key(query)
    if key in _CITY_FALLBACK:
        return _CITY_FALLBACK[key]

    raise HTTPException(status_code=422, detail=f"Geocoding failed for location: {query}")


def distance_check(*, origin: str, destination: str, max_distance_km: float = 0.0) -> Dict[str, object]:
    o = str(origin or "").strip()
    d = str(destination or "").strip()
    if not o or not d:
        raise HTTPException(status_code=422, detail="origin and destination are required")

    o_lat, o_lon = _geocode(o)
    d_lat, d_lon = _geocode(d)

    linear = _haversine_km(o_lat, o_lon, d_lat, d_lon)
    # Approximate realistic road distance.
    road_km = round(linear * 1.22, 2)
    duration_minutes = int(round((road_km / 70.0) * 60.0))

    max_km = max(0.0, float(max_distance_km or 0.0))
    within_limit = True if max_km <= 0 else road_km <= max_km
    text = (
        f"Distanz {road_km:.2f} km, ca. {duration_minutes} min Fahrzeit."
        if max_km <= 0
        else f"Distanz {road_km:.2f} km, ca. {duration_minutes} min. Limit {max_km:.2f} km: {'ok' if within_limit else 'überschritten'}."
    )

    return {
        "origin": o,
        "destination": d,
        "distance_km": road_km,
        "estimated_duration_minutes": duration_minutes,
        "within_limit": within_limit,
        "text": text,
    }
