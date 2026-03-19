from __future__ import annotations

from typing import Any, Dict


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _calc_end_hour(start_time: str, duration_hours: float) -> int | None:
    raw = str(start_time or "").strip()
    if not raw:
        return None
    try:
        hh = int(raw.split(":", 1)[0])
        return int((hh + duration_hours) % 24)
    except Exception:
        return None


def pricing_compute_quote(
    *,
    facts: Dict[str, Any],
    pricing_rules: Dict[str, Any] | None = None,
    booking_rules: Dict[str, Any] | None = None,
    distance_km: float = 0.0,
) -> Dict[str, Any]:
    ff = dict(facts or {})
    pr = dict(pricing_rules or {})
    br = dict(booking_rules or {})

    duration_hours = max(0.0, _to_float(ff.get("duration_hours"), 0.0))
    hourly_rate = max(0.0, _to_float(pr.get("hourly_rate_eur"), 120.0))
    setup_flat = max(0.0, _to_float(pr.get("setup_flat_eur"), 80.0))
    teardown_flat = max(0.0, _to_float(pr.get("teardown_flat_eur"), 60.0))
    travel_per_km = max(0.0, _to_float(pr.get("travel_per_km_eur"), 0.7))
    overnight_flat = max(0.0, _to_float(pr.get("overnight_flat_eur"), 120.0))
    round_trip = bool(pr.get("travel_round_trip", True))

    overnight_distance_km = max(0.0, _to_float(br.get("overnight_distance_km"), 60.0))
    overnight_after_hour = int(_to_float(br.get("overnight_after_hour"), 22.0))
    start_time = str(ff.get("start_time") or "")
    end_hour = _calc_end_hour(start_time, duration_hours)

    travel_distance = max(0.0, _to_float(distance_km, 0.0))
    if round_trip:
        travel_distance *= 2.0

    labor_cost = duration_hours * hourly_rate
    travel_cost = travel_distance * travel_per_km

    overnight_included = bool(
        _to_float(distance_km, 0.0) > overnight_distance_km
        and end_hour is not None
        and end_hour >= overnight_after_hour
    )
    overnight_cost = overnight_flat if overnight_included else 0.0

    breakdown = {
        "labor_eur": round(labor_cost, 2),
        "setup_eur": round(setup_flat, 2),
        "teardown_eur": round(teardown_flat, 2),
        "travel_eur": round(travel_cost, 2),
        "overnight_eur": round(overnight_cost, 2),
    }
    total = round(sum(breakdown.values()), 2)

    lines = [
        "Preisübersicht (EUR):",
        f"- Arbeitszeit ({duration_hours:.2f} h): {breakdown['labor_eur']:.2f}",
        f"- Aufbau: {breakdown['setup_eur']:.2f}",
        f"- Abbau: {breakdown['teardown_eur']:.2f}",
        f"- Anfahrt: {breakdown['travel_eur']:.2f}",
    ]
    if overnight_included:
        lines.append(f"- Übernachtungspauschale: {breakdown['overnight_eur']:.2f}")
    lines.append(f"Gesamt: {total:.2f} EUR")

    return {
        "total_eur": total,
        "currency": "EUR",
        "breakdown": breakdown,
        "overnight_included": overnight_included,
        "text": "\n".join(lines),
    }
