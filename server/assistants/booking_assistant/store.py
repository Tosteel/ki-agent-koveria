from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from server.core.settings import Settings


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _assistant_dir(settings: Settings, user_id: str) -> Path:
    return settings.user_dir(user_id) / "assistants" / "booking_assistant"


def _state_path(settings: Settings, user_id: str) -> Path:
    return _assistant_dir(settings, user_id) / "state.json"


def _default_state() -> Dict[str, Any]:
    return {
        "processed_mail_ids": [],
        "reviews": [],
        "holds": [],
        "thread_booking_contexts": [],
        "updated_at": _now_iso(),
    }


def load_state(settings: Settings, user_id: str) -> Dict[str, Any]:
    path = _state_path(settings, user_id)
    if not path.exists():
        return _default_state()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _default_state()
    if not isinstance(raw, dict):
        return _default_state()

    out = _default_state()
    processed = raw.get("processed_mail_ids")
    reviews = raw.get("reviews")
    holds = raw.get("holds")
    thread_booking_contexts = raw.get("thread_booking_contexts")
    if isinstance(processed, list):
        out["processed_mail_ids"] = [str(x).strip() for x in processed if str(x).strip()]
    if isinstance(reviews, list):
        out["reviews"] = [x for x in reviews if isinstance(x, dict)]
    if isinstance(holds, list):
        out["holds"] = [x for x in holds if isinstance(x, dict)]
    if isinstance(thread_booking_contexts, list):
        out["thread_booking_contexts"] = [x for x in thread_booking_contexts if isinstance(x, dict)]
    out["updated_at"] = str(raw.get("updated_at") or out["updated_at"])
    return out


def save_state(settings: Settings, user_id: str, state: Dict[str, Any]) -> None:
    base = _assistant_dir(settings, user_id)
    base.mkdir(parents=True, exist_ok=True)
    payload = dict(state or {})
    payload["updated_at"] = _now_iso()
    _state_path(settings, user_id).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def has_processed(state: Dict[str, Any], mail_id: str) -> bool:
    seen = state.get("processed_mail_ids") if isinstance(state.get("processed_mail_ids"), list) else []
    return str(mail_id).strip() in {str(x).strip() for x in seen if str(x).strip()}


def mark_processed(state: Dict[str, Any], mail_id: str, *, max_items: int = 10000) -> None:
    mid = str(mail_id or "").strip()
    if not mid:
        return
    seen = state.get("processed_mail_ids")
    if not isinstance(seen, list):
        seen = []
    if mid not in seen:
        seen.append(mid)
    if len(seen) > max_items:
        seen = seen[-max_items:]
    state["processed_mail_ids"] = seen


def list_reviews(state: Dict[str, Any], *, status: str = "") -> List[Dict[str, Any]]:
    reviews = state.get("reviews") if isinstance(state.get("reviews"), list) else []
    if not status:
        return [dict(r) for r in reviews if isinstance(r, dict)]
    wanted = str(status).strip().lower()
    out: List[Dict[str, Any]] = []
    for review in reviews:
        if not isinstance(review, dict):
            continue
        if str(review.get("status") or "").strip().lower() == wanted:
            out.append(dict(review))
    return out


def add_review(state: Dict[str, Any], review: Dict[str, Any]) -> None:
    reviews = state.get("reviews")
    if not isinstance(reviews, list):
        reviews = []
    reviews.append(dict(review))
    state["reviews"] = reviews


def find_review(state: Dict[str, Any], review_id: str) -> Dict[str, Any] | None:
    rid = str(review_id or "").strip()
    if not rid:
        return None
    reviews = state.get("reviews") if isinstance(state.get("reviews"), list) else []
    for review in reviews:
        if not isinstance(review, dict):
            continue
        if str(review.get("id") or "").strip() == rid:
            return review
    return None
