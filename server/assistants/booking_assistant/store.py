from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
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
        "activity_log": [],
        "run_history": [],
        "run_lock": {},
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
    activity_log = raw.get("activity_log")
    run_history = raw.get("run_history")
    run_lock = raw.get("run_lock")
    if isinstance(processed, list):
        out["processed_mail_ids"] = [str(x).strip() for x in processed if str(x).strip()]
    if isinstance(reviews, list):
        out["reviews"] = [x for x in reviews if isinstance(x, dict)]
    if isinstance(holds, list):
        out["holds"] = [x for x in holds if isinstance(x, dict)]
    if isinstance(thread_booking_contexts, list):
        out["thread_booking_contexts"] = [x for x in thread_booking_contexts if isinstance(x, dict)]
    if isinstance(activity_log, list):
        out["activity_log"] = [x for x in activity_log if isinstance(x, dict)]
    if isinstance(run_history, list):
        out["run_history"] = [x for x in run_history if isinstance(x, dict)]
    if isinstance(run_lock, dict):
        out["run_lock"] = dict(run_lock)
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


def append_activity(state: Dict[str, Any], item: Dict[str, Any], *, max_items: int = 10000) -> None:
    log = state.get("activity_log")
    if not isinstance(log, list):
        log = []
    entry = dict(item or {})
    if not entry.get("timestamp"):
        entry["timestamp"] = _now_iso()
    log.append(entry)
    if len(log) > max_items:
        log = log[-max_items:]
    state["activity_log"] = log


def append_run_history(state: Dict[str, Any], item: Dict[str, Any], *, max_items: int = 2000) -> None:
    history = state.get("run_history")
    if not isinstance(history, list):
        history = []
    entry = dict(item or {})
    if not entry.get("timestamp"):
        entry["timestamp"] = _now_iso()
    history.append(entry)
    if len(history) > max_items:
        history = history[-max_items:]
    state["run_history"] = history


def _parse_iso(value: str) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None


def acquire_run_lock(
    state: Dict[str, Any],
    *,
    run_id: str,
    ttl_seconds: int = 7200,
) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    lock = state.get("run_lock") if isinstance(state.get("run_lock"), dict) else {}
    active_run_id = str(lock.get("run_id") or "").strip()
    expires_at = _parse_iso(str(lock.get("expires_at") or ""))
    if active_run_id and expires_at and expires_at > now:
        return {
            "acquired": False,
            "run_id": active_run_id,
            "expires_at": expires_at.isoformat().replace("+00:00", "Z"),
            "reason": "run_already_active",
        }

    started_at = now.isoformat().replace("+00:00", "Z")
    expires = (now + timedelta(seconds=max(60, int(ttl_seconds or 7200)))).isoformat().replace("+00:00", "Z")
    new_lock = {"run_id": str(run_id or "").strip(), "started_at": started_at, "expires_at": expires}
    state["run_lock"] = new_lock
    return {"acquired": True, **new_lock, "reason": "lock_acquired"}


def release_run_lock(state: Dict[str, Any], *, run_id: str) -> None:
    lock = state.get("run_lock")
    if not isinstance(lock, dict):
        return
    current = str(lock.get("run_id") or "").strip()
    if current and current != str(run_id or "").strip():
        return
    state["run_lock"] = {}


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
