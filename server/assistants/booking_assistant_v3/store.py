from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

from server.core.settings import Settings


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _assistant_dir(settings: Settings, user_id: str) -> Path:
    return settings.user_dir(user_id) / "assistants" / "booking_assistant_v3"


def _state_path(settings: Settings, user_id: str) -> Path:
    return _assistant_dir(settings, user_id) / "state.json"


def _default_state() -> Dict[str, Any]:
    return {
        "version": "3.0",
        "processed_mail_ids": [],
        "thread_cases": [],
        "reviews": [],
        "activity_log": [],
        "run_history": [],
        "run_lock": {},
        "last_status_at": "",
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
    for key in (
        "processed_mail_ids",
        "thread_cases",
        "reviews",
        "activity_log",
        "run_history",
    ):
        if isinstance(raw.get(key), list):
            out[key] = [x for x in raw.get(key) if isinstance(x, (dict, str, int, float, bool, list))]

    if isinstance(out.get("processed_mail_ids"), list):
        out["processed_mail_ids"] = [
            str(x).strip() for x in out["processed_mail_ids"] if str(x).strip()
        ]

    for key in ("thread_cases", "reviews", "activity_log", "run_history"):
        out[key] = [x for x in out.get(key, []) if isinstance(x, dict)]

    if isinstance(raw.get("run_lock"), dict):
        out["run_lock"] = dict(raw.get("run_lock") or {})

    out["last_status_at"] = str(raw.get("last_status_at") or "").strip()
    out["updated_at"] = str(raw.get("updated_at") or out["updated_at"])
    return out


def save_state(settings: Settings, user_id: str, state: Dict[str, Any]) -> None:
    base = _assistant_dir(settings, user_id)
    base.mkdir(parents=True, exist_ok=True)
    payload = dict(state or {})
    payload["version"] = "3.0"
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


def list_reviews(state: Dict[str, Any], *, status: str = "", kind: str = "") -> List[Dict[str, Any]]:
    reviews = state.get("reviews") if isinstance(state.get("reviews"), list) else []
    out: List[Dict[str, Any]] = []
    want_status = str(status or "").strip().lower()
    want_kind = str(kind or "").strip().lower()
    for review in reviews:
        if not isinstance(review, dict):
            continue
        if want_status and str(review.get("status") or "").strip().lower() != want_status:
            continue
        if want_kind and str(review.get("kind") or "").strip().lower() != want_kind:
            continue
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


def get_thread_case(state: Dict[str, Any], thread_id: str) -> Dict[str, Any] | None:
    tid = str(thread_id or "").strip()
    if not tid:
        return None
    cases = state.get("thread_cases") if isinstance(state.get("thread_cases"), list) else []
    for case in reversed(cases):
        if not isinstance(case, dict):
            continue
        if str(case.get("thread_id") or "").strip() == tid:
            return case
    return None


def upsert_thread_case(
    state: Dict[str, Any],
    *,
    thread_id: str,
    patch: Dict[str, Any],
    max_items: int = 2000,
) -> Dict[str, Any]:
    tid = str(thread_id or "").strip()
    if not tid:
        raise ValueError("thread_id is required")
    cases = state.get("thread_cases")
    if not isinstance(cases, list):
        cases = []

    for case in cases:
        if not isinstance(case, dict):
            continue
        if str(case.get("thread_id") or "").strip() != tid:
            continue
        case.update(dict(patch or {}))
        case["thread_id"] = tid
        case["updated_at"] = _now_iso()
        state["thread_cases"] = cases
        return case

    item = {
        "thread_id": tid,
        "status": "collecting",
        "required_field_names": [],
        "required_fields": {},
        "missing_required_fields": [],
        "facts": {},
        "offer": {},
        "calendar": {},
        "history": [],
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }
    item.update(dict(patch or {}))
    item["thread_id"] = tid
    item["updated_at"] = _now_iso()
    cases.append(item)
    if len(cases) > max_items:
        cases = cases[-max_items:]
    state["thread_cases"] = cases
    return item


def append_case_history(
    state: Dict[str, Any],
    *,
    thread_id: str,
    history_item: Dict[str, Any],
    max_items: int = 250,
) -> None:
    case = upsert_thread_case(state, thread_id=thread_id, patch={})
    history = case.get("history") if isinstance(case.get("history"), list) else []
    entry = dict(history_item or {})
    if not entry.get("timestamp"):
        entry["timestamp"] = _now_iso()
    history.append(entry)
    if len(history) > max_items:
        history = history[-max_items:]
    case["history"] = history
    case["updated_at"] = _now_iso()
