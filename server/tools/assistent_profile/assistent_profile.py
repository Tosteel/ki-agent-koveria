from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from fastapi import HTTPException


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_placeholder_text(value: str) -> bool:
    low = str(value or "").strip().lower()
    if not low:
        return True
    if low in {"string", "null", "none", "n/a"}:
        return True
    if low.startswith("additionalprop"):
        return True
    return False


def _sanitize_instructions(values: List[str] | None) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for raw in (values or []):
        v = str(raw or "").strip()
        if not v or _is_placeholder_text(v):
            continue
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _sanitize_patch_value(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: Dict[str, Any] = {}
        for k, v in value.items():
            key = str(k or "").strip()
            if not key or _is_placeholder_text(key):
                continue
            nested = _sanitize_patch_value(v)
            if nested is None:
                continue
            if isinstance(nested, dict) and not nested:
                continue
            if isinstance(nested, list) and not nested:
                continue
            cleaned[key] = nested
        return cleaned
    if isinstance(value, list):
        cleaned_list: List[Any] = []
        for item in value:
            nested = _sanitize_patch_value(item)
            if nested is None:
                continue
            if isinstance(nested, dict) and not nested:
                continue
            if isinstance(nested, list) and not nested:
                continue
            cleaned_list.append(nested)
        return cleaned_list
    if isinstance(value, str):
        v = value.strip()
        if not v or _is_placeholder_text(v):
            return None
        return v
    return value


def _sanitize_patch(patch: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(patch, dict):
        return {}
    cleaned = _sanitize_patch_value(patch)
    if isinstance(cleaned, dict):
        return cleaned
    return {}


def _safe_name(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        raise HTTPException(status_code=422, detail="assistent_profile_name is required")
    bad = {"..", "/", "\\", ":"}
    if any(x in raw for x in bad):
        raise HTTPException(status_code=422, detail="invalid assistent_profile_name")
    return raw


def _profile_path(user_dir: Path, assistent_profile_name: str) -> Path:
    base = user_dir / "assistants"
    base.mkdir(parents=True, exist_ok=True)
    name = _safe_name(assistent_profile_name)
    return base / f"{name}.json"


def _default_profile(name: str, codename: str = "", instructions: List[str] | None = None, rules: Dict[str, Any] | None = None) -> Dict[str, Any]:
    now = _now_iso()
    return {
        "assistent_profile_name": name,
        "codename": str(codename or "").strip(),
        "instructions": _sanitize_instructions(instructions or []),
        "rules": _sanitize_patch(rules or {}),
        "created_at": now,
        "updated_at": now,
    }


def _load_profile(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"profile not found: {path.name}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"profile parse failed: {exc}") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail="profile file is invalid")
    return data


def _save_profile(path: Path, profile: Dict[str, Any]) -> None:
    profile["updated_at"] = _now_iso()
    path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")


def _deep_merge(dst: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(dst)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out.get(k, {}), v)
        else:
            out[k] = v
    return out


def assistent_profile_create(
    *,
    user_dir: Path,
    assistent_profile_name: str,
    codename: str = "",
    instructions: List[str] | None = None,
    rules: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    name = _safe_name(assistent_profile_name)
    path = _profile_path(user_dir, name)
    if path.exists():
        raise HTTPException(status_code=409, detail=f"profile already exists: {name}")
    profile = _default_profile(name=name, codename=codename, instructions=instructions, rules=rules)
    _save_profile(path, profile)
    return {
        "ok": True,
        "assistent_profile_name": name,
        "path": str(path),
        "profile": profile,
        "text": f"Profile created: {name}",
    }


def assistent_profile_get(
    *,
    user_dir: Path,
    assistent_profile_name: str,
) -> Dict[str, Any]:
    name = _safe_name(assistent_profile_name)
    path = _profile_path(user_dir, name)
    profile = _load_profile(path)
    return {
        "ok": True,
        "assistent_profile_name": name,
        "path": str(path),
        "profile": profile,
        "text": f"Profile loaded: {name}",
    }


def assistent_profile_update(
    *,
    user_dir: Path,
    assistent_profile_name: str,
    codename: str = "",
    instructions_add: List[str] | None = None,
    rules_patch: Dict[str, Any] | None = None,
    raw_patch: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    name = _safe_name(assistent_profile_name)
    path = _profile_path(user_dir, name)
    profile = _load_profile(path)

    if str(codename or "").strip():
        profile["codename"] = str(codename).strip()

    add = _sanitize_instructions(instructions_add or [])
    if add:
        existing = profile.get("instructions")
        if not isinstance(existing, list):
            existing = []
        seen = {str(x).strip() for x in existing if str(x).strip()}
        for item in add:
            if item not in seen:
                existing.append(item)
                seen.add(item)
        profile["instructions"] = existing

    clean_rules_patch = _sanitize_patch(rules_patch or {})
    if clean_rules_patch:
        rules = profile.get("rules")
        if not isinstance(rules, dict):
            rules = {}
        profile["rules"] = _deep_merge(rules, clean_rules_patch)

    clean_raw_patch = _sanitize_patch(raw_patch or {})
    if clean_raw_patch:
        profile = _deep_merge(profile, clean_raw_patch)

    _save_profile(path, profile)
    return {
        "ok": True,
        "assistent_profile_name": name,
        "path": str(path),
        "profile": profile,
        "text": f"Profile updated: {name}",
    }


def _hour_from_iso(value: str) -> int | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return int(dt.hour)
    except Exception:
        return None


def assistent_profile_check(
    *,
    user_dir: Path,
    assistent_profile_name: str,
    action: str,
    context_text: str = "",
    context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    name = _safe_name(assistent_profile_name)
    path = _profile_path(user_dir, name)
    profile = _load_profile(path)
    rules = profile.get("rules")
    if not isinstance(rules, dict):
        rules = {}

    action_name = str(action or "").strip().lower()
    text = str(context_text or "").lower()
    ctx = context or {}

    allowed = True
    reasons: List[str] = []
    warnings: List[str] = []
    matched: List[str] = []

    mail_rules = rules.get("mail") if isinstance(rules.get("mail"), dict) else {}
    if isinstance(mail_rules, dict):
        if bool(mail_rules.get("never_auto_send")) and action_name in {"mail_send", "mail_answer", "gmail_send_mail", "gmail_answer_mail"}:
            allowed = False
            reasons.append("Auto-mail sending disabled by profile rule.")
            matched.append("rules.mail.never_auto_send")

        blocked_topics = mail_rules.get("block_auto_reply_topics")
        if isinstance(blocked_topics, list) and action_name in {"mail_send", "mail_answer", "gmail_send_mail", "gmail_answer_mail"}:
            for topic in blocked_topics:
                t = str(topic or "").strip().lower()
                if t and t in text:
                    allowed = False
                    reasons.append(f"Auto-reply blocked for topic: {t}")
                    matched.append("rules.mail.block_auto_reply_topics")
                    break

    cal_rules = rules.get("calendar") if isinstance(rules.get("calendar"), dict) else {}
    if isinstance(cal_rules, dict) and action_name in {
        "calendar_create_event",
        "calendar_hold_event",
        "calendar_check_availability",
        "calendar_propose_slots",
    }:
        max_hour = cal_rules.get("latest_meeting_end_hour")
        if isinstance(max_hour, int):
            end_iso = str(ctx.get("end_iso") or "")
            h = _hour_from_iso(end_iso)
            if h is not None and h > int(max_hour):
                allowed = False
                reasons.append(f"Meetings after {int(max_hour)}:00 are not allowed.")
                matched.append("rules.calendar.latest_meeting_end_hour")

    if not matched:
        warnings.append("No specific rule matched for this action.")

    return {
        "ok": True,
        "assistent_profile_name": name,
        "action": action_name,
        "allowed": allowed,
        "reasons": reasons,
        "warnings": warnings,
        "matched_rules": matched,
        "text": f"Profile check: {'allowed' if allowed else 'blocked'}",
    }
