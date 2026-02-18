#uvicorn client.gui_web:app --host 0.0.0.0 --port 8013 --reload
#python3 client/gui_web.py

from __future__ import annotations

import json
import ast
import mimetypes
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urlsplit, urlunsplit

import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parent
HTML_PATH = APP_DIR / "gui_web.html"
LOGO_PATH = APP_DIR / "assets" / "logo.png"
BOT_AVATAR_PATH = APP_DIR / "assets" / "bot-avatar.png"
LEGACY_SETTINGS_PATH = APP_DIR / "gui_web_config.json"
USER_DATA_DIR = APP_DIR / "data" / "users"
ACTIVE_USER_PATH = APP_DIR / "data" / "active_user.txt"
DEFAULT_USER_ID = "user1"
SERVER_DATA_DIR = APP_DIR.parent / "server" / "data"

DEFAULT_SETTINGS: Dict[str, str] = {
    "ask_ionos_url": "http://127.0.0.1:8012/agent/ask",
    "api_key": "",
    "provider": "ionos",
}


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: Optional[List["ChatHistoryMessage"]] = None
    user_id: Optional[str] = ""


class ChatHistoryMessage(BaseModel):
    role: Literal["user", "bot"]
    text: str = Field(..., min_length=1)


class SettingsRequest(BaseModel):
    ask_ionos_url: str = Field(..., min_length=1)
    api_key: Optional[str] = ""
    provider: str = "ionos"
    user_id: Optional[str] = ""


class TaskExplainRequest(BaseModel):
    steps: List[str] = Field(default_factory=list)
    user_id: Optional[str] = ""


class TaskSaveRequest(BaseModel):
    task_text: str = Field(..., min_length=1)
    planned_steps: List[str] = Field(default_factory=list)
    user_id: Optional[str] = ""


class TaskRenameRequest(BaseModel):
    task_id: int = Field(..., ge=1)
    title: str = Field(..., min_length=1)
    user_id: Optional[str] = ""


class TaskDeleteRequest(BaseModel):
    task_id: int = Field(..., ge=1)
    user_id: Optional[str] = ""


class AgentCreateFromTaskRequest(BaseModel):
    task_id: int = Field(..., ge=1)
    user_id: Optional[str] = ""


class AgentDeleteRequest(BaseModel):
    agent_id: int = Field(..., ge=1)
    user_id: Optional[str] = ""


class AgentRenameRequest(BaseModel):
    agent_id: int = Field(..., ge=1)
    title: str = Field(..., min_length=1)
    user_id: Optional[str] = ""


class AgentUpdateRequest(BaseModel):
    agent_id: int = Field(..., ge=1)
    planned_steps: List[str] = Field(default_factory=list)
    text: Optional[str] = ""
    dialog: List[Dict[str, Any]] = Field(default_factory=list)
    placeholders: List[Dict[str, Any]] = Field(default_factory=list)
    user_id: Optional[str] = ""


class AgentReplanRequest(BaseModel):
    agent_id: int = Field(..., ge=1)
    planned_steps: List[str] = Field(default_factory=list)
    change_request: str = Field(..., min_length=1)
    user_id: Optional[str] = ""


class TaskRerunRequest(BaseModel):
    task_id: int = Field(..., ge=1)
    planned_steps: List[str] = Field(default_factory=list)
    user_id: Optional[str] = ""


class TaskRerunDeleteRequest(BaseModel):
    task_id: int = Field(..., ge=1)
    rerun_index: int = Field(..., ge=0)
    user_id: Optional[str] = ""


class TaskUpdateRequest(BaseModel):
    task_id: int = Field(..., ge=1)
    planned_steps: List[str] = Field(default_factory=list)
    text: Optional[str] = ""
    dialog: List[Dict[str, Any]] = Field(default_factory=list)
    user_id: Optional[str] = ""


class TaskReplanRequest(BaseModel):
    planned_steps: List[str] = Field(default_factory=list)
    change_request: str = Field(..., min_length=1)
    user_id: Optional[str] = ""


class TriggerRenameRequest(BaseModel):
    trigger_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    user_id: Optional[str] = ""


class TriggerDeleteRequest(BaseModel):
    trigger_id: str = Field(..., min_length=1)
    user_id: Optional[str] = ""


class TriggerCreateManualRequest(BaseModel):
    task_id: int = Field(..., ge=1)
    name: str = Field(..., min_length=1)
    user_id: Optional[str] = ""


class TriggerUpdateProxyRequest(BaseModel):
    trigger_id: str = Field(..., min_length=1)
    name: Optional[str] = None
    trigger_type: Optional[str] = None
    task_id: Optional[int] = Field(default=None, ge=1)
    config: Optional[Dict[str, Any]] = None
    enabled: Optional[bool] = None
    user_id: Optional[str] = ""


class TriggerInterpretRequest(BaseModel):
    trigger: Dict[str, Any] = Field(default_factory=dict)
    message: str = Field(..., min_length=1)
    user_id: Optional[str] = ""


class ChatMemoryMessage(BaseModel):
    role: Literal["user", "bot"]
    text: str = Field(..., min_length=1)
    downloadUrl: str = ""
    downloadLabel: str = ""
    plannedSteps: List[str] = Field(default_factory=list)
    options: List[Dict[str, str]] = Field(default_factory=list)
    timestamp: str = ""


class ChatMemoryChat(BaseModel):
    id: int = Field(..., ge=1)
    title: str = Field(..., min_length=1)
    messages: List[ChatMemoryMessage] = Field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""


class ChatMemoryRequest(BaseModel):
    chats: List[ChatMemoryChat] = Field(default_factory=list)
    active_chat_id: Optional[int] = Field(default=None, ge=1)
    user_id: Optional[str] = ""


def _sanitize_user_id(user_id: Optional[str]) -> str:
    raw = (user_id or "").strip()
    if not raw:
        return DEFAULT_USER_ID
    sanitized = re.sub(r"[^a-zA-Z0-9._-]+", "_", raw)
    return sanitized.strip("._-") or DEFAULT_USER_ID


def _user_settings_path(user_id: str) -> Path:
    safe_user_id = _sanitize_user_id(user_id)
    return USER_DATA_DIR / safe_user_id / "gui_web_config.json"


def _user_chat_memory_path(user_id: str) -> Path:
    safe_user_id = _sanitize_user_id(user_id)
    return USER_DATA_DIR / safe_user_id / "chat_memory.json"


def _user_task_path(user_id: str) -> Path:
    safe_user_id = _sanitize_user_id(user_id)
    return USER_DATA_DIR / safe_user_id / "tasks_memory.json"


def _user_agent_path(user_id: str) -> Path:
    safe_user_id = _sanitize_user_id(user_id)
    return USER_DATA_DIR / safe_user_id / "agents_config.json"


def _planned_steps_block(steps: List[str]) -> str:
    clean_steps = [str(s).strip() for s in (steps or []) if str(s).strip()]
    if not clean_steps:
        return ""
    return (
        "===== PLANNED STEPS =====\n"
        + "\n".join(clean_steps)
        + "\n========================="
    )


def _extract_api_base_url(ask_ionos_url: str) -> Optional[str]:
    try:
        parsed = urlsplit(ask_ionos_url.strip())
    except Exception:
        return None
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return None
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def _normalize_ask_url(raw_url: str) -> str:
    raw = (raw_url or "").strip()
    if not raw:
        return ""
    try:
        parsed = urlsplit(raw)
    except Exception:
        return raw
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return raw
    path = parsed.path or ""
    if path.endswith("/agent/askIonos"):
        path = path[: -len("/agent/askIonos")] + "/agent/ask"
    elif path.endswith("/agent/askOpenAI"):
        path = path[: -len("/agent/askOpenAI")] + "/agent/ask"
    elif not path.endswith("/agent/ask"):
        path = "/agent/ask"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _agent_run_url_from_ask_url(ask_url: str) -> str:
    raw = (ask_url or "").strip()
    if not raw:
        return ""
    parsed = urlsplit(raw)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return ""
    path = parsed.path or ""
    if path.endswith("/agent/askIonos"):
        path = path[: -len("/agent/askIonos")] + "/agent/run"
    elif path.endswith("/agent/askOpenAI"):
        path = path[: -len("/agent/askOpenAI")] + "/agent/run"
    else:
        path = "/agent/run"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _agent_plan_url_from_ask_url(ask_url: str) -> str:
    raw = (ask_url or "").strip()
    if not raw:
        return ""
    parsed = urlsplit(raw)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return ""
    return urlunsplit((parsed.scheme, parsed.netloc, "/agent/plan", "", ""))


def _agent_clarify_url_from_ask_url(ask_url: str) -> str:
    raw = (ask_url or "").strip()
    if not raw:
        return ""
    parsed = urlsplit(raw)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return ""
    return urlunsplit((parsed.scheme, parsed.netloc, "/agent/clarify", "", ""))


def _compact_planned_steps(steps: List[str]) -> List[str]:
    compact: List[str] = []
    for idx, raw in enumerate(steps, start=1):
        line = str(raw or "").strip()
        m = re.match(r"^\s*\d+\.\s*tool=([^\s]+)\s*args=(.*)$", line)
        if not m:
            compact.append(f"{idx}. {line[:180]}")
            continue

        tool = m.group(1).strip()
        args_raw = (m.group(2) or "").strip()
        short_parts: List[str] = []
        try:
            parsed = json.loads(args_raw) if args_raw else {}
            if isinstance(parsed, dict):
                for key, val in parsed.items():
                    if key in {"body", "text", "content", "composed_text"}:
                        continue
                    sval = str(val)
                    if len(sval) > 80:
                        sval = sval[:77] + "..."
                    short_parts.append(f"{key}={sval}")
        except Exception:
            pass

        if short_parts:
            compact.append(f"{idx}. tool={tool} ({', '.join(short_parts[:4])})")
        else:
            compact.append(f"{idx}. tool={tool}")
    return compact


def _extract_rerun_answer(data: Dict[str, Any]) -> str:
    outputs = data.get("outputs")
    if not isinstance(outputs, list):
        return json.dumps(data, ensure_ascii=False)
    for out in reversed(outputs):
        if not isinstance(out, dict) or not out.get("ok"):
            continue
        payload = out.get("payload")
        if isinstance(payload, dict):
            for key in ("composed_text", "text", "summary", "answer", "message"):
                val = payload.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
    return json.dumps(data, ensure_ascii=False)


def _parse_planned_steps_lines(lines: List[str]) -> List[Dict[str, Any]]:
    steps: List[Dict[str, Any]] = []
    for raw in lines:
        line = str(raw or "").strip()
        if not line:
            continue
        m = re.match(r"^\s*\d+\.\s*tool=([^\s]+)\s+args=(.+)$", line)
        if not m:
            continue
        tool = m.group(1).strip()
        args_raw = m.group(2).strip()
        if not tool:
            continue
        args: Dict[str, Any] = {}
        if args_raw:
            try:
                loaded = json.loads(args_raw)
                if isinstance(loaded, dict):
                    args = loaded
            except Exception:
                try:
                    loaded = ast.literal_eval(args_raw)
                    if isinstance(loaded, dict):
                        args = loaded
                except Exception:
                    args = {}
        steps.append({"tool": tool, "args": args})
    return steps


def _parse_json_strictish_text(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


_PLACEHOLDER_PATTERN = re.compile(r"\{\{([a-z0-9_]+)\}\}")
_PLACEHOLDER_SINGLE_PATTERN = re.compile(r"\{([a-z0-9_]+)\}")


def _infer_placeholder_type(name: str, paths: List[str]) -> str:
    low = (name or "").strip().lower()
    combined = " ".join(paths).lower()
    if "mail" in low or "email" in low or "@" in combined or ".to[" in combined:
        return "email"
    if "date" in low or "datum" in low:
        return "date"
    if any(tok in low for tok in ("count", "num", "anzahl", "limit", "top_k")):
        return "number"
    return "string"


def _extract_placeholders_from_steps(steps: List[str]) -> List[Dict[str, Any]]:
    refs: Dict[str, List[str]] = {}
    parsed_steps = _parse_planned_steps_lines(steps)
    for step_idx, step in enumerate(parsed_steps):
        args = step.get("args")
        if not isinstance(args, dict):
            continue

        def walk(value: Any, path: str) -> None:
            if isinstance(value, str):
                for m in _PLACEHOLDER_PATTERN.finditer(value):
                    name = str(m.group(1) or "").strip().lower()
                    if not name:
                        continue
                    refs.setdefault(name, [])
                    if path not in refs[name]:
                        refs[name].append(path)
                for m in _PLACEHOLDER_SINGLE_PATTERN.finditer(value):
                    name = str(m.group(1) or "").strip().lower()
                    if not name or name.startswith("steps"):
                        continue
                    refs.setdefault(name, [])
                    if path not in refs[name]:
                        refs[name].append(path)
                return
            if isinstance(value, dict):
                for k, v in value.items():
                    walk(v, f"{path}.{k}")
                return
            if isinstance(value, list):
                for i, v in enumerate(value):
                    walk(v, f"{path}[{i}]")

        walk(args, f"steps[{step_idx}].args")

    out: List[Dict[str, Any]] = []
    for name in sorted(refs.keys()):
        out.append(
            {
                "name": name,
                "type": _infer_placeholder_type(name, refs[name]),
                "required": True,
                "description": f"Platzhalter {name}",
                "used_in": refs[name],
            }
        )
    return out


def _resolve_user_id_from_api(ask_ionos_url: str, api_key: str) -> Optional[str]:
    ask_url = ask_ionos_url.strip()
    token = api_key.strip()
    if not ask_url or not token:
        return None

    api_base = _extract_api_base_url(ask_url)
    if not api_base:
        return None

    user_url = f"{api_base}/user"
    try:
        resp = requests.get(
            user_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=6,
        )
        if resp.status_code >= 400:
            return None
        payload = resp.json() if resp.content else {}
    except Exception:
        return None

    user_id = payload.get("user_id")
    if not isinstance(user_id, str) or not user_id.strip():
        return None
    return _sanitize_user_id(user_id)


def _merge_settings(data: Any) -> Dict[str, str]:
    try:
        parsed = data if isinstance(data, dict) else {}
    except Exception:
        parsed = {}
    merged = dict(DEFAULT_SETTINGS)
    if isinstance(parsed, dict):
        for k in ("ask_ionos_url", "api_key", "provider"):
            v = parsed.get(k)
            if isinstance(v, str):
                merged[k] = v
    merged["ask_ionos_url"] = _normalize_ask_url(merged.get("ask_ionos_url", ""))
    provider = str(merged.get("provider", "ionos")).strip().lower()
    merged["provider"] = provider if provider in {"ionos", "openai"} else "ionos"
    return merged


def _get_active_user_id() -> str:
    if ACTIVE_USER_PATH.exists():
        try:
            return _sanitize_user_id(ACTIVE_USER_PATH.read_text(encoding="utf-8"))
        except Exception:
            return DEFAULT_USER_ID
    return DEFAULT_USER_ID


def _set_active_user_id(user_id: str) -> None:
    ACTIVE_USER_PATH.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE_USER_PATH.write_text(_sanitize_user_id(user_id), encoding="utf-8")


def _load_settings_for_user(user_id: str) -> Dict[str, str]:
    path = _user_settings_path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        if LEGACY_SETTINGS_PATH.exists():
            try:
                legacy_data = json.loads(LEGACY_SETTINGS_PATH.read_text(encoding="utf-8"))
            except Exception:
                legacy_data = {}
            merged = _merge_settings(legacy_data)
        else:
            merged = dict(DEFAULT_SETTINGS)
        path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        return merged
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    return _merge_settings(data)


def _save_settings_for_user(user_id: str, settings: Dict[str, str]) -> None:
    path = _user_settings_path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")


def _user_work_dir(user_id: str) -> Path:
    return SERVER_DATA_DIR / "users" / _sanitize_user_id(user_id) / "work"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_chat_memory_for_user(user_id: str) -> Dict[str, Any]:
    path = _user_chat_memory_path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return {"chats": [], "active_chat_id": None}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"chats": [], "active_chat_id": None}

    if not isinstance(data, dict):
        return {"chats": [], "active_chat_id": None}

    raw_chats = data.get("chats")
    raw_active = data.get("active_chat_id")

    chats: List[Dict[str, Any]] = []
    if isinstance(raw_chats, list):
        for raw_chat in raw_chats:
            try:
                chat_obj = ChatMemoryChat.model_validate(raw_chat)
            except Exception:
                continue
            chats.append(chat_obj.model_dump())

    active_chat_id: Optional[int] = None
    if isinstance(raw_active, int) and raw_active > 0:
        active_chat_id = raw_active

    return {"chats": chats, "active_chat_id": active_chat_id}


def _save_chat_memory_for_user(user_id: str, req: ChatMemoryRequest) -> None:
    path = _user_chat_memory_path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "updated_at": _now_iso(),
        "chats": [chat.model_dump() for chat in req.chats],
        "active_chat_id": req.active_chat_id,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_tasks_memory_for_user(user_id: str) -> Dict[str, Any]:
    local_memory = _load_tasks_memory_local_for_user(user_id)
    settings = _load_settings_for_user(user_id)
    api_base = _extract_api_base_url(settings.get("ask_ionos_url", ""))
    api_key = settings.get("api_key", "").strip()
    if not api_base or not api_key:
        return {"tasks": local_memory.get("tasks", []), "sync_status": "offline_cache"}
    try:
        resp = requests.get(
            f"{api_base}/tasks/memory",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=8,
        )
        if resp.status_code >= 400:
            return {"tasks": local_memory.get("tasks", []), "sync_status": "offline_cache"}
        data = resp.json() if resp.content else {}
        raw_tasks = data.get("tasks") if isinstance(data, dict) else []
        if isinstance(raw_tasks, list):
            local_tasks = local_memory.get("tasks") if isinstance(local_memory.get("tasks"), list) else []
            if not raw_tasks and local_tasks:
                _write_tasks_memory_for_user(user_id, local_tasks)
                return {"tasks": local_tasks, "sync_status": "synchronized"}
            _write_tasks_memory_local_for_user(user_id, raw_tasks)
            refreshed = _load_tasks_memory_local_for_user(user_id)
            return {"tasks": refreshed.get("tasks", []), "sync_status": "synchronized"}
    except Exception:
        return {"tasks": local_memory.get("tasks", []), "sync_status": "offline_cache"}
    return {"tasks": local_memory.get("tasks", []), "sync_status": "offline_cache"}


def _load_tasks_memory_local_for_user(user_id: str) -> Dict[str, Any]:
    path = _user_task_path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return {"tasks": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"tasks": []}
    if not isinstance(data, dict):
        return {"tasks": []}
    raw_tasks = data.get("tasks")
    tasks: List[Dict[str, Any]] = []
    if isinstance(raw_tasks, list):
        for t in raw_tasks:
            if not isinstance(t, dict):
                continue
            tasks.append(
                {
                    "id": int(t.get("id") or 0),
                    "title": str(t.get("title") or "").strip(),
                    "text": str(t.get("text") or "").strip(),
                    "planned_steps": [str(s) for s in (t.get("planned_steps") or []) if str(s).strip()],
                    "planned_steps_text": str(t.get("planned_steps_text") or "").strip(),
                    "created_at": str(t.get("created_at") or ""),
                    "dialog": [
                        {
                            "role": ("user" if str(m.get("role") or "").strip().lower() == "user" else "bot"),
                            "text": str(m.get("text") or "").strip(),
                            "plannedSteps": [str(s).strip() for s in (m.get("plannedSteps") or []) if str(s).strip()],
                            "options": [
                                {
                                    "type": str(o.get("type") or "").strip(),
                                    "taskId": int(o.get("taskId") or 0),
                                    "created": str(o.get("created") or "").strip(),
                                    "plannedSteps": [str(s).strip() for s in (o.get("plannedSteps") or []) if str(s).strip()],
                                }
                                for o in (m.get("options") or [])
                                if isinstance(o, dict) and str(o.get("type") or "").strip()
                            ],
                            "timestamp": str(m.get("timestamp") or ""),
                        }
                        for m in (t.get("dialog") or [])
                        if isinstance(m, dict) and str(m.get("text") or "").strip()
                    ],
                    "reruns": [
                        {
                            "answer": str(r.get("answer") or "").strip(),
                            "created_at": str(r.get("created_at") or ""),
                        }
                        for r in (t.get("reruns") or [])
                        if isinstance(r, dict) and str(r.get("answer") or "").strip()
                    ],
                }
            )
    tasks = [t for t in tasks if t["id"] > 0 and t["text"]]
    return {"tasks": tasks}


def _write_tasks_memory_local_for_user(user_id: str, tasks: List[Dict[str, Any]]) -> None:
    path = _user_task_path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "updated_at": _now_iso(),
        "tasks": tasks,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_tasks_memory_for_user(user_id: str, tasks: List[Dict[str, Any]]) -> None:
    _write_tasks_memory_local_for_user(user_id, tasks)
    settings = _load_settings_for_user(user_id)
    api_base = _extract_api_base_url(settings.get("ask_ionos_url", ""))
    api_key = settings.get("api_key", "").strip()
    if not api_base or not api_key:
        return
    try:
        requests.post(
            f"{api_base}/tasks/memory/sync",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"tasks": tasks},
            timeout=8,
        )
    except Exception:
        pass


def _load_agents_for_user(user_id: str) -> Dict[str, Any]:
    local_memory = _load_agents_local_for_user(user_id)
    settings = _load_settings_for_user(user_id)
    api_base = _extract_api_base_url(settings.get("ask_ionos_url", ""))
    api_key = settings.get("api_key", "").strip()
    if not api_base or not api_key:
        return {"agents": local_memory.get("agents", []), "sync_status": "offline_cache"}
    try:
        resp = requests.get(
            f"{api_base}/agents/memory",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=8,
        )
        if resp.status_code >= 400:
            return {"agents": local_memory.get("agents", []), "sync_status": "offline_cache"}
        data = resp.json() if resp.content else {}
        raw_agents = data.get("agents") if isinstance(data, dict) else []
        if isinstance(raw_agents, list):
            local_agents = local_memory.get("agents") if isinstance(local_memory.get("agents"), list) else []
            if not raw_agents and local_agents:
                _write_agents_for_user(user_id, local_agents)
                return {"agents": local_agents, "sync_status": "synchronized"}
            _write_agents_local_for_user(user_id, raw_agents)
            refreshed = _load_agents_local_for_user(user_id)
            return {"agents": refreshed.get("agents", []), "sync_status": "synchronized"}
    except Exception:
        return {"agents": local_memory.get("agents", []), "sync_status": "offline_cache"}
    return {"agents": local_memory.get("agents", []), "sync_status": "offline_cache"}


def _load_agents_local_for_user(user_id: str) -> Dict[str, Any]:
    path = _user_agent_path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return {"agents": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"agents": []}
    if not isinstance(data, dict):
        return {"agents": []}
    raw_agents = data.get("agents")
    if not isinstance(raw_agents, list):
        return {"agents": []}
    agents: List[Dict[str, Any]] = []
    for a in raw_agents:
        if not isinstance(a, dict):
            continue
        agent_id = int(a.get("id") or 0)
        text = str(a.get("text") or "").strip()
        if agent_id < 1 or not text:
            continue
        agents.append(
            {
                "id": agent_id,
                "title": str(a.get("title") or "").strip(),
                "text": text,
                "planned_steps": [str(s).strip() for s in (a.get("planned_steps") or []) if str(s).strip()],
                "created_at": str(a.get("created_at") or ""),
                "source_task_id": int(a.get("source_task_id") or 0),
                "placeholders": [
                    {
                        "name": str(p.get("name") or "").strip().lower(),
                        "type": str(p.get("type") or "string").strip().lower() or "string",
                        "required": bool(p.get("required", True)),
                        "description": str(p.get("description") or "").strip(),
                        "used_in": [str(u).strip() for u in (p.get("used_in") or []) if str(u).strip()],
                    }
                    for p in (a.get("placeholders") or [])
                    if isinstance(p, dict) and str(p.get("name") or "").strip()
                ] or _extract_placeholders_from_steps(
                    [str(s).strip() for s in (a.get("planned_steps") or []) if str(s).strip()]
                ),
                "dialog": [
                    {
                        "role": ("user" if str(m.get("role") or "").strip().lower() == "user" else "bot"),
                        "text": str(m.get("text") or "").strip(),
                        "plannedSteps": [str(s).strip() for s in (m.get("plannedSteps") or []) if str(s).strip()],
                        "timestamp": str(m.get("timestamp") or ""),
                    }
                    for m in (a.get("dialog") or [])
                    if isinstance(m, dict) and str(m.get("text") or "").strip()
                ],
            }
        )
    return {"agents": agents}


def _write_agents_for_user(user_id: str, agents: List[Dict[str, Any]]) -> None:
    _write_agents_local_for_user(user_id, agents)
    settings = _load_settings_for_user(user_id)
    api_base = _extract_api_base_url(settings.get("ask_ionos_url", ""))
    api_key = settings.get("api_key", "").strip()
    if not api_base or not api_key:
        return
    try:
        requests.post(
            f"{api_base}/agents/memory/sync",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"agents": agents},
            timeout=8,
        )
    except Exception:
        pass


def _write_agents_local_for_user(user_id: str, agents: List[Dict[str, Any]]) -> None:
    path = _user_agent_path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned: List[Dict[str, Any]] = []
    for a in (agents or []):
        if not isinstance(a, dict):
            continue
        agent_id = int(a.get("id") or 0)
        text = str(a.get("text") or "").strip()
        if agent_id < 1 or not text:
            continue
        cleaned.append(
            {
                "id": agent_id,
                "title": str(a.get("title") or "").strip(),
                "text": text,
                "planned_steps": [str(s).strip() for s in (a.get("planned_steps") or []) if str(s).strip()],
                "created_at": str(a.get("created_at") or _now_iso()),
                "source_task_id": int(a.get("source_task_id") or 0),
                "placeholders": [
                    {
                        "name": str(p.get("name") or "").strip().lower(),
                        "type": str(p.get("type") or "string").strip().lower() or "string",
                        "required": bool(p.get("required", True)),
                        "description": str(p.get("description") or "").strip(),
                        "used_in": [str(u).strip() for u in (p.get("used_in") or []) if str(u).strip()],
                    }
                    for p in (a.get("placeholders") or [])
                    if isinstance(p, dict) and str(p.get("name") or "").strip()
                ] or _extract_placeholders_from_steps(
                    [str(s).strip() for s in (a.get("planned_steps") or []) if str(s).strip()]
                ),
                "dialog": [
                    {
                        "role": ("user" if str(m.get("role") or "").strip().lower() == "user" else "bot"),
                        "text": str(m.get("text") or "").strip(),
                        "plannedSteps": [str(s).strip() for s in (m.get("plannedSteps") or []) if str(s).strip()],
                        "timestamp": str(m.get("timestamp") or ""),
                    }
                    for m in (a.get("dialog") or [])
                    if isinstance(m, dict) and str(m.get("text") or "").strip()
                ],
            }
        )
    payload = {"version": 1, "updated_at": _now_iso(), "agents": cleaned}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


app = FastAPI(title="Koveria Web GUI", version="0.1.0")
MAX_HISTORY_ITEMS = 12


def _build_goal(message: str, history: Optional[List[ChatHistoryMessage]]) -> str:
    _ = history
    return message.strip()


def _build_history_payload(history: Optional[List[ChatHistoryMessage]]) -> List[Dict[str, str]]:
    if not history:
        return []
    items: List[Dict[str, str]] = []
    for msg in history[-MAX_HISTORY_ITEMS:]:
        text = msg.text.strip()
        if not text:
            continue
        role = "user" if msg.role == "user" else "assistant"
        items.append({"role": role, "text": text})
    return items


@app.get("/")
def root() -> FileResponse:
    if not HTML_PATH.exists():
        raise HTTPException(status_code=500, detail="gui_web.html not found")
    return FileResponse(HTML_PATH)


@app.get("/assets/logo.png")
def logo() -> FileResponse:
    if not LOGO_PATH.exists():
        raise HTTPException(status_code=404, detail="Logo not found")
    return FileResponse(LOGO_PATH, media_type="image/png")


@app.get("/assets/bot-avatar.png")
def bot_avatar() -> FileResponse:
    if not BOT_AVATAR_PATH.exists():
        raise HTTPException(status_code=404, detail="Bot avatar not found")
    return FileResponse(BOT_AVATAR_PATH, media_type="image/png")


@app.get("/api/download")
def download(path: str = Query(..., min_length=1), user_id: str = Query("")) -> FileResponse:
    resolved_user_id = _sanitize_user_id(user_id) if user_id.strip() else _get_active_user_id()
    user_id = resolved_user_id
    user_work = _user_work_dir(user_id).resolve()
    rel = path.strip().lstrip("/")
    candidate = (user_work / rel).resolve()
    if user_work not in candidate.parents and candidate != user_work:
        raise HTTPException(status_code=400, detail="Invalid path")
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    media_type, _ = mimetypes.guess_type(candidate.name)
    return FileResponse(candidate, filename=candidate.name, media_type=media_type or "application/octet-stream")


@app.get("/api/settings")
def get_settings(user_id: str = Query("")) -> JSONResponse:
    if not user_id.strip():
        return JSONResponse({"user_id": "", **DEFAULT_SETTINGS})
    resolved_user_id = _sanitize_user_id(user_id)
    settings = _load_settings_for_user(resolved_user_id)
    return JSONResponse({"user_id": resolved_user_id, **settings})


@app.post("/api/settings")
def set_settings(req: SettingsRequest) -> JSONResponse:
    settings = {
        "ask_ionos_url": _normalize_ask_url(req.ask_ionos_url),
        "api_key": (req.api_key or "").strip(),
        "provider": str(req.provider or "ionos").strip().lower(),
    }
    if settings["provider"] not in {"ionos", "openai"}:
        settings["provider"] = "ionos"
    if not settings["ask_ionos_url"]:
        raise HTTPException(status_code=422, detail="ask_ionos_url must not be empty")

    explicit_user_id = _sanitize_user_id(req.user_id) if (req.user_id or "").strip() else ""
    resolved_user_id = _resolve_user_id_from_api(settings["ask_ionos_url"], settings["api_key"])
    # If API key resolves to another user, switch to that user profile and memory.
    user_id = resolved_user_id or explicit_user_id or _get_active_user_id()
    user_id = _sanitize_user_id(user_id)

    _save_settings_for_user(user_id, settings)

    return JSONResponse({"ok": True, "user_id": user_id, "settings": settings})


@app.post("/api/chat")
def chat(req: ChatRequest) -> JSONResponse:
    user_id = _sanitize_user_id(req.user_id) if (req.user_id or "").strip() else _get_active_user_id()
    settings = _load_settings_for_user(user_id)
    url = settings["ask_ionos_url"].strip()
    api_key = settings.get("api_key", "").strip()
    provider = str(settings.get("provider", "ionos")).strip().lower()
    if provider not in {"ionos", "openai"}:
        provider = "ionos"

    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    goal = _build_goal(req.message, req.history)
    history_payload = _build_history_payload(req.history)

    try:
        resp = requests.post(
            url,
            headers=headers,
            json={"goal": goal, "history": history_payload, "provider": provider},
            timeout=180,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"API request failed: {exc}") from exc

    if resp.status_code >= 400:
        snippet = resp.text[:500]
        raise HTTPException(status_code=resp.status_code, detail=f"API error: {snippet}")

    data: Dict[str, Any] = resp.json() if resp.content else {}
    print("\n===== REPLAN RAW RESPONSE =====")
    try:
        print(json.dumps(data, ensure_ascii=False))
    except Exception:
        print(str(data))
    print("================================\n")
    answer = data.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        answer = json.dumps(data, ensure_ascii=False)

    return JSONResponse({"ok": True, "answer": answer, "raw": data})


@app.post("/api/chat-clarify")
def chat_clarify(req: ChatRequest) -> JSONResponse:
    user_id = _sanitize_user_id(req.user_id) if (req.user_id or "").strip() else _get_active_user_id()
    settings = _load_settings_for_user(user_id)
    clarify_url = _agent_clarify_url_from_ask_url(settings["ask_ionos_url"].strip())
    if not clarify_url:
        raise HTTPException(status_code=422, detail="Ungültige Agent-Ask URL.")
    api_key = settings.get("api_key", "").strip()
    provider = str(settings.get("provider", "ionos")).strip().lower()
    if provider not in {"ionos", "openai"}:
        provider = "ionos"

    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    goal = _build_goal(req.message, req.history)
    history_payload = _build_history_payload(req.history)

    try:
        resp = requests.post(
            clarify_url,
            headers=headers,
            json={"goal": goal, "history": history_payload, "provider": provider},
            timeout=60,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"API request failed: {exc}") from exc

    if resp.status_code >= 400:
        snippet = resp.text[:500]
        raise HTTPException(status_code=resp.status_code, detail=f"API error: {snippet}")

    data: Dict[str, Any] = resp.json() if resp.content else {}
    normalized_goal = str(data.get("normalized_goal") or "").strip()
    status = str(data.get("status") or "ready").strip().lower()
    return JSONResponse(
        {
            "ok": True,
            "status": status if status in {"ready", "needs_info"} else "ready",
            "normalized_goal": normalized_goal,
            "raw": data,
        }
    )


@app.post("/api/planned-task-explain")
def planned_task_explain(req: TaskExplainRequest) -> JSONResponse:
    user_id = _sanitize_user_id(req.user_id) if (req.user_id or "").strip() else _get_active_user_id()
    settings = _load_settings_for_user(user_id)
    run_url = _agent_run_url_from_ask_url(settings.get("ask_ionos_url", ""))
    api_key = settings.get("api_key", "").strip()
    if not run_url:
        raise HTTPException(status_code=422, detail="Ungültige Agent-Ask URL.")

    steps = [str(s).strip() for s in (req.steps or []) if str(s).strip()]
    if not steps:
        return JSONResponse({"ok": True, "answer": "Für diesen Chat wurden noch keine geplanten Aufgaben gefunden."})

    compact_steps = _compact_planned_steps(steps)
    source_text = "PLANNED STEPS:\n" + "\n".join(compact_steps)
    compose_instruction = (
        "Zeile 1: Ein individueller, konkreter Titel zur Aufgabe."
        "Verboten im Titel: Anführungszeichen, 'letzte geplante Titel' "
        "Ab Zeile 2 ausschließlich als nummerierte Liste auf Deutsch. "
        "Keine Einleitung, kein Fließtext, keine Zusammenfassung außerhalb der Liste."
        "Jeder Schritt maximal ein kurzer Satz. "
        "Format strikt eine Zeile pro Step: 1. Wissensdatenbank: Nach ... durchsucht."
        "2. Textgenerierung: Ergebnisse aus der Wissensdatenbank in einen Text überführt."
    )

    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "log_label": "TEXTBASE FOR FINAL ANSWER",
        "steps": [
            {
                "tool": "llm_compose",
                "args": {
                    "text": source_text,
                    "instruction": compose_instruction,
                    "goal": "Erkläre die letzte geplante Aufgabe mit individuellem Titel",
                    "max_chars": 500,
                },
            }
        ]
    }

    try:
        resp = requests.post(run_url, headers=headers, json=payload, timeout=90)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"API request failed: {exc}") from exc

    if resp.status_code >= 400:
        snippet = resp.text[:500]
        raise HTTPException(status_code=resp.status_code, detail=f"API error: {snippet}")

    data: Dict[str, Any] = resp.json() if resp.content else {}
    outputs = data.get("outputs")
    if isinstance(outputs, list) and outputs:
        payload0 = outputs[0].get("payload") if isinstance(outputs[0], dict) else {}
        if isinstance(payload0, dict):
            answer = payload0.get("composed_text") or payload0.get("text")
            if isinstance(answer, str) and answer.strip():
                return JSONResponse({"ok": True, "answer": answer.strip()})

    return JSONResponse(
        {
            "ok": True,
            "answer": "Ich konnte die letzte Aufgabe nicht umformulieren. Bitte versuche es erneut.",
        }
    )


@app.get("/api/chat-memory")
def get_chat_memory(user_id: str = Query("")) -> JSONResponse:
    if not user_id.strip():
        return JSONResponse({"user_id": "", "chats": [], "active_chat_id": None})
    resolved_user_id = _sanitize_user_id(user_id)
    memory = _load_chat_memory_for_user(resolved_user_id)
    return JSONResponse({"user_id": resolved_user_id, **memory})


@app.post("/api/chat-memory")
def set_chat_memory(req: ChatMemoryRequest) -> JSONResponse:
    user_id = _sanitize_user_id(req.user_id) if (req.user_id or "").strip() else _get_active_user_id()
    _save_chat_memory_for_user(user_id, req)
    return JSONResponse({"ok": True, "user_id": user_id})


@app.get("/api/tasks-memory")
def get_tasks_memory(user_id: str = Query("")) -> JSONResponse:
    if not user_id.strip():
        return JSONResponse({"user_id": "", "tasks": []})
    resolved_user_id = _sanitize_user_id(user_id)
    memory = _load_tasks_memory_for_user(resolved_user_id)
    return JSONResponse({"user_id": resolved_user_id, **memory})


@app.get("/api/agents-memory")
def get_agents_memory(user_id: str = Query("")) -> JSONResponse:
    if not user_id.strip():
        return JSONResponse({"user_id": "", "agents": []})
    resolved_user_id = _sanitize_user_id(user_id)
    memory = _load_agents_for_user(resolved_user_id)
    return JSONResponse({"user_id": resolved_user_id, **memory})


@app.get("/api/triggers")
def get_triggers(user_id: str = Query("")) -> JSONResponse:
    if not user_id.strip():
        return JSONResponse({"user_id": "", "triggers": []})
    resolved_user_id = _sanitize_user_id(user_id)
    settings = _load_settings_for_user(resolved_user_id)
    api_base = _extract_api_base_url(settings.get("ask_ionos_url", ""))
    api_key = settings.get("api_key", "").strip()
    if not api_base or not api_key:
        return JSONResponse({"user_id": resolved_user_id, "triggers": [], "sync_status": "offline_cache"})

    try:
        resp = requests.get(
            f"{api_base}/triggers",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=8,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"API request failed: {exc}") from exc

    if resp.status_code >= 400:
        snippet = resp.text[:500]
        raise HTTPException(status_code=resp.status_code, detail=f"API error: {snippet}")

    data: Dict[str, Any] = resp.json() if resp.content else {}
    triggers = data.get("triggers") if isinstance(data, dict) else []
    if not isinstance(triggers, list):
        triggers = []
    return JSONResponse({"user_id": resolved_user_id, "triggers": triggers, "sync_status": "synchronized"})


@app.post("/api/triggers/rename")
def rename_trigger(req: TriggerRenameRequest) -> JSONResponse:
    user_id = _sanitize_user_id(req.user_id) if (req.user_id or "").strip() else _get_active_user_id()
    trigger_id = req.trigger_id.strip()
    name = req.name.strip()
    if not trigger_id:
        raise HTTPException(status_code=422, detail="trigger_id must not be empty")
    if not name:
        raise HTTPException(status_code=422, detail="name must not be empty")

    settings = _load_settings_for_user(user_id)
    api_base = _extract_api_base_url(settings.get("ask_ionos_url", ""))
    api_key = settings.get("api_key", "").strip()
    if not api_base or not api_key:
        raise HTTPException(status_code=422, detail="API settings missing")

    try:
        resp = requests.patch(
            f"{api_base}/triggers/{trigger_id}",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"name": name},
            timeout=8,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"API request failed: {exc}") from exc
    if resp.status_code >= 400:
        snippet = resp.text[:500]
        raise HTTPException(status_code=resp.status_code, detail=f"API error: {snippet}")

    data: Dict[str, Any] = resp.json() if resp.content else {}
    return JSONResponse({"ok": True, "user_id": user_id, "trigger": data.get("trigger") if isinstance(data, dict) else {}})


@app.post("/api/triggers/delete")
def delete_trigger(req: TriggerDeleteRequest) -> JSONResponse:
    user_id = _sanitize_user_id(req.user_id) if (req.user_id or "").strip() else _get_active_user_id()
    trigger_id = req.trigger_id.strip()
    if not trigger_id:
        raise HTTPException(status_code=422, detail="trigger_id must not be empty")

    settings = _load_settings_for_user(user_id)
    api_base = _extract_api_base_url(settings.get("ask_ionos_url", ""))
    api_key = settings.get("api_key", "").strip()
    if not api_base or not api_key:
        raise HTTPException(status_code=422, detail="API settings missing")

    try:
        resp = requests.delete(
            f"{api_base}/triggers/{trigger_id}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=8,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"API request failed: {exc}") from exc
    if resp.status_code >= 400:
        snippet = resp.text[:500]
        raise HTTPException(status_code=resp.status_code, detail=f"API error: {snippet}")

    data: Dict[str, Any] = resp.json() if resp.content else {}
    if isinstance(data, dict) and data.get("ok") is False:
        raise HTTPException(status_code=404, detail=str(data.get("error") or "trigger_not_found"))
    return JSONResponse({"ok": True, "user_id": user_id, "trigger_id": trigger_id})


@app.post("/api/triggers/create-manual")
def create_manual_trigger(req: TriggerCreateManualRequest) -> JSONResponse:
    user_id = _sanitize_user_id(req.user_id) if (req.user_id or "").strip() else _get_active_user_id()
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="name must not be empty")

    settings = _load_settings_for_user(user_id)
    api_base = _extract_api_base_url(settings.get("ask_ionos_url", ""))
    api_key = settings.get("api_key", "").strip()
    if not api_base or not api_key:
        raise HTTPException(status_code=422, detail="API settings missing")

    payload = {
        "name": name,
        "trigger_type": "manually",
        "task_id": int(req.task_id),
        "config": {},
        "enabled": False,
    }
    try:
        resp = requests.post(
            f"{api_base}/triggers",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=8,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"API request failed: {exc}") from exc
    if resp.status_code >= 400:
        snippet = resp.text[:500]
        raise HTTPException(status_code=resp.status_code, detail=f"API error: {snippet}")

    data: Dict[str, Any] = resp.json() if resp.content else {}
    trigger = data.get("trigger") if isinstance(data, dict) else {}
    return JSONResponse({"ok": True, "user_id": user_id, "trigger": trigger})


@app.post("/api/triggers/update")
def update_trigger_proxy(req: TriggerUpdateProxyRequest) -> JSONResponse:
    user_id = _sanitize_user_id(req.user_id) if (req.user_id or "").strip() else _get_active_user_id()
    trigger_id = req.trigger_id.strip()
    if not trigger_id:
        raise HTTPException(status_code=422, detail="trigger_id must not be empty")

    payload: Dict[str, Any] = {}
    if req.name is not None:
        payload["name"] = str(req.name).strip()
    if req.trigger_type is not None:
        payload["trigger_type"] = str(req.trigger_type).strip()
    if req.task_id is not None:
        payload["task_id"] = int(req.task_id)
    if req.config is not None:
        payload["config"] = req.config
    if req.enabled is not None:
        payload["enabled"] = bool(req.enabled)
    if not payload:
        raise HTTPException(status_code=422, detail="No update fields provided")

    settings = _load_settings_for_user(user_id)
    api_base = _extract_api_base_url(settings.get("ask_ionos_url", ""))
    api_key = settings.get("api_key", "").strip()
    if not api_base or not api_key:
        raise HTTPException(status_code=422, detail="API settings missing")

    try:
        resp = requests.patch(
            f"{api_base}/triggers/{trigger_id}",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=8,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"API request failed: {exc}") from exc
    if resp.status_code >= 400:
        snippet = resp.text[:500]
        raise HTTPException(status_code=resp.status_code, detail=f"API error: {snippet}")
    data: Dict[str, Any] = resp.json() if resp.content else {}
    if isinstance(data, dict) and data.get("ok") is False:
        raise HTTPException(status_code=404, detail=str(data.get("error") or "trigger_not_found"))
    return JSONResponse({"ok": True, "user_id": user_id, "trigger": data.get("trigger") if isinstance(data, dict) else {}})


@app.post("/api/triggers/interpret")
def interpret_trigger_update(req: TriggerInterpretRequest) -> JSONResponse:
    user_id = _sanitize_user_id(req.user_id) if (req.user_id or "").strip() else _get_active_user_id()
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="message must not be empty")

    settings = _load_settings_for_user(user_id)
    run_url = _agent_run_url_from_ask_url(settings.get("ask_ionos_url", ""))
    api_key = settings.get("api_key", "").strip()
    if not run_url or not api_key:
        raise HTTPException(status_code=422, detail="API settings missing")

    trigger_ctx = req.trigger if isinstance(req.trigger, dict) else {}
    source_text = (
        "Aktueller Trigger (JSON):\n"
        + json.dumps(trigger_ctx, ensure_ascii=False)
        + "\n\nÄnderungswunsch:\n"
        + message
    )
    instruction = (
        "Analysiere den Änderungswunsch und gib AUSSCHLIESSLICH gültiges JSON zurück.\n"
        "Schema:\n"
        "{"
        "\"action\":\"update_trigger|ask_user\","
        "\"patch\":{optional: name, trigger_type, task_id, enabled, config},"
        "\"needs_info\":true|false,"
        "\"question\":\"...\""
        "}\n"
        "Regeln:\n"
        "- trigger_type nur 'manually' oder 'time_schedule'.\n"
        "- Für time_schedule setze config.interval_seconds (10..86400).\n"
        "- Wenn Angaben fehlen, needs_info=true und eine kurze Rückfrage in question.\n"
        "- Kein Fließtext, kein Markdown, nur JSON."
    )
    payload = {
        "steps": [
            {
                "tool": "llm_compose",
                "args": {
                    "text": source_text,
                    "instruction": instruction,
                    "goal": "Interpretiere Trigger-Änderung",
                    "max_chars": 1200,
                },
            }
        ]
    }
    headers: Dict[str, str] = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    try:
        resp = requests.post(run_url, headers=headers, json=payload, timeout=60)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"API request failed: {exc}") from exc
    if resp.status_code >= 400:
        snippet = resp.text[:500]
        raise HTTPException(status_code=resp.status_code, detail=f"API error: {snippet}")

    data: Dict[str, Any] = resp.json() if resp.content else {}
    outputs = data.get("outputs")
    text_out = ""
    if isinstance(outputs, list) and outputs:
        p0 = outputs[0].get("payload") if isinstance(outputs[0], dict) else {}
        if isinstance(p0, dict):
            val = p0.get("composed_text") or p0.get("text")
            if isinstance(val, str):
                text_out = val.strip()
    parsed = _parse_json_strictish_text(text_out)
    action = str(parsed.get("action") or "").strip().lower()
    needs_info = bool(parsed.get("needs_info", False))
    question = str(parsed.get("question") or "").strip()
    raw_patch = parsed.get("patch") if isinstance(parsed.get("patch"), dict) else {}
    patch: Dict[str, Any] = {}
    if isinstance(raw_patch, dict):
        if "name" in raw_patch:
            patch["name"] = str(raw_patch.get("name") or "").strip()
        if "trigger_type" in raw_patch:
            patch["trigger_type"] = str(raw_patch.get("trigger_type") or "").strip()
        if "task_id" in raw_patch:
            try:
                patch["task_id"] = int(raw_patch.get("task_id"))
            except Exception:
                pass
        if "enabled" in raw_patch:
            patch["enabled"] = bool(raw_patch.get("enabled"))
        if "config" in raw_patch and isinstance(raw_patch.get("config"), dict):
            patch["config"] = raw_patch.get("config")

    if action not in {"update_trigger", "ask_user"}:
        action = "ask_user"
        needs_info = True
        if not question:
            question = "Wie genau soll der Trigger geändert werden? (z. B. 'auf time_schedule 300s ändern')"

    return JSONResponse(
        {
            "ok": True,
            "action": action,
            "patch": patch,
            "needs_info": needs_info,
            "question": question,
            "raw_text": text_out,
        }
    )


@app.post("/api/tasks/save")
def save_task(req: TaskSaveRequest) -> JSONResponse:
    user_id = _sanitize_user_id(req.user_id) if (req.user_id or "").strip() else _get_active_user_id()
    task_text = req.task_text.strip()
    planned_steps = [str(s).strip() for s in (req.planned_steps or []) if str(s).strip()]
    if not task_text:
        raise HTTPException(status_code=422, detail="task_text must not be empty")

    existing = _load_tasks_memory_for_user(user_id)
    tasks = existing.get("tasks") if isinstance(existing.get("tasks"), list) else []
    next_id = max((int(t.get("id") or 0) for t in tasks), default=0) + 1

    task_entry = {
        "id": next_id,
        "title": task_text[:80],
        "text": task_text,
        "planned_steps": planned_steps,
        "planned_steps_text": _planned_steps_block(planned_steps),
        "created_at": _now_iso(),
        "dialog": [],
        "reruns": [],
    }
    tasks.append(task_entry)
    _write_tasks_memory_for_user(user_id, tasks)
    return JSONResponse({"ok": True, "user_id": user_id, "saved": task_entry})


@app.post("/api/tasks/rename")
def rename_task(req: TaskRenameRequest) -> JSONResponse:
    user_id = _sanitize_user_id(req.user_id) if (req.user_id or "").strip() else _get_active_user_id()
    title = req.title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="title must not be empty")

    memory = _load_tasks_memory_for_user(user_id)
    tasks = memory.get("tasks") if isinstance(memory.get("tasks"), list) else []
    updated: Optional[Dict[str, Any]] = None
    for task in tasks:
        if int(task.get("id") or 0) == req.task_id:
            task["title"] = title
            updated = task
            break
    if updated is None:
        raise HTTPException(status_code=404, detail="Task not found")

    _write_tasks_memory_for_user(user_id, tasks)
    return JSONResponse({"ok": True, "user_id": user_id, "task": updated})


@app.post("/api/tasks/update")
def update_task(req: TaskUpdateRequest) -> JSONResponse:
    user_id = _sanitize_user_id(req.user_id) if (req.user_id or "").strip() else _get_active_user_id()
    memory = _load_tasks_memory_for_user(user_id)
    tasks = memory.get("tasks") if isinstance(memory.get("tasks"), list) else []
    updated: Optional[Dict[str, Any]] = None

    clean_steps = [str(s).strip() for s in (req.planned_steps or []) if str(s).strip()]
    clean_text = str(req.text or "").strip()
    clean_dialog = []
    for m in (req.dialog or []):
        if not isinstance(m, dict):
            continue
        text = str(m.get("text") or "").strip()
        if not text:
            continue
        role = "user" if str(m.get("role") or "").strip().lower() == "user" else "bot"
        clean_dialog.append(
            {
                "role": role,
                "text": text,
                "plannedSteps": [str(s).strip() for s in (m.get("plannedSteps") or []) if str(s).strip()],
                "options": [
                    {
                        "type": str(o.get("type") or "").strip(),
                        "taskId": int(o.get("taskId") or 0),
                        "created": str(o.get("created") or "").strip(),
                        "plannedSteps": [str(s).strip() for s in (o.get("plannedSteps") or []) if str(s).strip()],
                    }
                    for o in (m.get("options") or [])
                    if isinstance(o, dict) and str(o.get("type") or "").strip()
                ],
                "timestamp": str(m.get("timestamp") or ""),
            }
        )

    for task in tasks:
        if int(task.get("id") or 0) != req.task_id:
            continue
        task["planned_steps"] = clean_steps
        task["planned_steps_text"] = _planned_steps_block(clean_steps)
        if clean_text:
            task["text"] = clean_text
        task["dialog"] = clean_dialog
        updated = task
        break
    if updated is None:
        raise HTTPException(status_code=404, detail="Task not found")

    _write_tasks_memory_for_user(user_id, tasks)
    return JSONResponse({"ok": True, "user_id": user_id, "task": updated})


@app.post("/api/tasks/replan")
def replan_task(req: TaskReplanRequest) -> JSONResponse:
    user_id = _sanitize_user_id(req.user_id) if (req.user_id or "").strip() else _get_active_user_id()
    settings = _load_settings_for_user(user_id)
    plan_url = _agent_plan_url_from_ask_url(settings.get("ask_ionos_url", ""))
    api_key = settings.get("api_key", "").strip()
    if not plan_url:
        raise HTTPException(status_code=422, detail="Ungültige Agent-Ask URL.")

    current_steps = [str(s).strip() for s in (req.planned_steps or []) if str(s).strip()]
    if not current_steps:
        raise HTTPException(status_code=422, detail="planned_steps must not be empty")
    change_request = str(req.change_request or "").strip()
    if not change_request:
        raise HTTPException(status_code=422, detail="change_request must not be empty")

    planning_goal = (
        f"{change_request}\n\n"
        "Wenn Eingaben zur Laufzeit variabel sein sollen, verwende ausschließlich das Format "
        "{{placeholder_name}} mit Kleinbuchstaben, Zahlen und Unterstrich."
    )

    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "goal": planning_goal,
        "additional_props": {
            "planned_steps": current_steps,
            "placeholder_syntax": "{{placeholder_name}}",
        },
    }

    try:
        resp = requests.post(plan_url, headers=headers, json=payload, timeout=90)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"API request failed: {exc}") from exc

    if resp.status_code >= 400:
        snippet = resp.text[:500]
        raise HTTPException(status_code=resp.status_code, detail=f"API error: {snippet}")

    data: Dict[str, Any] = resp.json() if resp.content else {}
    raw_steps = data.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise HTTPException(status_code=422, detail="Replanning returned no valid planned steps")

    planned_steps: List[str] = []
    for idx, step in enumerate(raw_steps, start=1):
        if isinstance(step, str):
            line = step.strip()
            if re.match(r"^\d+\.\s*tool=.+\s+args=.+$", line):
                planned_steps.append(line)
            continue
        if not isinstance(step, dict):
            continue

        tool = str(step.get("tool") or "").strip()
        args_raw = step.get("args")
        args: Dict[str, Any] = {}
        if isinstance(args_raw, dict):
            args = args_raw
        elif isinstance(args_raw, str):
            txt = args_raw.strip()
            if txt:
                try:
                    parsed = json.loads(txt)
                    if isinstance(parsed, dict):
                        args = parsed
                except Exception:
                    try:
                        parsed = ast.literal_eval(txt)
                        if isinstance(parsed, dict):
                            args = parsed
                    except Exception:
                        args = {"value": txt}

        if not tool:
            continue
        args_json = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
        planned_steps.append(f"{idx}. tool={tool} args={args_json}")

    if not planned_steps:
        print("\n===== REPLAN PARSE ERROR =====")
        print("No valid planned_steps could be extracted from /agent/plan response.")
        print("raw_steps=", raw_steps)
        print("==============================\n")
        raise HTTPException(status_code=422, detail="Replanning returned no valid planned steps")

    raw_text = _planned_steps_block(planned_steps)
    print("\n===== REPLAN PARSED STEPS =====")
    for s in planned_steps:
        print(s)
    print("================================\n")
    return JSONResponse({"ok": True, "planned_steps": planned_steps, "raw_text": raw_text})


@app.post("/api/tasks/delete")
def delete_task(req: TaskDeleteRequest) -> JSONResponse:
    user_id = _sanitize_user_id(req.user_id) if (req.user_id or "").strip() else _get_active_user_id()
    memory = _load_tasks_memory_for_user(user_id)
    tasks = memory.get("tasks") if isinstance(memory.get("tasks"), list) else []
    kept: List[Dict[str, Any]] = []
    deleted = False
    for task in tasks:
        if int(task.get("id") or 0) == req.task_id:
            deleted = True
            continue
        kept.append(task)
    if not deleted:
        raise HTTPException(status_code=404, detail="Task not found")

    _write_tasks_memory_for_user(user_id, kept)
    return JSONResponse({"ok": True, "user_id": user_id, "task_id": req.task_id})


@app.post("/api/tasks/rerun")
def rerun_task(req: TaskRerunRequest) -> JSONResponse:
    user_id = _sanitize_user_id(req.user_id) if (req.user_id or "").strip() else _get_active_user_id()
    settings = _load_settings_for_user(user_id)
    ask_url = settings.get("ask_ionos_url", "").strip()
    run_url = _agent_run_url_from_ask_url(ask_url)
    api_key = settings.get("api_key", "").strip()
    if not run_url:
        raise HTTPException(status_code=422, detail="Ungültige Agent-Ask URL.")

    clean_steps = [str(s).strip() for s in (req.planned_steps or []) if str(s).strip()]
    if not clean_steps:
        raise HTTPException(status_code=422, detail="planned_steps must not be empty")
    explicit_steps = _parse_planned_steps_lines(clean_steps)
    if not explicit_steps:
        raise HTTPException(status_code=422, detail="planned_steps contain no executable steps")

    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        resp = requests.post(run_url, headers=headers, json={"steps": explicit_steps}, timeout=180)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"API request failed: {exc}") from exc

    if resp.status_code >= 400:
        snippet = resp.text[:500]
        raise HTTPException(status_code=resp.status_code, detail=f"API error: {snippet}")

    data: Dict[str, Any] = resp.json() if resp.content else {}
    answer = _extract_rerun_answer(data)

    memory = _load_tasks_memory_for_user(user_id)
    tasks = memory.get("tasks") if isinstance(memory.get("tasks"), list) else []
    updated: Optional[Dict[str, Any]] = None
    for task in tasks:
        if int(task.get("id") or 0) != req.task_id:
            continue
        reruns = task.get("reruns")
        if not isinstance(reruns, list):
            reruns = []
        reruns.append({"answer": str(answer or "").strip(), "created_at": _now_iso()})
        task["reruns"] = reruns
        updated = task
        break

    if updated is None:
        raise HTTPException(status_code=404, detail="Task not found")

    _write_tasks_memory_for_user(user_id, tasks)
    return JSONResponse({"ok": True, "answer": answer, "raw": data, "task": updated})


@app.post("/api/tasks/rerun/delete")
def delete_task_rerun(req: TaskRerunDeleteRequest) -> JSONResponse:
    user_id = _sanitize_user_id(req.user_id) if (req.user_id or "").strip() else _get_active_user_id()
    memory = _load_tasks_memory_for_user(user_id)
    tasks = memory.get("tasks") if isinstance(memory.get("tasks"), list) else []
    updated: Optional[Dict[str, Any]] = None
    for task in tasks:
        if int(task.get("id") or 0) != req.task_id:
            continue
        reruns = task.get("reruns")
        if not isinstance(reruns, list):
            raise HTTPException(status_code=404, detail="Rerun not found")
        if req.rerun_index < 0 or req.rerun_index >= len(reruns):
            raise HTTPException(status_code=404, detail="Rerun not found")
        reruns.pop(req.rerun_index)
        task["reruns"] = reruns
        updated = task
        break
    if updated is None:
        raise HTTPException(status_code=404, detail="Task not found")

    _write_tasks_memory_for_user(user_id, tasks)
    return JSONResponse({"ok": True, "user_id": user_id, "task": updated})


@app.post("/api/agents/create-from-task")
def create_agent_from_task(req: AgentCreateFromTaskRequest) -> JSONResponse:
    user_id = _sanitize_user_id(req.user_id) if (req.user_id or "").strip() else _get_active_user_id()
    task_memory = _load_tasks_memory_for_user(user_id)
    tasks = task_memory.get("tasks") if isinstance(task_memory.get("tasks"), list) else []
    src_task: Optional[Dict[str, Any]] = None
    for t in tasks:
        if int(t.get("id") or 0) == req.task_id:
            src_task = t
            break
    if src_task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    text = str(src_task.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="Task text is empty")
    title = str(src_task.get("title") or "").strip() or text[:80]
    planned_steps = [str(s).strip() for s in (src_task.get("planned_steps") or []) if str(s).strip()]

    agents_memory = _load_agents_for_user(user_id)
    agents = agents_memory.get("agents") if isinstance(agents_memory.get("agents"), list) else []
    next_id = max((int(a.get("id") or 0) for a in agents), default=0) + 1
    created = {
        "id": next_id,
        "title": title,
        "text": text,
        "planned_steps": planned_steps,
        "created_at": _now_iso(),
        "source_task_id": int(src_task.get("id") or 0),
        "placeholders": _extract_placeholders_from_steps(planned_steps),
        "dialog": [],
    }
    agents.append(created)
    _write_agents_for_user(user_id, agents)
    return JSONResponse({"ok": True, "user_id": user_id, "agent": created})


@app.post("/api/agents/delete")
def delete_agent(req: AgentDeleteRequest) -> JSONResponse:
    user_id = _sanitize_user_id(req.user_id) if (req.user_id or "").strip() else _get_active_user_id()
    memory = _load_agents_for_user(user_id)
    agents = memory.get("agents") if isinstance(memory.get("agents"), list) else []
    kept: List[Dict[str, Any]] = []
    deleted = False
    for a in agents:
        if int(a.get("id") or 0) == req.agent_id:
            deleted = True
            continue
        kept.append(a)
    if not deleted:
        raise HTTPException(status_code=404, detail="Agent not found")
    _write_agents_for_user(user_id, kept)
    return JSONResponse({"ok": True, "user_id": user_id, "agent_id": req.agent_id})


@app.post("/api/agents/rename")
def rename_agent(req: AgentRenameRequest) -> JSONResponse:
    user_id = _sanitize_user_id(req.user_id) if (req.user_id or "").strip() else _get_active_user_id()
    title = str(req.title or "").strip()
    if not title:
        raise HTTPException(status_code=422, detail="title must not be empty")
    memory = _load_agents_for_user(user_id)
    agents = memory.get("agents") if isinstance(memory.get("agents"), list) else []
    updated: Optional[Dict[str, Any]] = None
    for a in agents:
        if int(a.get("id") or 0) != req.agent_id:
            continue
        a["title"] = title
        updated = a
        break
    if updated is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    _write_agents_for_user(user_id, agents)
    return JSONResponse({"ok": True, "user_id": user_id, "agent": updated})


@app.post("/api/agents/update")
def update_agent(req: AgentUpdateRequest) -> JSONResponse:
    user_id = _sanitize_user_id(req.user_id) if (req.user_id or "").strip() else _get_active_user_id()
    memory = _load_agents_for_user(user_id)
    agents = memory.get("agents") if isinstance(memory.get("agents"), list) else []
    updated: Optional[Dict[str, Any]] = None

    clean_steps = [str(s).strip() for s in (req.planned_steps or []) if str(s).strip()]
    clean_text = str(req.text or "").strip()
    clean_dialog = []
    clean_placeholders = []
    for p in (req.placeholders or []):
        if not isinstance(p, dict):
            continue
        name = str(p.get("name") or "").strip().lower()
        if not name or not re.match(r"^[a-z0-9_]+$", name):
            continue
        clean_placeholders.append(
            {
                "name": name,
                "type": str(p.get("type") or "string").strip().lower() or "string",
                "required": bool(p.get("required", True)),
                "description": str(p.get("description") or "").strip(),
                "used_in": [str(u).strip() for u in (p.get("used_in") or []) if str(u).strip()],
            }
        )
    for m in (req.dialog or []):
        if not isinstance(m, dict):
            continue
        text = str(m.get("text") or "").strip()
        if not text:
            continue
        role = "user" if str(m.get("role") or "").strip().lower() == "user" else "bot"
        clean_dialog.append(
            {
                "role": role,
                "text": text,
                "plannedSteps": [str(s).strip() for s in (m.get("plannedSteps") or []) if str(s).strip()],
                "timestamp": str(m.get("timestamp") or ""),
            }
        )
    if not clean_placeholders:
        clean_placeholders = _extract_placeholders_from_steps(clean_steps)

    for agent in agents:
        if int(agent.get("id") or 0) != req.agent_id:
            continue
        agent["planned_steps"] = clean_steps
        if clean_text:
            agent["text"] = clean_text
        agent["dialog"] = clean_dialog
        agent["placeholders"] = clean_placeholders
        updated = agent
        break
    if updated is None:
        raise HTTPException(status_code=404, detail="Agent not found")

    _write_agents_for_user(user_id, agents)
    return JSONResponse({"ok": True, "user_id": user_id, "agent": updated})


@app.post("/api/agents/replan")
def replan_agent(req: AgentReplanRequest) -> JSONResponse:
    user_id = _sanitize_user_id(req.user_id) if (req.user_id or "").strip() else _get_active_user_id()
    settings = _load_settings_for_user(user_id)
    plan_url = _agent_plan_url_from_ask_url(settings.get("ask_ionos_url", ""))
    api_key = settings.get("api_key", "").strip()
    if not plan_url:
        raise HTTPException(status_code=422, detail="Ungültige Agent-Ask URL.")

    current_steps = [str(s).strip() for s in (req.planned_steps or []) if str(s).strip()]
    if not current_steps:
        raise HTTPException(status_code=422, detail="planned_steps must not be empty")
    change_request = str(req.change_request or "").strip()
    if not change_request:
        raise HTTPException(status_code=422, detail="change_request must not be empty")

    planning_goal = (
        f"{change_request}\n\n"
        "WICHTIGE FORMATREGELN FUER PLATZHALTER:\n"
        "1) Verwende Platzhalter AUSSCHLIESSLICH als {{placeholder_name}}.\n"
        "2) Erlaubte Zeichen im Namen: a-z, 0-9, _. Keine Leerzeichen.\n"
        "3) Einfache Klammern sind VERBOTEN: {placeholder_name}.\n"
        "4) Step-Referenzen bleiben unveraendert, z. B. {steps[0].text}.\n"
        "5) Wenn ein statischer Wert variabel sein soll, ersetze ihn durch {{...}}.\n"
        "Beispiele:\n"
        "- Falsch: {email_platzhalter}\n"
        "- Richtig: {{email_platzhalter}}\n"
        "- Richtig: {steps[0].text}\n"
    )
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "goal": planning_goal,
        "additional_props": {
            "planned_steps": current_steps,
            "placeholder_syntax": "{{placeholder_name}}",
        },
    }

    try:
        resp = requests.post(plan_url, headers=headers, json=payload, timeout=90)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"API request failed: {exc}") from exc

    if resp.status_code >= 400:
        snippet = resp.text[:500]
        raise HTTPException(status_code=resp.status_code, detail=f"API error: {snippet}")

    data: Dict[str, Any] = resp.json() if resp.content else {}
    raw_steps = data.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise HTTPException(status_code=422, detail="Replanning returned no valid planned steps")

    planned_steps: List[str] = []
    for idx, step in enumerate(raw_steps, start=1):
        if isinstance(step, str):
            line = step.strip()
            m_line = re.match(r"^\s*\d+\.\s*tool=([^\s]+)\s+args=(.+)$", line)
            if m_line:
                tool = str(m_line.group(1) or "").strip()
                args_raw = str(m_line.group(2) or "").strip()
                args_obj: Dict[str, Any] = {}
                if args_raw:
                    try:
                        parsed = json.loads(args_raw)
                        if isinstance(parsed, dict):
                            args_obj = parsed
                    except Exception:
                        try:
                            parsed = ast.literal_eval(args_raw)
                            if isinstance(parsed, dict):
                                args_obj = parsed
                        except Exception:
                            args_obj = {}
                args_json = json.dumps(args_obj, ensure_ascii=False, separators=(",", ":"))
                planned_steps.append(f"{idx}. tool={tool} args={args_json}")
            continue
        if not isinstance(step, dict):
            continue

        tool = str(step.get("tool") or "").strip()
        args_raw = step.get("args")
        args: Dict[str, Any] = {}
        if isinstance(args_raw, dict):
            args = args_raw
        elif isinstance(args_raw, str):
            txt = args_raw.strip()
            if txt:
                try:
                    parsed = json.loads(txt)
                    if isinstance(parsed, dict):
                        args = parsed
                except Exception:
                    try:
                        parsed = ast.literal_eval(txt)
                        if isinstance(parsed, dict):
                            args = parsed
                    except Exception:
                        args = {"value": txt}

        if not tool:
            continue
        args_json = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
        planned_steps.append(f"{idx}. tool={tool} args={args_json}")

    if not planned_steps:
        raise HTTPException(status_code=422, detail="Replanning returned no valid planned steps")

    raw_text = _planned_steps_block(planned_steps)
    placeholders = _extract_placeholders_from_steps(planned_steps)
    return JSONResponse(
        {
            "ok": True,
            "planned_steps": planned_steps,
            "raw_text": raw_text,
            "placeholders": placeholders,
            "placeholder_syntax": "{{placeholder_name}}",
        }
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8013, reload=False)
