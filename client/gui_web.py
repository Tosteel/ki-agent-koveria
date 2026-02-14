from __future__ import annotations

import json
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
    "ask_ionos_url": "http://127.0.0.1:8012/agent/askIonos",
    "api_key": "",
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
    user_id: Optional[str] = ""


class ChatMemoryMessage(BaseModel):
    role: Literal["user", "bot"]
    text: str = Field(..., min_length=1)
    downloadUrl: str = ""
    downloadLabel: str = ""
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


def _extract_api_base_url(ask_ionos_url: str) -> Optional[str]:
    try:
        parsed = urlsplit(ask_ionos_url.strip())
    except Exception:
        return None
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return None
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


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
        for k in ("ask_ionos_url", "api_key"):
            v = parsed.get(k)
            if isinstance(v, str):
                merged[k] = v
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


app = FastAPI(title="Koveria Web GUI", version="0.1.0")
MAX_HISTORY_ITEMS = 12


def _build_goal(message: str, history: Optional[List[ChatHistoryMessage]]) -> str:
    if not history:
        return message.strip()

    entries: List[str] = []
    for item in history[-MAX_HISTORY_ITEMS:]:
        role = "Nutzer" if item.role == "user" else "Assistent"
        text = item.text.strip()
        if text:
            entries.append(f"{role}: {text}")

    if not entries:
        return message.strip()

    context = "\n".join(entries)
    return (
        "Nutze den folgenden Chat-Verlauf als Kontext und beantworte danach die neue Frage.\n\n"
        f"{context}\n\n"
        f"Neue Frage: {message.strip()}"
    )


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
        "ask_ionos_url": req.ask_ionos_url.strip(),
        "api_key": (req.api_key or "").strip(),
    }
    if not settings["ask_ionos_url"]:
        raise HTTPException(status_code=422, detail="ask_ionos_url must not be empty")

    explicit_user_id = _sanitize_user_id(req.user_id) if (req.user_id or "").strip() else ""
    resolved_user_id = _resolve_user_id_from_api(settings["ask_ionos_url"], settings["api_key"])
    user_id = explicit_user_id or resolved_user_id or _get_active_user_id()
    user_id = _sanitize_user_id(user_id)

    _save_settings_for_user(user_id, settings)

    return JSONResponse({"ok": True, "user_id": user_id, "settings": settings})


@app.post("/api/chat")
def chat(req: ChatRequest) -> JSONResponse:
    user_id = _sanitize_user_id(req.user_id) if (req.user_id or "").strip() else _get_active_user_id()
    settings = _load_settings_for_user(user_id)
    url = settings["ask_ionos_url"].strip()
    api_key = settings.get("api_key", "").strip()

    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    goal = _build_goal(req.message, req.history)

    try:
        resp = requests.post(url, headers=headers, json={"goal": goal}, timeout=180)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"API request failed: {exc}") from exc

    if resp.status_code >= 400:
        snippet = resp.text[:500]
        raise HTTPException(status_code=resp.status_code, detail=f"API error: {snippet}")

    data: Dict[str, Any] = resp.json() if resp.content else {}
    answer = data.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        answer = json.dumps(data, ensure_ascii=False)

    return JSONResponse({"ok": True, "answer": answer, "raw": data})


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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8013, reload=False)
