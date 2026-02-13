from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parent
HTML_PATH = APP_DIR / "gui_web.html"
LOGO_PATH = APP_DIR / "assets" / "koveria_logo.png"
BOT_AVATAR_PATH = APP_DIR / "assets" / "bot-avatar.png"
SETTINGS_PATH = APP_DIR / "gui_web_config.json"

DEFAULT_SETTINGS: Dict[str, str] = {
    "ask_ionos_url": "http://127.0.0.1:8012/agent/askIonos",
    "api_key": "",
}


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)


class SettingsRequest(BaseModel):
    ask_ionos_url: str = Field(..., min_length=1)
    api_key: Optional[str] = ""


def _load_settings() -> Dict[str, str]:
    if not SETTINGS_PATH.exists():
        SETTINGS_PATH.write_text(json.dumps(DEFAULT_SETTINGS, ensure_ascii=False, indent=2), encoding="utf-8")
        return dict(DEFAULT_SETTINGS)
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    merged = dict(DEFAULT_SETTINGS)
    if isinstance(data, dict):
        for k in ("ask_ionos_url", "api_key"):
            v = data.get(k)
            if isinstance(v, str):
                merged[k] = v
    return merged


def _save_settings(settings: Dict[str, str]) -> None:
    SETTINGS_PATH.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")


app = FastAPI(title="Koveria Web GUI", version="0.1.0")


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


@app.get("/api/settings")
def get_settings() -> JSONResponse:
    return JSONResponse(_load_settings())


@app.post("/api/settings")
def set_settings(req: SettingsRequest) -> JSONResponse:
    settings = {
        "ask_ionos_url": req.ask_ionos_url.strip(),
        "api_key": (req.api_key or "").strip(),
    }
    if not settings["ask_ionos_url"]:
        raise HTTPException(status_code=422, detail="ask_ionos_url must not be empty")
    _save_settings(settings)
    return JSONResponse({"ok": True, "settings": settings})


@app.post("/api/chat")
def chat(req: ChatRequest) -> JSONResponse:
    settings = _load_settings()
    url = settings["ask_ionos_url"].strip()
    api_key = settings.get("api_key", "").strip()

    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        resp = requests.post(url, headers=headers, json={"goal": req.message.strip()}, timeout=180)
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8013, reload=False)
