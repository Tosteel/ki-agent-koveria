# uvicorn client.gui_offerflow:app --host 0.0.0.0 --port 8014 --reload
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parent
HTML_PATH = APP_DIR / "gui_offerflow.html"
USER_DATA_DIR = APP_DIR / "data" / "users"
DEFAULT_USER_ID = "user1"

DEFAULT_SETTINGS: Dict[str, str] = {
    "api_base_url": "http://127.0.0.1:8012",
    "api_key": "",
}


class OfferflowSettingsRequest(BaseModel):
    api_base_url: str = Field(..., min_length=1)
    api_key: str = ""
    user_id: str = ""


class OfferflowStepRunRequest(BaseModel):
    step: int = Field(..., ge=1, le=7)
    payload: Dict[str, Any] = Field(default_factory=dict)
    user_id: str = ""


def _sanitize_user_id(user_id: str) -> str:
    raw = (user_id or "").strip()
    if not raw:
        return DEFAULT_USER_ID
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-")
    return cleaned or DEFAULT_USER_ID


def _settings_path(user_id: str) -> Path:
    safe_user = _sanitize_user_id(user_id)
    return USER_DATA_DIR / safe_user / "gui_offerflow_config.json"


def _load_settings(user_id: str) -> Dict[str, str]:
    p = _settings_path(user_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text(json.dumps(DEFAULT_SETTINGS, indent=2), encoding="utf-8")
        return dict(DEFAULT_SETTINGS)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    out = dict(DEFAULT_SETTINGS)
    out.update({k: str(v) for k, v in data.items() if k in out})
    return out


def _save_settings(user_id: str, cfg: Dict[str, str]) -> None:
    p = _settings_path(user_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


app = FastAPI(title="OfferFlow UI", version="0.1.0")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(HTML_PATH)


@app.get("/api/config")
def get_config(user_id: str = Query(default="")) -> Dict[str, Any]:
    uid = _sanitize_user_id(user_id)
    cfg = _load_settings(uid)
    return {"ok": True, "user_id": uid, **cfg}


@app.post("/api/config")
def save_config(req: OfferflowSettingsRequest) -> Dict[str, Any]:
    uid = _sanitize_user_id(req.user_id)
    cfg = {
        "api_base_url": req.api_base_url.rstrip("/"),
        "api_key": req.api_key,
    }
    _save_settings(uid, cfg)
    return {"ok": True, "user_id": uid, **cfg}


@app.post("/api/step/run")
def run_step(req: OfferflowStepRunRequest) -> Dict[str, Any]:
    uid = _sanitize_user_id(req.user_id)
    cfg = _load_settings(uid)

    base = cfg["api_base_url"].rstrip("/")
    url = f"{base}/offerflow/step-{req.step}/run"
    headers = {"Content-Type": "application/json"}
    api_key = cfg.get("api_key", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        resp = requests.post(url, headers=headers, json=req.payload, timeout=180)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Backend request failed: {exc}") from exc

    if resp.status_code >= 400:
        detail = resp.text
        try:
            detail_json = resp.json()
            detail = json.dumps(detail_json, ensure_ascii=False)
        except Exception:
            pass
        raise HTTPException(status_code=resp.status_code, detail=detail)

    try:
        data = resp.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Invalid backend JSON: {exc}") from exc

    return {"ok": True, "step": req.step, "data": data}
