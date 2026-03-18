# uvicorn client.gui_matchup:app --host 0.0.0.0 --port 8015 --reload
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

APP_DIR = Path(__file__).resolve().parent
HTML_PATH = APP_DIR / "gui_matchup.html"

BACKEND_BASE_URL = os.getenv("GUI_MATCHUP_API_BASE_URL", "http://127.0.0.1:8012").rstrip("/")
BACKEND_API_KEY = os.getenv("GUI_MATCHUP_API_KEY", "a9c48f5a-446c-4ee3-84ec-e4d1b7bee79c").strip()
PROVIDER = os.getenv("GUI_MATCHUP_PROVIDER", "ionos").strip() or "ionos"

app = FastAPI(title="Startup Matchup UI", version="0.1.0")


def _headers() -> Dict[str, str]:
    headers = {"Accept": "application/json"}
    if BACKEND_API_KEY:
        headers["Authorization"] = f"Bearer {BACKEND_API_KEY}"
    return headers


def _raise_backend_error(resp: requests.Response, prefix: str) -> None:
    detail = resp.text
    try:
        detail = resp.json()
    except Exception:
        pass
    raise HTTPException(status_code=resp.status_code, detail=f"{prefix}: {detail}")


def _upload_workshop_file(file: UploadFile) -> Dict[str, Any]:
    url = f"{BACKEND_BASE_URL}/tools/files/upload"
    files = {
        "file": (
            file.filename or "workshop_upload.bin",
            file.file,
            file.content_type or "application/octet-stream",
        )
    }
    try:
        resp = requests.post(url, headers=_headers(), files=files, timeout=180)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Upload failed: {exc}") from exc
    if resp.status_code >= 400:
        _raise_backend_error(resp, "Upload failed")
    try:
        return resp.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Invalid upload response JSON: {exc}") from exc


def _build_steps(
    *,
    upload_path: str,
    company_name_step2: str,
    company_name_step9: str,
    report_year: Optional[int],
    created_by: str,
    startup_count: int,
    recipient_email: str,
) -> List[Dict[str, Any]]:
    output_name = f"startup_matchup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    subject = f"Startup Matchup Report - {company_name_step9 or company_name_step2}"
    if report_year:
        subject = f"{subject} ({report_year})"
    return [
        {
            "tool": "startup_matchup_step_1_workshop_analysis",
            "args": {
                "workshop_document_path": upload_path,
                "provider": PROVIDER,
            },
        },
        {
            "tool": "startup_matchup_step_2_company_profile",
            "args": {
                "workshop_analysis": "{{steps[0].payload.workshop_analysis}}",
                "company_name": company_name_step2,
                "provider": PROVIDER,
            },
        },
        {
            "tool": "startup_matchup_step_3_gap_analysis",
            "args": {
                "workshop_analysis": "{{steps[0].payload.workshop_analysis}}",
                "company_profile": "{{steps[1].payload.company_profile}}",
                "provider": PROVIDER,
                "max_queries": 16,
            },
        },
        {
            "tool": "startup_matchup_step_4_startup_search",
            "args": {
                "gap_analysis": "{{steps[2].payload.gap_analysis}}",
                "max_queries": 16,
                "per_query_results": 8,
                "brave_enable_research": False,
            },
        },
        {
            "tool": "startup_matchup_step_4_1_startup_structuring",
            "args": {
                "startup_candidates_raw": "{{steps[3].payload.startup_candidates_raw}}",
                "provider": PROVIDER,
            },
        },
        {
            "tool": "startup_matchup_step_5_startup_ranking",
            "args": {
                "startup_structured_list": "{{steps[4].payload.startup_structured_list}}",
                "company_profile": "{{steps[1].payload.company_profile}}",
                "gap_analysis": "{{steps[2].payload.gap_analysis}}",
                "top_k": startup_count,
                "provider": PROVIDER,
            },
        },
        {
            "tool": "startup_matchup_step_6_startup_deep_research",
            "args": {
                "startup_ranked_list": "{{steps[5].payload.startup_ranked_list}}",
                "top_n": startup_count,
                "brave_enable_research": False,
            },
        },
        {
            "tool": "startup_matchup_step_7_startup_profiles",
            "args": {
                "startup_deep_profiles_raw": "{{steps[6].payload.startup_deep_profiles_raw}}",
                "gap_analysis": "{{steps[2].payload.gap_analysis}}",
                "startup_ranked_list": "{{steps[5].payload.startup_ranked_list}}",
                "provider": PROVIDER,
            },
        },
        {
            "tool": "startup_matchup_step_8_final_report",
            "args": {
                "workshop_analysis": "{{steps[0].payload.workshop_analysis}}",
                "company_profile": "{{steps[1].payload.company_profile}}",
                "gap_analysis": "{{steps[2].payload.gap_analysis}}",
                "startup_ranked_list": "{{steps[5].payload.startup_ranked_list}}",
                "startup_profiles": "{{steps[7].payload.startup_profiles}}",
                "provider": PROVIDER,
                "top_k": startup_count,
            },
        },
        {
            "tool": "startup_matchup_step_9_pdf_report",
            "args": {
                "final_report": "{{steps[8].payload.final_report}}",
                "output_path": output_name,
                "company_name": company_name_step9,
                "report_year": report_year,
                "created_by": created_by,
            },
        },
        {
            "tool": "mail_send",
            "args": {
                "to": [recipient_email],
                "subject": subject,
                "body": "Der Startup-Matchup-Bericht ist fertig. Das PDF ist im Anhang.",
                "attachments": ["{{steps[9].payload.pdf_report.output_path}}"],
            },
        },
    ]


def _run_agent(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    url = f"{BACKEND_BASE_URL}/agent/run"
    payload = {"log_label": "MATCHUP GUI RUN 1-9", "steps": steps}
    headers = {"Content-Type": "application/json", **_headers()}
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=1200)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"agent/run failed: {exc}") from exc
    if resp.status_code >= 400:
        _raise_backend_error(resp, "agent/run failed")
    try:
        return resp.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Invalid agent/run response JSON: {exc}") from exc


def _extract_pdf_output_path(agent_resp: Dict[str, Any]) -> str:
    outputs = agent_resp.get("outputs") if isinstance(agent_resp.get("outputs"), list) else []
    for entry in reversed(outputs):
        if not isinstance(entry, dict):
            continue
        if str(entry.get("tool") or "").strip() != "startup_matchup_step_9_pdf_report":
            continue
        payload = entry.get("payload") if isinstance(entry.get("payload"), dict) else {}
        pdf = payload.get("pdf_report") if isinstance(payload.get("pdf_report"), dict) else {}
        p = str(pdf.get("output_path") or "").strip()
        if p:
            return p
    return ""


@app.get("/")
def index() -> FileResponse:
    return FileResponse(HTML_PATH)


@app.post("/api/report/create")
def create_report(
    workshop_file: UploadFile = File(...),
    company_name_step2: str = Form(...),
    company_name_step9: str = Form(...),
    report_year: int = Form(...),
    created_by: str = Form(...),
    startup_count: int = Form(...),
    recipient_email: str = Form(...),
) -> Dict[str, Any]:
    if not BACKEND_API_KEY:
        raise HTTPException(
            status_code=400,
            detail="GUI_MATCHUP_API_KEY is not set. Please export a valid backend API key.",
        )
    if startup_count < 5 or startup_count > 30:
        raise HTTPException(status_code=422, detail="startup_count must be between 5 and 30.")
    if "@" not in recipient_email or "." not in recipient_email.split("@")[-1]:
        raise HTTPException(status_code=422, detail="recipient_email is invalid.")

    upload_data = _upload_workshop_file(workshop_file)
    upload_path = str(upload_data.get("upload_path") or "").strip()
    if not upload_path:
        raise HTTPException(status_code=502, detail="Upload response has no upload_path.")

    steps = _build_steps(
        upload_path=upload_path,
        company_name_step2=company_name_step2.strip(),
        company_name_step9=company_name_step9.strip(),
        report_year=report_year,
        created_by=created_by.strip(),
        startup_count=startup_count,
        recipient_email=recipient_email.strip(),
    )
    agent_data = _run_agent(steps)
    pdf_output_path = _extract_pdf_output_path(agent_data)

    return {
        "ok": True,
        "upload_path": upload_path,
        "pdf_output_path": pdf_output_path,
        "agent_response": agent_data,
    }
