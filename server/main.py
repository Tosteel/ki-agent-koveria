# uvicorn server.main:app --host 0.0.0.0 --port 8012 --reload
from __future__ import annotations

import json
from datetime import datetime, timezone
from fastapi import FastAPI, Depends, HTTPException
from typing import Any, Dict, List, Optional
from uuid import uuid4
from pathlib import Path

from .core.logging import setup_logging
from .core.settings import Settings, get_settings
from .core.models import (
    AgentRunRequest, AgentRunResponse,
    AgentAskRequest, AgentAskResponse,
)
from .deps import get_current_user, settings as dep_settings
from .auth import get_token_for_user

from .tools.rag_knowledgebase.models import RagQueryRequest
from .tools.filesystem.models import FileReadRequest, FileReadResponse, FileWriteRequest, FileWriteResponse
from .tools.pdf.models import PdfExportRequest, PdfExportResponse
from .tools.powerpoint.models import PptExportRequest, PptExportResponse
from .tools.mail.models import MailSendRequest, MailSendResponse
from .tools.search_multitable.models import SearchGenerateJsonRequest
from .tools.rag_knowledgebase import RagService
from .tools.search_multitable import SearchService
from .tools.filesystem import read_text, write_text
from .tools.pdf import export_text_pdf
from .tools.powerpoint import export_text_pptx
from .tools.mail import send_mail
from .tools.loader import register_all_tools
from .triggers import TriggerRegistry, TriggerRuntime, register_all_triggers
from .triggers.store import load_user_triggers, save_user_triggers

from .agent.tool_registry import ToolRegistry, ToolContext
from .agent.orchestrator import Orchestrator
from pydantic import BaseModel, Field

from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
security = HTTPBearer(auto_error=False)

app = FastAPI(title="ki-agent-koveria", version="0.1.0")


class TriggerCreateRequest(BaseModel):
    name: str = Field(..., min_length=1)
    trigger_type: str = Field(..., min_length=1)
    task_id: int = Field(..., ge=1)
    config: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class TriggerUpdateRequest(BaseModel):
    name: Optional[str] = None
    task_id: Optional[int] = Field(default=None, ge=1)
    config: Optional[Dict[str, Any]] = None
    enabled: Optional[bool] = None


class TaskMemorySyncRequest(BaseModel):
    tasks: List[Dict[str, Any]] = Field(default_factory=list)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _user_tasks_memory_path(s: Settings, user_id: str) -> Path:
    return s.user_dir(user_id) / "tasks_memory.json"


def _normalize_tasks_payload(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    tasks: List[Dict[str, Any]] = []
    for t in raw:
        if not isinstance(t, dict):
            continue
        task_id = int(t.get("id") or 0)
        text = str(t.get("text") or "").strip()
        if task_id <= 0 or not text:
            continue
        reruns_raw = t.get("reruns")
        reruns: List[Dict[str, Any]] = []
        if isinstance(reruns_raw, list):
            for r in reruns_raw:
                if not isinstance(r, dict):
                    continue
                answer = str(r.get("answer") or "").strip()
                if not answer:
                    continue
                reruns.append(
                    {
                        "answer": answer,
                        "created_at": str(r.get("created_at") or ""),
                    }
                )
        dialog_raw = t.get("dialog")
        dialog: List[Dict[str, Any]] = []
        if isinstance(dialog_raw, list):
            for m in dialog_raw:
                if not isinstance(m, dict):
                    continue
                msg_text = str(m.get("text") or "").strip()
                if not msg_text:
                    continue
                dialog.append(
                    {
                        "role": "user" if str(m.get("role") or "").strip().lower() == "user" else "bot",
                        "text": msg_text,
                        "timestamp": str(m.get("timestamp") or ""),
                    }
                )
        tasks.append(
            {
                "id": task_id,
                "title": str(t.get("title") or "").strip(),
                "text": text,
                "planned_steps": [str(s).strip() for s in (t.get("planned_steps") or []) if str(s).strip()],
                "planned_steps_text": str(t.get("planned_steps_text") or "").strip(),
                "created_at": str(t.get("created_at") or ""),
                "dialog": dialog,
                "reruns": reruns,
            }
        )
    return tasks


def _load_tasks_memory_for_user(s: Settings, user_id: str) -> Dict[str, Any]:
    path = _user_tasks_memory_path(s, user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return {"tasks": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"tasks": []}
    if not isinstance(data, dict):
        return {"tasks": []}
    return {"tasks": _normalize_tasks_payload(data.get("tasks"))}


def _save_tasks_memory_for_user(s: Settings, user_id: str, tasks: List[Dict[str, Any]]) -> None:
    path = _user_tasks_memory_path(s, user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "updated_at": _now_iso(),
        "tasks": _normalize_tasks_payload(tasks),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _first_nonempty_str(*values: Any) -> str:
    for v in values:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _extract_hit_source(hit: Dict[str, Any]) -> str:
    document_raw = hit.get("document")
    document_str = document_raw.strip() if isinstance(document_raw, str) else ""
    src = _first_nonempty_str(
        hit.get("source"),
        hit.get("file"),
        hit.get("file_name"),
        hit.get("filename"),
        hit.get("document"),
        hit.get("document_name"),
        hit.get("document_title"),
        hit.get("uri"),
        hit.get("path"),
        hit.get("link_source"),
        hit.get("link_server"),
        hit.get("source_url"),
        hit.get("url"),
        hit.get("document_id"),
        hit.get("id"),
    )
    if src:
        return src
    if document_str:
        return document_str

    for key in ("metadata", "meta", "payload", "document", "_source"):
        nested = hit.get(key)
        if isinstance(nested, str) and nested.strip():
            return nested.strip()
        if isinstance(nested, dict):
            src = _first_nonempty_str(
                nested.get("source"),
                nested.get("file"),
                nested.get("file_name"),
                nested.get("filename"),
                nested.get("document"),
                nested.get("document_name"),
                nested.get("document_title"),
                nested.get("uri"),
                nested.get("path"),
                nested.get("link_source"),
                nested.get("link_server"),
                nested.get("source_url"),
                nested.get("url"),
                nested.get("document_id"),
                nested.get("id"),
            )
            if src:
                return src
    return "unknown"


def _extract_hit_link(hit: Dict[str, Any]) -> str:
    link = _first_nonempty_str(
        hit.get("link_source"),
        hit.get("link_server"),
        hit.get("source_url"),
        hit.get("url"),
        hit.get("link"),
    )
    if link:
        return link

    for key in ("metadata", "meta", "payload", "document", "_source"):
        nested = hit.get(key)
        if isinstance(nested, dict):
            link = _first_nonempty_str(
                nested.get("link_source"),
                nested.get("link_server"),
                nested.get("source_url"),
                nested.get("url"),
                nested.get("link"),
            )
            if link:
                return link
    return ""


def _extract_hit_text(hit: Dict[str, Any]) -> str:
    txt = _first_nonempty_str(
        hit.get("text"),
        hit.get("snippet"),
        hit.get("content"),
        hit.get("chunk"),
        hit.get("page_content"),
        hit.get("body"),
    )
    if txt:
        return txt

    for key in ("metadata", "meta", "payload", "document", "_source"):
        nested = hit.get(key)
        if isinstance(nested, dict):
            txt = _first_nonempty_str(
                nested.get("text"),
                nested.get("snippet"),
                nested.get("content"),
                nested.get("chunk"),
                nested.get("page_content"),
                nested.get("body"),
            )
            if txt:
                return txt

    return ""


def _rag_result_to_text(query: str, rag_result: Dict[str, Any]) -> str:
    lines = [f"RAG Query: {query}", ""]
    for i, h in enumerate(rag_result.get("hits", []), start=1):
        source = _extract_hit_source(h if isinstance(h, dict) else {})
        link = _extract_hit_link(h if isinstance(h, dict) else {})
        score = h.get("score") if isinstance(h, dict) else None
        snippet = _extract_hit_text(h if isinstance(h, dict) else {})
        lines.append(f"[{i}] source={source} score={score}")
        if link and link != source:
            lines.append(f"link={link}")
        if snippet:
            lines.append(snippet)
        else:
            lines.append("(kein Textausschnitt im Treffer enthalten)")
        lines.append("")
    return "\n".join(lines).strip() or "Kein Inhalt."


def _search_result_to_text(user_prompt: str, result: Dict[str, Any]) -> str:
    if not isinstance(result, dict):
        return str(result or "")

    for key in ("text", "answer", "summary", "content", "markdown"):
        val = result.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    for key in ("rows", "results", "items", "data"):
        val = result.get(key)
        if isinstance(val, list) and val:
            lines = [f"Search Prompt: {user_prompt}", ""]
            for i, item in enumerate(val, start=1):
                if isinstance(item, dict):
                    lines.append(f"[{i}] " + ", ".join(f"{k}={v}" for k, v in item.items()))
                else:
                    lines.append(f"[{i}] {item}")
            return "\n".join(lines).strip()

    return json.dumps(result, ensure_ascii=False, indent=2)


def _wants_summary(goal: str) -> bool:
    g = (goal or "").lower()
    return any(k in g for k in ("zusammenfass", "fasse", "summary", "kurz"))


def _rewrite_summarize_to_compose(steps: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    out: list[Dict[str, Any]] = []
    for st in steps:
        tool = (st.get("tool") or "").strip()
        if tool == "llm_summarize":
            out.append({"tool": "llm_compose", "args": dict(st.get("args") or {})})
        else:
            out.append(st)
    return out


def _inject_llm_summary_before_pdf(steps: list[Dict[str, Any]], goal: str) -> list[Dict[str, Any]]:
    # Planner kann noch llm_summarize erzeugen; wir erzwingen llm_compose.
    steps = _rewrite_summarize_to_compose(steps)
    if not _wants_summary(goal):
        return steps
    if any((s.get("tool") or "").strip() == "llm_compose" for s in steps):
        return steps

    out: list[Dict[str, Any]] = []
    for st in steps:
        if (st.get("tool") or "").strip() == "pdf_export":
            args = dict(st.get("args") or {})
            source_text = args.get("text") or "{last.text}"
            out.append(
                {
                    "tool": "llm_compose",
                    "args": {
                        "text": source_text,
                        "goal": goal,
                        "instruction": f"Formuliere die relevanten Ergebnisse als kohärenten, gut lesbaren Text: {goal}",
                        "max_chars": 1800,
                    },
                }
            )
            args["text"] = "{last.text}"
            out.append({"tool": "pdf_export", "args": args})
        else:
            out.append(st)
    return out


def _compact_tool_outputs(tool_outputs: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    compact: list[Dict[str, Any]] = []
    for o in tool_outputs:
        item: Dict[str, Any] = {
            "step": o.get("step"),
            "tool": o.get("tool"),
            "ok": o.get("ok"),
        }
        payload = o.get("payload")
        if isinstance(payload, dict):
            payload = dict(payload)
        if o.get("ok"):
            # Response kompakt halten: nur payload zurückgeben (enthält bereits die result-Felder).
            item["payload"] = payload
        else:
            item["error"] = o.get("error")
            item["payload"] = payload
        compact.append(item)
    return compact


def _outputs_for_final_answer(tool_outputs: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    """
    Build a token-lean view for final_answer prompts.
    Keeps execution status + essential fields, drops verbose blobs.
    """
    compact = _compact_tool_outputs(tool_outputs)
    lean: list[Dict[str, Any]] = []
    for o in compact:
        item: Dict[str, Any] = {
            "step": o.get("step"),
            "tool": o.get("tool"),
            "ok": o.get("ok"),
        }
        payload = o.get("payload")
        if not isinstance(payload, dict):
            lean.append(item)
            continue

        p = dict(payload)
        tool = item["tool"]
        if tool == "rag_knowledgebase":
            # Large snippets are expensive in prompt tokens.
            p.pop("hits", None)
            if isinstance(payload.get("hits"), list):
                p["hit_count"] = len(payload["hits"])
        elif tool == "llm_summarize":
            # summary is enough for downstream natural-language answer.
            p.pop("usage", None)
            p.pop("model", None)
        elif tool == "llm_compose":
            # composed_text is enough for final answer.
            p.pop("usage", None)
            p.pop("model", None)

        item["payload"] = p
        if not o.get("ok"):
            item["error"] = o.get("error")
        lean.append(item)
    return lean


def _run_clarification_gate(llm: Any, goal: str) -> Dict[str, Any]:
    g = (goal or "").strip().lower()
    # Retrieval/Recherche soll nicht durch überstrenge Rückfragen blockiert werden.
    # Bei solchen Zielen lieber best-effort planen und ausführen.
    search_like_markers = [
        "suche",
        "such",
        "recherche",
        "finde",
        "in meinem wissen",
        "wissen",
        "knowledgebase",
        "rag",
        "websuche",
        "internet",
    ]
    if any(m in g for m in search_like_markers):
        return {"status": "ready", "normalized_goal": goal, "missing_fields": [], "questions": []}

    if not hasattr(llm, "enabled") or not llm.enabled():
        return {"status": "ready", "normalized_goal": goal, "missing_fields": [], "questions": []}
    if not hasattr(llm, "clarify_goal"):
        return {"status": "ready", "normalized_goal": goal, "missing_fields": [], "questions": []}

    try:
        out = llm.clarify_goal(goal=goal)
    except Exception:
        return {"status": "ready", "normalized_goal": goal, "missing_fields": [], "questions": []}

    status = out.get("status")
    if status not in {"ready", "needs_info"}:
        status = "ready"
    return {
        "status": status,
        "normalized_goal": str(out.get("normalized_goal") or goal),
        "missing_fields": list(out.get("missing_fields") or []),
        "questions": list(out.get("questions") or []),
    }


def _clarification_response(req_goal: str, gate: Dict[str, Any]) -> AgentAskResponse:
    questions = [str(q).strip() for q in (gate.get("questions") or []) if str(q).strip()]
    if not questions:
        questions = ["Welche Informationen fehlen genau, damit ich starten kann?"]
    answer = "Bevor ich starte, brauche ich noch:\n" + "\n".join(f"- {q}" for q in questions)
    return AgentAskResponse(
        ok=True,
        goal=req_goal,
        steps=[],
        tool_outputs=[],
        answer=answer,
        requires_user_input=True,
        missing_fields=list(gate.get("missing_fields") or []),
        questions=questions,
    )


def _history_to_context(history: Any, max_items: int = 12) -> str:
    if not isinstance(history, list):
        return ""
    lines: list[str] = []
    for item in history[-max_items:]:
        if not isinstance(item, dict):
            continue
        role_raw = str(item.get("role") or "").strip().lower()
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        role = "Nutzer" if role_raw == "user" else "Assistent"
        lines.append(f"{role}: {text}")
    return "\n".join(lines).strip()


def _goal_with_context(goal: str, history: Any) -> str:
    base = (goal or "").strip()
    if not base:
        return base
    hist = _history_to_context(history, max_items=12)
    if not hist:
        return base
    return f"Aktuelle Anfrage:\n{base}\n\nDialogverlauf (letzte 12 Nachrichten):\n{hist}"

from .services.llm_openai import LlmRuntime
from server.services.llm_ionos import IonosLLM

from .agent.planner import Planner

from dotenv import load_dotenv
load_dotenv()

@app.on_event("startup")
def _startup() -> None:
    setup_logging()
    s = get_settings()
    trigger_registry = _build_trigger_registry()
    runtime = TriggerRuntime(settings=s, registry=trigger_registry, step_executor=_execute_steps_for_trigger, poll_seconds=1.0)
    runtime.start()
    app.state.trigger_runtime = runtime


@app.on_event("shutdown")
def _shutdown() -> None:
    runtime = getattr(app.state, "trigger_runtime", None)
    if runtime is not None:
        runtime.stop()

def _ensure_user_dirs(s: Settings, user_id: str) -> None:
    s.user_work_dir(user_id).mkdir(parents=True, exist_ok=True)
    s.user_rag_dir(user_id).mkdir(parents=True, exist_ok=True)
    s.user_logs_dir(user_id).mkdir(parents=True, exist_ok=True)

# ----------------------------- Health / User -----------------------------
@app.get("/health")
def health(user_id: str = Depends(get_current_user)) -> Dict[str, str]:
    return {"status": "ok", "user": user_id}


@app.get("/user")
def user(user_id: str = Depends(get_current_user)) -> Dict[str, str]:
    return {"user_id": user_id}


@app.get("/tasks/memory")
def get_tasks_memory(
    user_id: str = Depends(get_current_user),
    s: Settings = Depends(dep_settings),
) -> Dict[str, Any]:
    _ensure_user_dirs(s, user_id)
    memory = _load_tasks_memory_for_user(s, user_id)
    return {"user_id": user_id, **memory}


@app.post("/tasks/memory/sync")
def sync_tasks_memory(
    req: TaskMemorySyncRequest,
    user_id: str = Depends(get_current_user),
    s: Settings = Depends(dep_settings),
) -> Dict[str, Any]:
    _ensure_user_dirs(s, user_id)
    tasks = _normalize_tasks_payload(req.tasks)
    _save_tasks_memory_for_user(s, user_id, tasks)
    return {"ok": True, "user_id": user_id, "count": len(tasks)}


def _get_trigger_runtime() -> TriggerRuntime:
    runtime = getattr(app.state, "trigger_runtime", None)
    if runtime is None:
        raise RuntimeError("trigger runtime not initialized")
    return runtime


@app.get("/triggers/types")
def trigger_types() -> Dict[str, Any]:
    reg = _build_trigger_registry()
    return {"types": reg.available_types()}


@app.get("/triggers")
def list_triggers(
    user_id: str = Depends(get_current_user),
    s: Settings = Depends(dep_settings),
) -> Dict[str, Any]:
    _ensure_user_dirs(s, user_id)
    payload = load_user_triggers(s, user_id)
    triggers = payload.get("triggers") if isinstance(payload.get("triggers"), list) else []
    return {"user_id": user_id, "triggers": triggers}


@app.post("/triggers")
def create_trigger(
    req: TriggerCreateRequest,
    user_id: str = Depends(get_current_user),
    s: Settings = Depends(dep_settings),
) -> Dict[str, Any]:
    _ensure_user_dirs(s, user_id)
    reg = _build_trigger_registry()
    # validate trigger type + config upfront
    reg.create_instance(req.trigger_type.strip(), req.config or {})

    payload = load_user_triggers(s, user_id)
    triggers = payload.get("triggers") if isinstance(payload.get("triggers"), list) else []
    trigger_id = str(uuid4())
    item = {
        "id": trigger_id,
        "name": req.name.strip(),
        "trigger_type": req.trigger_type.strip(),
        "task_id": int(req.task_id),
        "config": req.config or {},
        "enabled": bool(req.enabled),
        "created_at": _now_iso(),
        "last_fired_at": "",
        "last_error": "",
    }
    triggers.append(item)
    save_user_triggers(s, user_id, triggers)
    return {"ok": True, "trigger": item}


@app.patch("/triggers/{trigger_id}")
def update_trigger(
    trigger_id: str,
    req: TriggerUpdateRequest,
    user_id: str = Depends(get_current_user),
    s: Settings = Depends(dep_settings),
) -> Dict[str, Any]:
    _ensure_user_dirs(s, user_id)
    payload = load_user_triggers(s, user_id)
    triggers = payload.get("triggers") if isinstance(payload.get("triggers"), list) else []
    updated: Optional[Dict[str, Any]] = None
    for t in triggers:
        if str(t.get("id") or "") != trigger_id:
            continue
        if req.name is not None:
            t["name"] = req.name.strip()
        if req.task_id is not None:
            t["task_id"] = int(req.task_id)
        if req.config is not None:
            # validate config for this trigger type
            reg = _build_trigger_registry()
            reg.create_instance(str(t.get("trigger_type") or ""), req.config or {})
            t["config"] = req.config or {}
        if req.enabled is not None:
            t["enabled"] = bool(req.enabled)
        updated = t
        break
    if updated is None:
        return {"ok": False, "error": "trigger_not_found"}

    save_user_triggers(s, user_id, triggers)
    return {"ok": True, "trigger": updated}


@app.delete("/triggers/{trigger_id}")
def delete_trigger(
    trigger_id: str,
    user_id: str = Depends(get_current_user),
    s: Settings = Depends(dep_settings),
) -> Dict[str, Any]:
    _ensure_user_dirs(s, user_id)
    payload = load_user_triggers(s, user_id)
    triggers = payload.get("triggers") if isinstance(payload.get("triggers"), list) else []
    kept: List[Dict[str, Any]] = []
    deleted = False
    for t in triggers:
        if str(t.get("id") or "") == trigger_id:
            deleted = True
            continue
        kept.append(t)
    if not deleted:
        return {"ok": False, "error": "trigger_not_found"}
    save_user_triggers(s, user_id, kept)
    return {"ok": True, "trigger_id": trigger_id}


@app.post("/triggers/{trigger_id}/run-now")
def run_trigger_now(
    trigger_id: str,
    user_id: str = Depends(get_current_user),
) -> Dict[str, Any]:
    runtime = _get_trigger_runtime()
    try:
        result = runtime.run_trigger_now(user_id=user_id, trigger_id=trigger_id)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": bool(result.get("ok")), "result": result}


# ----------------------------- Phase 1: Direct APIs -----------------------------
@app.post("/rag/query")
def rag_query(
    req: RagQueryRequest,
    user_id: str = Depends(get_current_user),
    s: Settings = Depends(dep_settings),
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Dict[str, Any]:
    _ensure_user_dirs(s, user_id)

    api_key = credentials.credentials
    service = RagService(s.rag_base_url, api_key)

    data = service.query(query=req.query, top_k=req.top_k, classification=req.classification)
    return data


@app.post("/search/generate_json")
def search_generate_json(
    req: SearchGenerateJsonRequest,
    user_id: str = Depends(get_current_user),
    s: Settings = Depends(dep_settings),
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Dict[str, Any]:
    _ensure_user_dirs(s, user_id)
    service = SearchService(s.search_base_url, credentials.credentials)
    return service.search_generate_json(user_prompt=req.user_prompt)


@app.post("/files/read", response_model=FileReadResponse)
def files_read(
    req: FileReadRequest,
    user_id: str = Depends(get_current_user),
    s: Settings = Depends(dep_settings),
) -> FileReadResponse:
    _ensure_user_dirs(s, user_id)
    content = read_text(s.user_work_dir(user_id), req.path, encoding=req.encoding)
    return FileReadResponse(path=req.path, content=content)


@app.post("/files/write", response_model=FileWriteResponse)
def files_write(
    req: FileWriteRequest,
    user_id: str = Depends(get_current_user),
    s: Settings = Depends(dep_settings),
) -> FileWriteResponse:
    _ensure_user_dirs(s, user_id)
    n = write_text(
        s.user_work_dir(user_id),
        req.path,
        req.content,
        encoding=req.encoding,
        overwrite=req.overwrite,
    )
    return FileWriteResponse(path=req.path, bytes_written=n)


@app.post("/pdf/export", response_model=PdfExportResponse)
def pdf_export(
    req: PdfExportRequest,
    user_id: str = Depends(get_current_user),
    s: Settings = Depends(dep_settings),
) -> PdfExportResponse:
    _ensure_user_dirs(s, user_id)
    out = (s.user_work_dir(user_id) / req.output_path.strip().lstrip("/")).resolve()
    size = export_text_pdf(out, title=req.title, text=req.text)
    return PdfExportResponse(output_path=req.output_path, bytes_written=size)


@app.post("/ppt/export", response_model=PptExportResponse)
def ppt_export(
    req: PptExportRequest,
    user_id: str = Depends(get_current_user),
    s: Settings = Depends(dep_settings),
) -> PptExportResponse:
    _ensure_user_dirs(s, user_id)
    out = (s.user_work_dir(user_id) / req.output_path.strip().lstrip("/")).resolve()
    result = export_text_pptx(
        out,
        title=req.title,
        text=req.text,
        use_llm_layout=req.use_llm_layout,
        allow_heuristic_fallback=req.allow_heuristic_fallback,
        goal=req.goal,
        instruction=req.instruction,
        max_slides=req.max_slides,
        max_boxes_per_slide=req.max_boxes_per_slide,
    )
    return PptExportResponse(
        output_path=req.output_path,
        bytes_written=int(result.get("bytes_written") or 0),
        layout_mode=str(result.get("layout_mode") or "heuristic"),
    )


@app.post("/mail/send", response_model=MailSendResponse)
def mail_send(
    req: MailSendRequest,
    user_id: str = Depends(get_current_user),
    s: Settings = Depends(dep_settings),
) -> MailSendResponse:
    _ensure_user_dirs(s, user_id)
    result = send_mail(
        to=req.to,
        subject=req.subject,
        body=req.body,
        attachment_paths=req.attachment_paths,
        work_dir=s.user_work_dir(user_id),
        cc=req.cc,
        bcc=req.bcc,
        from_email=req.from_email,
        reply_to=req.reply_to,
        is_html=req.is_html,
    )
    return MailSendResponse(**result)


# ----------------------------- Phase 1: Agent (Tool Registry + Orchestrator) -----------------------------
def _build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    return register_all_tools(registry)


def _build_trigger_registry() -> TriggerRegistry:
    registry = TriggerRegistry()
    return register_all_triggers(registry)


def _execute_steps_for_trigger(user_id: str, steps: List[Dict[str, Any]], goal: str) -> List[Dict[str, Any]]:
    s = get_settings()
    _ensure_user_dirs(s, user_id)
    registry = _build_registry()
    orch = Orchestrator(registry)
    token = get_token_for_user(user_id)
    ctx = ToolContext(user_id=user_id, settings=s, api_key=token, goal=goal)
    return orch.run_steps(ctx, steps)



@app.post("/agent/run", response_model=AgentRunResponse)
def agent_run(
    req: AgentRunRequest,
    user_id: str = Depends(get_current_user),
    s: Settings = Depends(dep_settings),
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> AgentRunResponse:
    _ensure_user_dirs(s, user_id)

    registry = _build_registry()
    orch = Orchestrator(registry)
    ctx = ToolContext(
        user_id=user_id,
        settings=s,
        api_key=credentials.credentials,  # ← wichtig
        goal="",
    )

    # Nur explizite Steps ausführen; kein impliziter Default-Flow.
    if not req.steps:
        raise HTTPException(status_code=422, detail="steps must not be empty for /agent/run")

    tool_outputs = orch.run_steps(ctx, [step.model_dump() for step in req.steps])
    ok = all(o.get("ok") for o in tool_outputs) if tool_outputs else True
    return AgentRunResponse(ok=ok, outputs=_compact_tool_outputs(tool_outputs))

@app.post("/agent/askOpenAI", response_model=AgentAskResponse)
def agent_askOpenAI(
    req: AgentAskRequest,
    user_id: str = Depends(get_current_user),
    s: Settings = Depends(dep_settings),
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> AgentAskResponse:
    _ensure_user_dirs(s, user_id)
    goal_ctx = _goal_with_context(req.goal, req.history)

    api_key = credentials.credentials
    registry = _build_registry()           # existiert bei dir bereits
    orch = Orchestrator(registry)          # existiert bei dir bereits
    ctx = ToolContext(user_id=user_id, settings=s, api_key=api_key, goal=goal_ctx)

    llm = LlmRuntime()
    gate = _run_clarification_gate(llm, goal_ctx)
    if gate["status"] == "needs_info":
        return _clarification_response(req.goal, gate)

    effective_goal = gate["normalized_goal"]
    ctx.goal = effective_goal
    planner = Planner(llm, registry)
    steps = planner.create_steps(goal=effective_goal)
    steps = _inject_llm_summary_before_pdf(steps, effective_goal)
    #steps = planner.create_steps(req.goal)

    tool_outputs_full = orch.run_steps(ctx, steps)
    tool_outputs = _compact_tool_outputs(tool_outputs_full)

    llm_view = _outputs_for_final_answer(tool_outputs_full)
    if llm.enabled():
        answer = llm.final_answer(goal=effective_goal, tool_outputs=llm_view)
    else:
        # Fallback: sehr kurze Antwort aus Outputs
        answer = str(llm_view)

    ok = all(o.get("ok") for o in tool_outputs_full) if tool_outputs_full else True
    return AgentAskResponse(ok=ok, goal=req.goal, steps=steps, tool_outputs=tool_outputs, answer=answer)

@app.post("/agent/askIonos", response_model=AgentAskResponse)
def agent_askIonos(
    req: AgentAskRequest,
    user_id: str = Depends(get_current_user),
    s: Settings = Depends(dep_settings),
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> AgentAskResponse:
    _ensure_user_dirs(s, user_id)
    goal_ctx = _goal_with_context(req.goal, req.history)

    api_key = credentials.credentials
    registry = _build_registry()           # existiert bei dir bereits
    orch = Orchestrator(registry)          # existiert bei dir bereits
    ctx = ToolContext(user_id=user_id, settings=s, api_key=api_key, goal=goal_ctx)

    llm = IonosLLM()
    gate = _run_clarification_gate(llm, goal_ctx)

    print("\n===== CLARIFICATION =====")
    print(f"status={gate.get('status')}")
    print(f"normalized_goal={gate.get('normalized_goal')}")
    print(f"missing_fields={gate.get('missing_fields')}")
    print(f"questions={gate.get('questions')}")
    print("=========================\n")

    if gate["status"] == "needs_info":
        return _clarification_response(req.goal, gate)

    effective_goal = gate["normalized_goal"]
    ctx.goal = effective_goal
    planner = Planner(llm, registry)
    steps = planner.create_steps(goal=effective_goal)
    steps = _inject_llm_summary_before_pdf(steps, effective_goal)

    print("\n===== PLANNED STEPS =====")
    for i, s in enumerate(steps, 1):
        print(f"{i}. tool={s.get('tool')} args={s.get('args')}")
    print("=========================\n")

    tool_outputs_full = orch.run_steps(ctx, steps)
    tool_outputs = _compact_tool_outputs(tool_outputs_full)

    llm_view = _outputs_for_final_answer(tool_outputs_full)
    if llm.enabled():
        answer = llm.final_answer(goal=effective_goal, tool_outputs=llm_view)
    else:
        # Fallback: sehr kurze Antwort aus Outputs
        answer = str(llm_view)

    print("\n===== FINAL ANSWER =====")
    print(answer)
    print("========================\n")

    ok = all(o.get("ok") for o in tool_outputs_full) if tool_outputs_full else True

    return AgentAskResponse(ok=ok, goal=req.goal, steps=steps, tool_outputs=tool_outputs, answer=answer)
