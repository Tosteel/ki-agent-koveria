# uvicorn server.main:app --host 0.0.0.0 --port 8012 --reload

from __future__ import annotations

from fastapi import FastAPI, Depends, Response
from fastapi.security import HTTPAuthorizationCredentials
from typing import Any, Dict

from .core.logging import setup_logging
from .core.settings import Settings
from .core.models import (
    RagQueryRequest, RagQueryResponse,
    FileReadRequest, FileReadResponse,
    FileWriteRequest, FileWriteResponse,
    PdfExportRequest, PdfExportResponse,
    AgentRunRequest, AgentRunResponse,
)
from .deps import get_current_user, settings as dep_settings

from .services.rag_koveria import RagService
from .tools.filesystem import read_text, write_text
from .tools.pdf import export_text_pdf

from .agent.tool_registry import ToolRegistry, ToolContext
from .agent.orchestrator import Orchestrator

from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
security = HTTPBearer(auto_error=False)

app = FastAPI(title="ki-agent-koveria", version="0.1.0")

from .core.models import AgentAskRequest, AgentAskResponse
from .services.llm_openai import LlmRuntime
from server.services.llm_ionos import IonosLLM

from .agent.planner import Planner

@app.on_event("startup")
def _startup() -> None:
    setup_logging()


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


# ----------------------------- Phase 1: Direct APIs -----------------------------
from typing import Any, Dict

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


# ----------------------------- Phase 1: Agent (Tool Registry + Orchestrator) -----------------------------
def _build_registry() -> ToolRegistry:
    registry = ToolRegistry()

    def tool_read_file(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = FileReadRequest(**args)
        content = read_text(ctx.settings.user_work_dir(ctx.user_id), req.path, encoding=req.encoding)
        return FileReadResponse(path=req.path, content=content).model_dump()

    def tool_write_file(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = FileWriteRequest(**args)
        n = write_text(
            ctx.settings.user_work_dir(ctx.user_id),
            req.path,
            req.content,
            encoding=req.encoding,
            overwrite=req.overwrite,
        )
        return FileWriteResponse(path=req.path, bytes_written=n).model_dump()

    def tool_query_rag(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = RagQueryRequest(**args)
        service = RagService(ctx.settings.rag_base_url, ctx.api_key)
        return service.query(query=req.query, top_k=req.top_k)

    def tool_pdf_export(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = PdfExportRequest(**args)
        out = (ctx.settings.user_work_dir(ctx.user_id) / req.output_path.strip().lstrip("/")).resolve()
        size = export_text_pdf(out, title=req.title, text=req.text)
        return PdfExportResponse(output_path=req.output_path, bytes_written=size).model_dump()

    registry.register("read_file", tool_read_file)
    registry.register("write_file", tool_write_file)
    registry.register("query_rag", tool_query_rag)
    registry.register("pdf_export", tool_pdf_export)
    return registry


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
    )

    # Wenn steps explizit: ausführen
    if req.steps:
        outputs = orch.run_steps(ctx, [step.model_dump() for step in req.steps])
        ok = all(o.get("ok") for o in outputs) if outputs else True
        return AgentRunResponse(ok=ok, outputs=outputs)

    # Default Flow (Phase 1):
    # 1) query_rag (wenn rag_query gesetzt)
    # 2) write_file
    # 3) pdf_export
    outputs = []

    rag_result = {"query": "", "hits": []}
    if req.rag_query:
        o = orch.run_steps(ctx, [{"tool": "query_rag", "args": {"query": req.rag_query, "top_k": req.top_k}}])
        outputs.extend(o)
        if o and o[0].get("ok"):
            rag_result = o[0]["result"]

    # Ergebnistext zusammensetzen
    lines = []
    if req.rag_query:
        lines.append(f"RAG Query: {req.rag_query}")
        lines.append("")
        for i, h in enumerate(rag_result.get("hits", []), start=1):
            lines.append(f"[{i}] source={h.get('source')} score={h.get('score')}")
            lines.append(h.get("text", ""))
            lines.append("")

    text_out = "\n".join(lines).strip() or "Kein Inhalt."

    outputs.extend(orch.run_steps(ctx, [{"tool": "write_file", "args": {"path": req.write_path, "content": text_out, "overwrite": True}}]))
    outputs.extend(orch.run_steps(ctx, [{"tool": "pdf_export", "args": {"output_path": req.pdf_path, "title": req.pdf_title, "text": text_out}}]))

    ok = all(o.get("ok") for o in outputs) if outputs else True
    return AgentRunResponse(ok=ok, outputs=outputs)

@app.post("/agent/askOpenAI", response_model=AgentAskResponse)
def agent_askOpenAI(
    req: AgentAskRequest,
    user_id: str = Depends(get_current_user),
    s: Settings = Depends(dep_settings),
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> AgentAskResponse:
    _ensure_user_dirs(s, user_id)

    api_key = credentials.credentials
    registry = _build_registry()           # existiert bei dir bereits
    orch = Orchestrator(registry)          # existiert bei dir bereits
    ctx = ToolContext(user_id=user_id, settings=s, api_key=api_key)

    llm = LlmRuntime()
    planner = Planner(llm)

    steps = planner.create_steps(req.goal)

    # Optional: top_k/classification in query_rag step injizieren
    for st in steps:
        if st.get("tool") == "query_rag":
            st.setdefault("args", {})
            st["args"]["top_k"] = req.top_k
            if req.classification is not None:
                st["args"]["classification"] = req.classification

    tool_outputs = orch.run_steps(ctx, steps)

    if llm.enabled():
        answer = llm.final_answer(goal=req.goal, tool_outputs=tool_outputs)
    else:
        # Fallback: sehr kurze Antwort aus Outputs
        answer = str(tool_outputs)

    ok = all(o.get("ok") for o in tool_outputs) if tool_outputs else True
    return AgentAskResponse(ok=ok, goal=req.goal, steps=steps, tool_outputs=tool_outputs, answer=answer)

@app.post("/agent/askIonos", response_model=AgentAskResponse)
def agent_askIonos(
    req: AgentAskRequest,
    user_id: str = Depends(get_current_user),
    s: Settings = Depends(dep_settings),
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> AgentAskResponse:
    _ensure_user_dirs(s, user_id)

    api_key = credentials.credentials
    registry = _build_registry()           # existiert bei dir bereits
    orch = Orchestrator(registry)          # existiert bei dir bereits
    ctx = ToolContext(user_id=user_id, settings=s, api_key=api_key)

    llm = IonosLLM()
    planner = Planner(llm)

    steps = planner.create_steps(req.goal)

    # Optional: top_k/classification in query_rag step injizieren
    for st in steps:
        if st.get("tool") == "query_rag":
            st.setdefault("args", {})
            st["args"]["top_k"] = req.top_k
            if req.classification is not None:
                st["args"]["classification"] = req.classification

    tool_outputs = orch.run_steps(ctx, steps)

    if llm.enabled():
        answer = llm.final_answer(goal=req.goal, tool_outputs=tool_outputs)
    else:
        # Fallback: sehr kurze Antwort aus Outputs
        answer = str(tool_outputs)

    ok = all(o.get("ok") for o in tool_outputs) if tool_outputs else True
    return AgentAskResponse(ok=ok, goal=req.goal, steps=steps, tool_outputs=tool_outputs, answer=answer)