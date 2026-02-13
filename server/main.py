# uvicorn server.main:app --host 0.0.0.0 --port 8012 --reload
from __future__ import annotations

from fastapi import FastAPI, Depends
from typing import Any, Dict

from .core.logging import setup_logging
from .core.settings import Settings
from .core.models import (
    RagQueryRequest, FileReadRequest, FileReadResponse,
    FileWriteRequest, FileWriteResponse,
    PdfExportRequest, PdfExportResponse,
    PptExportRequest, PptExportResponse,
    SearchGenerateJsonRequest,
    LlmSummaryRequest, LlmSummaryResponse,
    LlmComposeRequest, LlmComposeResponse,
    AgentRunRequest, AgentRunResponse,
)
from .deps import get_current_user, settings as dep_settings

from .tools.rag_koveria import RagService
from .tools.search_koveria import SearchService
from .tools.filesystem import read_text, write_text
from .tools.pdf import export_text_pdf
from .tools.powerpoint import export_text_pptx
from .tools.llm_summary import llm_summarize_text
from .tools.llm_compose import llm_compose_text

from .agent.tool_registry import ToolRegistry, ToolContext
from .agent.orchestrator import Orchestrator

from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
security = HTTPBearer(auto_error=False)

app = FastAPI(title="ki-agent-koveria", version="0.1.0")


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
            # query_rag liefert sowohl hits als auch text. Für API-Responses reichen hits;
            # text bleibt intern in tool_outputs_full für nachfolgende Steps verfügbar.
            if item["tool"] == "query_rag" and isinstance(payload.get("hits"), list):
                payload.pop("text", None)
            # llm_summarize liefert summary und text mit gleichem Inhalt.
            # Für API-Responses reicht summary; text bleibt intern verfügbar.
            if item["tool"] == "llm_summarize" and isinstance(payload.get("summary"), str):
                payload.pop("text", None)
            if item["tool"] == "llm_compose" and isinstance(payload.get("composed_text"), str):
                payload.pop("text", None)
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
        if tool == "query_rag":
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

from .core.models import AgentAskRequest, AgentAskResponse
from .services.llm_openai import LlmRuntime
from server.services.llm_ionos import IonosLLM

from .agent.planner import Planner

from dotenv import load_dotenv
load_dotenv()

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
        result = service.query(query=req.query, top_k=req.top_k, classification=req.classification)
        # payload-freundlich: nachfolgende Schritte (z.B. pdf_export) können direkt "text" verwenden
        result["text"] = _rag_result_to_text(req.query, result)
        return result

    def tool_pdf_export(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = PdfExportRequest(**args)
        out = (ctx.settings.user_work_dir(ctx.user_id) / req.output_path).resolve()
        size = export_text_pdf(out, title=req.title, text=req.text)
        return PdfExportResponse(output_path=req.output_path, bytes_written=size).model_dump()

    def tool_ppt_export(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = PptExportRequest(**args)
        out = (ctx.settings.user_work_dir(ctx.user_id) / req.output_path).resolve()
        result = export_text_pptx(
            out,
            title=req.title,
            text=req.text,
            use_llm_layout=req.use_llm_layout,
            allow_heuristic_fallback=req.allow_heuristic_fallback,
            goal=req.goal or ctx.goal,
            instruction=req.instruction,
            max_slides=req.max_slides,
            max_boxes_per_slide=req.max_boxes_per_slide,
        )
        return PptExportResponse(
            output_path=req.output_path,
            bytes_written=int(result.get("bytes_written") or 0),
            layout_mode=str(result.get("layout_mode") or "heuristic"),
        ).model_dump()

    def tool_llm_summarize(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = LlmSummaryRequest(**args)
        result = llm_summarize_text(
            text=req.text,
            goal=req.goal,
            instruction=req.instruction,
            max_chars=req.max_chars,
        )
        return LlmSummaryResponse(**result).model_dump()

    def tool_llm_compose(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = LlmComposeRequest(**args)
        result = llm_compose_text(
            text=req.text,
            goal=req.goal,
            instruction=req.instruction,
            max_chars=req.max_chars,
        )
        return LlmComposeResponse(**result).model_dump()

    def tool_search_web(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = SearchGenerateJsonRequest(**args)
        service = SearchService(ctx.settings.search_base_url, ctx.api_key)
        return service.search_generate_json(user_prompt=req.user_prompt)

    registry.register(
        "read_file",
        tool_read_file,
        request_model=FileReadRequest,
    )
    registry.register(
        "write_file",
        tool_write_file,
        request_model=FileWriteRequest,
    )
    registry.register(
        "query_rag",
        tool_query_rag,
        request_model=RagQueryRequest,
    )
    registry.register(
        "llm_summarize",
        tool_llm_summarize,
        request_model=LlmSummaryRequest,
    )
    registry.register(
        "llm_compose",
        tool_llm_compose,
        request_model=LlmComposeRequest,
    )
    registry.register(
        "pdf_export",
        tool_pdf_export,
        request_model=PdfExportRequest,
    )
    registry.register(
        "ppt_export",
        tool_ppt_export,
        request_model=PptExportRequest,
    )
    registry.register(
        "search_web",
        tool_search_web,
        request_model=SearchGenerateJsonRequest,
    )

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
        goal=req.rag_query or "",
    )

    # Wenn steps explizit: ausführen
    if req.steps:
        tool_outputs = orch.run_steps(ctx, [step.model_dump() for step in req.steps])
        ok = all(o.get("ok") for o in tool_outputs) if tool_outputs else True
        return AgentRunResponse(ok=ok, outputs=_compact_tool_outputs(tool_outputs))

    # Default Flow (Phase 1):
    # 1) query_rag (wenn rag_query gesetzt)
    # 2) write_file
    # 3) pdf_export
    tool_outputs = []

    rag_result = {"query": "", "hits": []}
    if req.rag_query:
        o = orch.run_steps(ctx, [{"tool": "query_rag", "args": {"query": req.rag_query, "top_k": req.top_k}}])
        tool_outputs.extend(o)
        if o and o[0].get("ok"):
            rag_result = o[0]["result"]

    text_out = _rag_result_to_text(req.rag_query or "", rag_result) if req.rag_query else "Kein Inhalt."

    tool_outputs.extend(orch.run_steps(ctx, [{"tool": "write_file", "args": {"path": req.write_path, "content": text_out, "overwrite": True}}]))
    tool_outputs.extend(orch.run_steps(ctx, [{"tool": "pdf_export", "args": {"output_path": req.pdf_path, "title": req.pdf_title, "text": text_out}}]))

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

    api_key = credentials.credentials
    registry = _build_registry()           # existiert bei dir bereits
    orch = Orchestrator(registry)          # existiert bei dir bereits
    ctx = ToolContext(user_id=user_id, settings=s, api_key=api_key, goal=req.goal)

    llm = LlmRuntime()
    planner = Planner(llm, registry)
    steps = planner.create_steps(goal=req.goal)
    steps = _inject_llm_summary_before_pdf(steps, req.goal)
    #steps = planner.create_steps(req.goal)

    # Optional: top_k/classification in query_rag step injizieren
    for st in steps:
        if st.get("tool") == "query_rag":
            st.setdefault("args", {})
            st["args"]["top_k"] = req.top_k
            if req.classification is not None:
                st["args"]["classification"] = req.classification

    tool_outputs_full = orch.run_steps(ctx, steps)
    tool_outputs = _compact_tool_outputs(tool_outputs_full)

    llm_view = _outputs_for_final_answer(tool_outputs_full)
    if llm.enabled():
        answer = llm.final_answer(goal=req.goal, tool_outputs=llm_view)
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

    api_key = credentials.credentials
    registry = _build_registry()           # existiert bei dir bereits
    orch = Orchestrator(registry)          # existiert bei dir bereits
    ctx = ToolContext(user_id=user_id, settings=s, api_key=api_key, goal=req.goal)

    llm = IonosLLM()
    planner = Planner(llm, registry)
    steps = planner.create_steps(goal=req.goal)
    steps = _inject_llm_summary_before_pdf(steps, req.goal)

    print("\n===== PLANNED STEPS =====")
    for i, s in enumerate(steps, 1):
        print(f"{i}. tool={s.get('tool')} args={s.get('args')}")
    print("=========================\n")

    # Optional: top_k/classification in query_rag step injizieren
    for st in steps:
        if st.get("tool") == "query_rag":
            st.setdefault("args", {})
            st["args"]["top_k"] = req.top_k
            if req.classification is not None:
                st["args"]["classification"] = req.classification

    tool_outputs_full = orch.run_steps(ctx, steps)
    tool_outputs = _compact_tool_outputs(tool_outputs_full)

    llm_view = _outputs_for_final_answer(tool_outputs_full)
    if llm.enabled():
        answer = llm.final_answer(goal=req.goal, tool_outputs=llm_view)
    else:
        # Fallback: sehr kurze Antwort aus Outputs
        answer = str(llm_view)

    print("\n===== FINAL ANSWER =====")
    print(answer)
    print("========================\n")

    ok = all(o.get("ok") for o in tool_outputs_full) if tool_outputs_full else True

    return AgentAskResponse(ok=ok, goal=req.goal, steps=steps, tool_outputs=tool_outputs, answer=answer)
