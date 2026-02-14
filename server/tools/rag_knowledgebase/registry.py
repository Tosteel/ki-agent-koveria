from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .models import RagQueryRequest
from .rag_knowledgebase import RagService


def _hit_text(hit: Dict[str, Any]) -> str:
    for key in ("text", "snippet", "content", "chunk", "page_content", "body"):
        val = hit.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _hit_source(hit: Dict[str, Any]) -> str:
    for key in ("source", "file", "document", "path", "url", "id"):
        val = hit.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return "unknown"


def _rag_result_to_text(query: str, rag_result: Dict[str, Any]) -> str:
    lines = [f"RAG Query: {query}", ""]
    for i, h in enumerate(rag_result.get("hits", []), start=1):
        if not isinstance(h, dict):
            continue
        source = _hit_source(h)
        score = h.get("score")
        snippet = _hit_text(h)
        lines.append(f"[{i}] source={source} score={score}")
        lines.append(snippet or "(kein Textausschnitt im Treffer enthalten)")
        lines.append("")
    return "\n".join(lines).strip() or "Kein Inhalt."


def register(registry: ToolRegistry) -> None:
    def tool_rag_knowledgebase(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = RagQueryRequest(**args)
        service = RagService(ctx.settings.rag_base_url, ctx.api_key)
        result = service.query(query=req.query, top_k=req.top_k, classification=req.classification)
        result["text"] = _rag_result_to_text(req.query, result)
        return result

    registry.register("rag_knowledgebase", tool_rag_knowledgebase, request_model=RagQueryRequest)
