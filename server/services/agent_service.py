from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from server.core.settings import Settings
from server.tools.loader import register_all_tools
from server.agent.tool_registry import ToolRegistry, ToolContext
from server.agent.orchestrator import Orchestrator
from server.services.llm_openai import LlmRuntime
from server.services.llm_ionos import IonosLLM


def first_nonempty_str(*values: Any) -> str:
    for v in values:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def extract_hit_source(hit: Dict[str, Any]) -> str:
    document_raw = hit.get("document")
    document_str = document_raw.strip() if isinstance(document_raw, str) else ""
    src = first_nonempty_str(
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
            src = first_nonempty_str(
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


def extract_hit_link(hit: Dict[str, Any]) -> str:
    link = first_nonempty_str(
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
            link = first_nonempty_str(
                nested.get("link_source"),
                nested.get("link_server"),
                nested.get("source_url"),
                nested.get("url"),
                nested.get("link"),
            )
            if link:
                return link
    return ""


def extract_hit_text(hit: Dict[str, Any]) -> str:
    txt = first_nonempty_str(
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
            txt = first_nonempty_str(
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


def rag_result_to_text(query: str, rag_result: Dict[str, Any]) -> str:
    lines = [f"RAG Query: {query}", ""]
    for i, h in enumerate(rag_result.get("hits", []), start=1):
        source = extract_hit_source(h if isinstance(h, dict) else {})
        link = extract_hit_link(h if isinstance(h, dict) else {})
        score = h.get("score") if isinstance(h, dict) else None
        snippet = extract_hit_text(h if isinstance(h, dict) else {})
        lines.append(f"[{i}] source={source} score={score}")
        if link and link != source:
            lines.append(f"link={link}")
        if snippet:
            lines.append(snippet)
        else:
            lines.append("(kein Textausschnitt im Treffer enthalten)")
        lines.append("")
    return "\n".join(lines).strip() or "Kein Inhalt."


def search_result_to_text(user_prompt: str, result: Dict[str, Any]) -> str:
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


def wants_summary(goal: str) -> bool:
    g = (goal or "").lower()
    return any(k in g for k in ("zusammenfass", "fasse", "summary", "kurz"))


def rewrite_summarize_to_compose(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for st in steps:
        tool = (st.get("tool") or "").strip()
        if tool == "llm_summarize":
            out.append({"tool": "llm_compose", "args": dict(st.get("args") or {})})
        else:
            out.append(st)
    return out


def inject_llm_summary_before_pdf(steps: List[Dict[str, Any]], goal: str) -> List[Dict[str, Any]]:
    steps = rewrite_summarize_to_compose(steps)
    if not wants_summary(goal):
        return steps
    if any((s.get("tool") or "").strip() == "llm_compose" for s in steps):
        return steps

    out: List[Dict[str, Any]] = []
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


def compact_tool_outputs(tool_outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compact: List[Dict[str, Any]] = []
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
            item["payload"] = payload
        else:
            item["error"] = o.get("error")
            item["payload"] = payload
        compact.append(item)
    return compact


def sanitize_execution_steps(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for st in steps:
        if not isinstance(st, dict):
            continue
        tool = str(st.get("tool") or "").strip()
        args = st.get("args") if isinstance(st.get("args"), dict) else {}
        args_out = dict(args)
        if tool == "pdf_export":
            output_path = str(args_out.get("output_path") or "").strip()
            if not output_path.lower().endswith(".pdf"):
                args_out["output_path"] = "result.pdf"
        elif tool == "ppt_export":
            output_path = str(args_out.get("output_path") or "").strip()
            if not output_path.lower().endswith(".pptx"):
                args_out["output_path"] = "result.pptx"
        out.append({"tool": tool, "args": args_out})
    return out


def outputs_for_final_answer(tool_outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compact = compact_tool_outputs(tool_outputs)
    lean: List[Dict[str, Any]] = []
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
            p.pop("hits", None)
            if isinstance(payload.get("hits"), list):
                p["hit_count"] = len(payload["hits"])
        elif tool == "llm_summarize":
            p.pop("usage", None)
            p.pop("model", None)
        elif tool == "llm_compose":
            p.pop("usage", None)
            p.pop("model", None)

        item["payload"] = p
        if not o.get("ok"):
            item["error"] = o.get("error")
        lean.append(item)
    return lean


def extract_execution_answer(tool_outputs: List[Dict[str, Any]]) -> str:
    for out in reversed(tool_outputs):
        if not isinstance(out, dict) or not out.get("ok"):
            continue
        payload = out.get("payload")
        if not isinstance(payload, dict):
            continue
        for key in ("composed_text", "text", "summary", "answer", "message"):
            val = payload.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return "Ausführung abgeschlossen."


def run_clarification_gate(llm: Any, goal: str) -> Dict[str, Any]:
    g = (goal or "").strip().lower()
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


def history_to_context(history: Any, max_items: int = 12) -> str:
    if not isinstance(history, list):
        return ""
    lines: List[str] = []
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


def goal_with_context(goal: str, history: Any) -> str:
    base = (goal or "").strip()
    if not base:
        return base
    hist = history_to_context(history, max_items=12)
    if not hist:
        return base
    return f"Aktuelle Anfrage:\n{base}\n\nDialogverlauf (letzte 12 Nachrichten):\n{hist}"


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    return register_all_tools(registry)


def provider_key(provider: str) -> str:
    p = str(provider or "").strip().lower()
    if p in {"openai", "ionos"}:
        return p
    return "ionos"


def llm_for_provider(provider: str) -> Any:
    p = provider_key(provider)
    if p == "openai":
        return LlmRuntime()
    return IonosLLM()


def run_steps_internal(
    *,
    user_id: str,
    settings: Settings,
    api_key: str,
    goal: str,
    steps: List[Dict[str, Any]],
    log_label: str = "PLANNED STEPS",
) -> Tuple[bool, List[Dict[str, Any]], List[Dict[str, Any]], str]:
    registry = build_registry()
    orch = Orchestrator(registry)
    ctx = ToolContext(user_id=user_id, settings=settings, api_key=api_key, goal=goal)
    sanitized_steps = sanitize_execution_steps(steps)

    print(f"\n===== {log_label} =====")
    for i, step in enumerate(sanitized_steps, 1):
        print(f"{i}. tool={step.get('tool')} args={step.get('args')}")
    print("=========================\n")

    tool_outputs_full = orch.run_steps(ctx, sanitized_steps)
    tool_outputs_compact = compact_tool_outputs(tool_outputs_full)
    ok = all(o.get("ok") for o in tool_outputs_full) if tool_outputs_full else True
    fallback_answer = extract_execution_answer(tool_outputs_full)
    return ok, tool_outputs_full, tool_outputs_compact, fallback_answer


def finalize_internal(*, provider: str, goal: str, tool_outputs_full: List[Dict[str, Any]]) -> str:
    llm = llm_for_provider(provider)
    llm_view = outputs_for_final_answer(tool_outputs_full)
    if hasattr(llm, "enabled") and llm.enabled():
        return str(llm.final_answer(goal=goal, tool_outputs=llm_view))
    return str(llm_view)
