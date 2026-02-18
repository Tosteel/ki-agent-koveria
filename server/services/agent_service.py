from __future__ import annotations

import json
import ast
import re
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field, create_model

from server.core.settings import Settings
from server.tools.loader import register_all_tools
from server.agent.tool_registry import ToolRegistry, ToolContext
from server.agent.orchestrator import Orchestrator
from server.services.llm_openai import LlmOpenai
from server.services.llm_ionos import IonosLLM
from server.services import memory_service
from server.services.agent_prompts import (
    get_goal_context_system_prompt,
    get_planner_guard_refine_system_prompt,
    get_planner_guard_system_prompt,
)


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
        elif tool == "llm_smalltalk":
            raw_chars = args_out.get("max_chars")
            try:
                max_chars = int(raw_chars)
            except Exception:
                max_chars = 280
            if max_chars < 60:
                max_chars = 60
            if max_chars > 1200:
                max_chars = 1200
            args_out["max_chars"] = max_chars
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


def _parse_json_strictish(text: str) -> Dict[str, Any]:
    t = str(text or "").strip()
    if not t:
        return {}
    if t.startswith("```"):
        t = t.strip("`").strip()
        if "\n" in t:
            first, rest = t.split("\n", 1)
            if first.strip().lower() in {"json", "javascript"}:
                t = rest.strip()
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    start = t.find("{")
    end = t.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(t[start : end + 1])
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def _extract_openai_output_text(resp: Dict[str, Any]) -> str:
    out = ""
    for item in resp.get("output", []):
        if not isinstance(item, dict):
            continue
        for c in item.get("content", []):
            if isinstance(c, dict) and c.get("type") == "output_text":
                out += str(c.get("text") or "")
    return out.strip()


def _llm_planner_guard(llm: Any, provider: str, goal: str, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not hasattr(llm, "enabled") or not llm.enabled():
        return {}

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "status": {"type": "string", "enum": ["ready", "replan"]},
            "missing": {"type": "array", "items": {"type": "string"}},
            "reasons": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["status", "missing", "reasons"],
    }
    compact_steps: List[Dict[str, Any]] = []
    for i, step in enumerate(steps, 1):
        if not isinstance(step, dict):
            continue
        args = step.get("args") if isinstance(step.get("args"), dict) else {}
        compact_steps.append(
            {
                "step": i,
                "tool": str(step.get("tool") or "").strip(),
                "args": args,
            }
        )

    system = get_planner_guard_system_prompt(provider)
    user = (
        f"Ziel:\n{goal}\n\n"
        f"Geplante Schritte:\n{json.dumps(compact_steps, ensure_ascii=False)}\n\n"
        "Prüfkriterien:\n"
        "- Wenn Ziel E-Mail-Versand verlangt, muss send_mail oder answer_mail enthalten sein.\n"
        "- Wenn Ziel PDF verlangt, muss pdf_export enthalten sein.\n"
        "- Wenn Ziel Präsentation/PPT verlangt, muss ppt_export enthalten sein.\n"
        "- Referenzen wie {steps[i].text} müssen zu existierenden Steps und sinnvollen Output-Feldern passen.\n"
        "- Pflichtargumente pro Tool müssen erkennbar gesetzt sein.\n"
        "- Wenn kein Schritt geplant ist: replan.\n"
        "\nAusgabevorgaben für JSON:\n"
        "- status=ready nur wenn keine harten Lücken bestehen.\n"
        "- status=replan nur mit KONKRETEN missing/reasons.\n"
        "- missing enthält maschinenlesbare Tokens, z.B.:\n"
        "  missing_tool:send_mail\n"
        "  missing_tool:pdf_export\n"
        "  bad_reference:steps[1].summary\n"
        "  missing_arg:send_mail.to\n"
        "  missing_arg:answer_mail.mail_id\n"
        "- reasons beschreibt pro missing präzise den Befund inkl. Step-Nummer/Tool."
    )

    try:
        if hasattr(llm, "chat_completions"):
            completion = llm.chat_completions(
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "planner_guard", "schema": schema, "strict": True},
                },
            )
            text = llm.extract_text(completion) if hasattr(llm, "extract_text") else ""
            parsed = _parse_json_strictish(text)
        elif hasattr(llm, "_call"):
            resp = llm._call(  # type: ignore[attr-defined]
                input_messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                text_format={"type": "json_schema", "name": "planner_guard", "schema": schema, "strict": False},
            )
            parsed = _parse_json_strictish(_extract_openai_output_text(resp))
        else:
            return {}
    except Exception:
        return {}

    status = str(parsed.get("status") or "").strip().lower()
    if status not in {"ready", "replan"}:
        return {}
    missing = [str(x).strip() for x in (parsed.get("missing") or []) if str(x).strip()]
    reasons = [str(x).strip() for x in (parsed.get("reasons") or []) if str(x).strip()]
    return {"status": status, "missing": missing, "reasons": reasons}


def run_planner_guard(llm: Any, provider: str, goal: str, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    llm_gate = _llm_planner_guard(llm, provider, goal, steps)
    if not llm_gate:
        return {
            "status": "replan",
            "missing": ["planner_guard"],
            "reasons": ["Planner-Guard konnte nicht vom LLM bewertet werden."],
            "instructions": "WICHTIGE PLANUNGSREGELN (müssen erfüllt sein):\n- Vollständigen, zielkonformen Plan erzeugen.",
        }

    status = str(llm_gate.get("status") or "replan").strip().lower()
    missing = [str(x).strip() for x in (llm_gate.get("missing") or []) if str(x).strip()]
    reasons = [str(x).strip() for x in (llm_gate.get("reasons") or []) if str(x).strip()]

    # If the guard answer is too generic, ask once for concrete, machine-readable gaps.
    generic_tokens = {"plan_mismatch", "unknown", "insufficient_info", "not_enough_info"}
    def _looks_generic(items: List[str], rs: List[str]) -> bool:
        if not items:
            return True
        if any(i in generic_tokens for i in items):
            return True
        if any("nicht genug information" in r.lower() for r in rs):
            return True
        has_structured = any(
            i.startswith("missing_tool:")
            or i.startswith("missing_arg:")
            or i.startswith("bad_reference:")
            for i in items
        )
        return not has_structured

    if status == "replan" and _looks_generic(missing, reasons) and hasattr(llm, "enabled") and llm.enabled():
        schema_refine = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "missing": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "reasons": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            },
            "required": ["missing", "reasons"],
        }
        compact_steps = []
        for i, step in enumerate(steps, 1):
            if not isinstance(step, dict):
                continue
            args = step.get("args") if isinstance(step.get("args"), dict) else {}
            compact_steps.append({"step": i, "tool": str(step.get("tool") or "").strip(), "args": args})
        system_refine = get_planner_guard_refine_system_prompt(provider)
        user_refine = (
            f"Ziel:\n{goal}\n\n"
            f"Schritte:\n{json.dumps(compact_steps, ensure_ascii=False)}\n\n"
            "Liefere KONKRETE missing/reasons.\n"
            "missing MUSS nur diese Formen nutzen:\n"
            "- missing_tool:<tool>\n"
            "- missing_arg:<tool>.<arg>\n"
            "- bad_reference:<reference>\n"
            "- missing_step:<purpose>\n"
            "Keine generischen Begriffe wie plan_mismatch."
        )
        try:
            if hasattr(llm, "chat_completions"):
                c = llm.chat_completions(
                    messages=[{"role": "system", "content": system_refine}, {"role": "user", "content": user_refine}],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "planner_guard_refine", "schema": schema_refine, "strict": True},
                    },
                )
                t = llm.extract_text(c) if hasattr(llm, "extract_text") else ""
                parsed = _parse_json_strictish(t)
            elif hasattr(llm, "_call"):
                r = llm._call(  # type: ignore[attr-defined]
                    input_messages=[{"role": "system", "content": system_refine}, {"role": "user", "content": user_refine}],
                    text_format={"type": "json_schema", "name": "planner_guard_refine", "schema": schema_refine, "strict": False},
                )
                parsed = _parse_json_strictish(_extract_openai_output_text(r))
            else:
                parsed = {}
            refined_missing = [str(x).strip() for x in (parsed.get("missing") or []) if str(x).strip()]
            refined_reasons = [str(x).strip() for x in (parsed.get("reasons") or []) if str(x).strip()]
            if refined_missing:
                missing = refined_missing
            if refined_reasons:
                reasons = refined_reasons
        except Exception:
            pass
    if status == "replan":
        if not reasons:
            reasons = ["Der Plan passt laut Guard nicht vollständig zum Ziel."]
        if not missing:
            missing = ["missing_step:goal_alignment"]
    instructions = ""
    if status == "replan":
        instructions = "WICHTIGE PLANUNGSREGELN (müssen erfüllt sein):\n- " + "\n- ".join(reasons)
    return {"status": status, "missing": missing, "reasons": reasons, "instructions": instructions}


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


def build_goal_with_context(llm: Any, provider: str, goal: str, history: Any) -> str:
    base = (goal or "").strip()
    if not base:
        return base
    hist = history_to_context(history, max_items=12)
    if not hist:
        return base
    if not hasattr(llm, "enabled") or not llm.enabled():
        return f"Aktuelle Anfrage:\n{base}\n\nKontext:\n{hist}"

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "standalone_goal": {"type": "string"},
            "carry_context": {"type": "array", "items": {"type": "string"}},
            "needs_context": {"type": "boolean"},
        },
        "required": ["standalone_goal", "carry_context", "needs_context"],
    }
    system = get_goal_context_system_prompt(provider)
    user = (
        f"Aktuelle Nutzeranfrage:\n{base}\n\n"
        f"Dialogverlauf (letzte 12 Nachrichten):\n{hist}\n\n"
        "Erzeuge ein standalone_goal. "
        "Falls für die Ausführung nötig, liefere 1-3 kurze Kontextpunkte in carry_context."
    )
    try:
        if hasattr(llm, "chat_completions"):
            completion = llm.chat_completions(
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "goal_context", "schema": schema, "strict": True},
                },
            )
            parsed = _parse_json_strictish(llm.extract_text(completion) if hasattr(llm, "extract_text") else "")
        elif hasattr(llm, "_call"):
            resp = llm._call(  # type: ignore[attr-defined]
                input_messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                text_format={"type": "json_schema", "name": "goal_context", "schema": schema, "strict": False},
            )
            parsed = _parse_json_strictish(_extract_openai_output_text(resp))
        else:
            parsed = {}
    except Exception:
        parsed = {}

    standalone_goal = str(parsed.get("standalone_goal") or base).strip() or base
    needs_context = bool(parsed.get("needs_context"))
    carry_context_raw = parsed.get("carry_context") or []
    carry_context = [str(x).strip() for x in carry_context_raw if str(x).strip()][:3]
    if not needs_context or not carry_context:
        return standalone_goal
    return f"Aktuelle Anfrage:\n{standalone_goal}\n\nKontext:\n" + "\n".join(f"- {c}" for c in carry_context)


def _slugify_tool_name(title: str, agent_id: Any) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", str(title or "").strip().lower()).strip("_")
    if not base:
        base = f"agent_{agent_id}"
    if not base.startswith("agent_"):
        base = f"agent_{base}"
    return base


def _parse_planned_steps(steps: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for raw in steps:
        line = str(raw or "").strip()
        if not line:
            continue
        marker = "tool="
        args_marker = " args="
        i = line.find(marker)
        j = line.find(args_marker)
        if i < 0 or j < 0 or j <= i:
            continue
        tool = line[i + len(marker) : j].strip()
        args_raw = line[j + len(args_marker) :].strip()
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
        out.append({"tool": tool, "args": args})
    return out


def _replace_agent_placeholders(value: Any, values: Dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {k: _replace_agent_placeholders(v, values) for k, v in value.items()}
    if isinstance(value, list):
        return [_replace_agent_placeholders(v, values) for v in value]
    if not isinstance(value, str):
        return value

    full = re.fullmatch(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}", value.strip())
    if full:
        key = full.group(1)
        return values.get(key, value)

    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        rep = values.get(key, match.group(0))
        return str(rep)

    return re.sub(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}", repl, value)


def _placeholder_type_to_py(placeholder_type: str) -> Any:
    p_type = str(placeholder_type or "").strip().lower()
    if p_type in {"int", "integer", "number"}:
        return int
    if p_type in {"float", "double"}:
        return float
    if p_type in {"bool", "boolean"}:
        return bool
    return str


def _register_user_agent_tools(registry: ToolRegistry, settings: Settings, user_id: str) -> ToolRegistry:
    memory = memory_service.load_agents_memory_for_user(settings, user_id)
    agents = memory.get("agents") if isinstance(memory.get("agents"), list) else []
    used_names: set[str] = set()

    for agent in agents:
        if not isinstance(agent, dict):
            continue
        agent_id = agent.get("id")
        title = str(agent.get("title") or f"agent_{agent_id}").strip()
        tool_name = _slugify_tool_name(title, agent_id)
        if tool_name in used_names:
            tool_name = f"{tool_name}_{agent_id}"
        used_names.add(tool_name)

        placeholders = agent.get("placeholders") if isinstance(agent.get("placeholders"), list) else []
        fields: Dict[str, Any] = {}
        for p in placeholders:
            if not isinstance(p, dict):
                continue
            p_name = str(p.get("name") or "").strip()
            if not p_name:
                continue
            py_type = _placeholder_type_to_py(str(p.get("type") or "string"))
            p_required = bool(p.get("required", True))
            p_desc = str(p.get("description") or f"Platzhalter {p_name}").strip()
            if p_required:
                fields[p_name] = (py_type, Field(..., description=p_desc))
            else:
                fields[p_name] = (py_type | None, Field(default=None, description=p_desc))

        request_model = create_model(f"AgentToolRequest_{tool_name}", **fields) if fields else create_model(
            f"AgentToolRequest_{tool_name}"
        )

        planned_steps_lines = [
            str(s).strip() for s in (agent.get("planned_steps") or []) if str(s).strip()
        ]
        planned_steps = _parse_planned_steps(planned_steps_lines)

        def _make_handler(
            *,
            _tool_name: str,
            _agent_id: Any,
            _agent_title: str,
            _planned_steps: List[Dict[str, Any]],
        ):
            def _handler(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
                if not _planned_steps:
                    raise RuntimeError(f"{_tool_name}: planned_steps_empty")

                replaced_steps: List[Dict[str, Any]] = []
                for step in _planned_steps:
                    replaced_steps.append(
                        {
                            "tool": str(step.get("tool") or "").strip(),
                            "args": _replace_agent_placeholders(dict(step.get("args") or {}), args),
                        }
                    )
                replaced_steps = sanitize_execution_steps(replaced_steps)

                inner_registry = register_all_tools(ToolRegistry())
                inner_orch = Orchestrator(inner_registry)
                inner_ctx = ToolContext(
                    user_id=ctx.user_id,
                    settings=ctx.settings,
                    api_key=ctx.api_key,
                    goal=ctx.goal,
                )
                outputs = inner_orch.run_steps(inner_ctx, replaced_steps)
                if any(not o.get("ok") for o in outputs):
                    last_error = next(
                        (str(o.get("error") or "agent_substep_failed") for o in reversed(outputs) if not o.get("ok")),
                        "agent_substep_failed",
                    )
                    raise RuntimeError(last_error)
                text = extract_execution_answer(outputs)
                last_payload = outputs[-1].get("payload") if outputs else {}
                output_path = ""
                if isinstance(last_payload, dict):
                    output_path = str(last_payload.get("output_path") or "").strip()
                return {
                    "agent_id": _agent_id,
                    "agent_title": _agent_title,
                    "tool_name": _tool_name,
                    "placeholder_values": args,
                    "executed_steps": replaced_steps,
                    "text": text,
                    "output_path": output_path,
                }

            return _handler

        registry.register(
            tool_name,
            _make_handler(
                _tool_name=tool_name,
                _agent_id=agent_id,
                _agent_title=title,
                _planned_steps=planned_steps,
            ),
            request_model=request_model,
        )
    return registry


def append_agent_tools_hint(goal: str, settings: Settings, user_id: str) -> str:
    base = str(goal or "").strip()
    memory = memory_service.load_agents_memory_for_user(settings, user_id)
    agents = memory.get("agents") if isinstance(memory.get("agents"), list) else []
    hints: List[str] = []
    for agent in agents:
        if not isinstance(agent, dict):
            continue
        title = str(agent.get("title") or "").strip()
        if not title:
            continue
        tool_name = _slugify_tool_name(title, agent.get("id"))
        phs = agent.get("placeholders") if isinstance(agent.get("placeholders"), list) else []
        ph_names = [str(p.get("name") or "").strip() for p in phs if isinstance(p, dict) and str(p.get("name") or "").strip()]
        mention = title.lower() in base.lower() or tool_name.lower() in base.lower()
        if mention:
            if ph_names:
                hints.append(f"- {title} -> tool={tool_name} placeholders={ph_names}")
            else:
                hints.append(f"- {title} -> tool={tool_name}")
    if not hints:
        return base
    return (
        f"{base}\n\n"
        "Hinweis: Wenn ein genannter Agent passt, nutze genau dessen Tool.\n"
        "Verfügbare genannte Agent-Tools:\n"
        + "\n".join(hints)
    )


def build_registry(*, settings: Settings | None = None, user_id: str | None = None) -> ToolRegistry:
    registry = ToolRegistry()
    register_all_tools(registry)
    if settings is not None and user_id:
        _register_user_agent_tools(registry, settings, user_id)
    return registry


def provider_key(provider: str) -> str:
    p = str(provider or "").strip().lower()
    if p in {"openai", "ionos"}:
        return p
    return "ionos"


def llm_for_provider(provider: str) -> Any:
    p = provider_key(provider)
    if p == "openai":
        return LlmOpenai()
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
    registry = build_registry(settings=settings, user_id=user_id)
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


def _execution_facts(tool_outputs_full: List[Dict[str, Any]]) -> Dict[str, Any]:
    facts: Dict[str, Any] = {
        "pdf_created": False,
        "pdf_output_paths": [],
        "mail_sent": False,
        "mail_recipients": [],
    }
    pdf_paths: List[str] = []
    recipients: List[str] = []

    for out in tool_outputs_full:
        if not isinstance(out, dict) or not out.get("ok"):
            continue
        tool = str(out.get("tool") or "").strip()
        payload = out.get("payload") if isinstance(out.get("payload"), dict) else {}

        if tool == "pdf_export":
            path = str(payload.get("output_path") or "").strip()
            if path:
                pdf_paths.append(path)
            facts["pdf_created"] = True

        if tool in {"send_mail", "answer_mail"}:
            sent_flag = payload.get("sent")
            if sent_flag is True or tool == "answer_mail":
                facts["mail_sent"] = True
                rcpts = payload.get("recipients")
                if isinstance(rcpts, list):
                    for r in rcpts:
                        rs = str(r).strip()
                        if rs:
                            recipients.append(rs)

    facts["pdf_output_paths"] = pdf_paths
    facts["mail_recipients"] = recipients
    return facts


def _enforce_fact_consistency(answer: str, facts: Dict[str, Any]) -> str:
    text = str(answer or "").strip()
    if not text:
        return text
    lower = text.lower()

    pdf_created = bool(facts.get("pdf_created"))
    mail_sent = bool(facts.get("mail_sent"))
    pdf_paths = [str(p).strip() for p in (facts.get("pdf_output_paths") or []) if str(p).strip()]
    recipients = [str(r).strip() for r in (facts.get("mail_recipients") or []) if str(r).strip()]

    pdf_negative_tokens = (
        "keine pdf",
        "nicht als pdf",
        "pdf konnte nicht",
        "pdf wurde nicht",
        "keine zusammenfassung als pdf",
    )
    if pdf_created and any(tok in lower for tok in pdf_negative_tokens):
        suffix = f" ({pdf_paths[-1]})" if pdf_paths else ""
        return f"Das PDF wurde erfolgreich erstellt{suffix}."

    mail_negative_tokens = (
        "mail konnte nicht",
        "e-mail konnte nicht",
        "nicht gesendet",
        "wurde nicht gesendet",
        "konnte nicht an",
    )
    if mail_sent and any(tok in lower for tok in mail_negative_tokens):
        if recipients:
            return f"Die E-Mail wurde erfolgreich gesendet an: {', '.join(recipients)}."
        return "Die E-Mail wurde erfolgreich gesendet."

    return text


def finalize_internal(*, provider: str, goal: str, tool_outputs_full: List[Dict[str, Any]]) -> str:
    llm = llm_for_provider(provider)
    llm_view = outputs_for_final_answer(tool_outputs_full)
    facts = _execution_facts(tool_outputs_full)
    goal_for_final = (
        f"{goal}\n\n"
        "Execution facts (ground truth, must be respected):\n"
        + json.dumps(facts, ensure_ascii=False)
    )
    if hasattr(llm, "enabled") and llm.enabled():
        raw = str(llm.final_answer(goal=goal_for_final, tool_outputs=llm_view))
        return _enforce_fact_consistency(raw, facts)
    return str(llm_view)
