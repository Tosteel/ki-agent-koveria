from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List


DEFAULT_TOOL_POLICY: Dict[str, Any] = {
    "capabilities": [],
    "side_effect_level": "none",
    "requires": {},
    "result_contract": {
        "success_if_any": [
            "text",
            "summary",
            "composed_text",
            "hits",
            "rows",
            "results",
            "items",
            "data",
            "sent",
            "bytes_written",
        ],
        "empty_if": ["hits_empty", "rows_empty", "text_blank", "no_sources"],
    },
    "retry_policy": {
        "max_retries": 0,
        "backoff_ms": 0,
        "retry_on": [],
    },
    "fallback": {
        "on_empty_capabilities": [],
        "on_transient_error_capabilities": [],
        "fallback_candidates": [],
    },
    "quality_signals": {
        "min_hits": 0,
        "require_sources": False,
        "min_text_length": 0,
    },
    "allows_goal_injection": True,
}


TOOL_POLICY_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "rag_knowledgebase": {
        "capabilities": ["knowledge_search"],
        "retry_policy": {
            "max_retries": 2,
            "backoff_ms": 350,
            "retry_on": ["timeout", "connection_refused", "connection_error", "rate_limit"],
        },
        "fallback": {
            "on_empty_capabilities": ["web_search"],
            "on_transient_error_capabilities": ["web_search"],
            "fallback_candidates": ["langsearch", "search_multitable", "websearch_table"],
        },
        "quality_signals": {
            "min_hits": 1,
            "require_sources": True,
            "min_text_length": 120,
        },
    },
    "websearch_table": {
        "capabilities": ["web_search"],
        "retry_policy": {"max_retries": 1, "backoff_ms": 250, "retry_on": ["timeout", "connection_error", "rate_limit"]},
        "quality_signals": {"min_hits": 1, "require_sources": True, "min_text_length": 120},
    },
    "search_multitable": {
        "capabilities": ["web_search"],
        "retry_policy": {"max_retries": 1, "backoff_ms": 250, "retry_on": ["timeout", "connection_error", "rate_limit"]},
        "quality_signals": {"min_hits": 1, "require_sources": True, "min_text_length": 120},
    },
    "langsearch": {
        "capabilities": ["web_search"],
        "retry_policy": {"max_retries": 1, "backoff_ms": 250, "retry_on": ["timeout", "connection_error", "rate_limit"]},
        "quality_signals": {"min_hits": 1, "require_sources": True, "min_text_length": 120},
    },
    "web_search_page": {
        "capabilities": ["web_search"],
        "retry_policy": {"max_retries": 1, "backoff_ms": 250, "retry_on": ["timeout", "connection_error", "rate_limit"]},
        "quality_signals": {"min_hits": 1, "require_sources": True, "min_text_length": 80},
    },
    "web_crawl_site": {
        "capabilities": ["web_search"],
        "retry_policy": {"max_retries": 1, "backoff_ms": 250, "retry_on": ["timeout", "connection_error", "rate_limit"]},
        "quality_signals": {"min_hits": 1, "require_sources": True, "min_text_length": 120},
    },
    "web_crawl_site_whitelist": {
        "capabilities": ["web_search", "web_search_whitelist"],
        "retry_policy": {"max_retries": 1, "backoff_ms": 250, "retry_on": ["timeout", "connection_error", "rate_limit"]},
        "quality_signals": {"min_hits": 1, "require_sources": True, "min_text_length": 120},
    },
    "llm_text_compose": {
        "capabilities": ["text_compose"],
        "quality_signals": {"min_hits": 0, "require_sources": False, "min_text_length": 40},
    },
    "llm_text_summarize": {
        "capabilities": ["text_compose"],
        "quality_signals": {"min_hits": 0, "require_sources": False, "min_text_length": 40},
    },
    "send_mail": {
        "capabilities": ["communication:email_send"],
        "side_effect_level": "high",
        "quality_signals": {"min_hits": 0, "require_sources": False, "min_text_length": 40},
    },
    "answer_mail": {
        "capabilities": ["communication:email_send"],
        "side_effect_level": "high",
        "quality_signals": {"min_hits": 0, "require_sources": False, "min_text_length": 20},
    },
    "write_file": {
        "capabilities": ["artifact_write"],
        "side_effect_level": "high",
    },
}


CAPABILITY_FALLBACK_PRIORITY: Dict[str, List[str]] = {
    "web_search": ["websearch_table", "search_multitable", "langsearch", "web_search_page", "web_crawl_site"],
    "knowledge_search": ["rag_knowledgebase"],
    "text_compose": ["llm_text_compose", "llm_text_summarize"],
}


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dict(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _as_str_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def normalize_tool_policy(raw: Dict[str, Any] | None) -> Dict[str, Any]:
    src = raw if isinstance(raw, dict) else {}
    policy = _deep_merge_dict(DEFAULT_TOOL_POLICY, src)
    policy["capabilities"] = _as_str_list(policy.get("capabilities"))
    policy["side_effect_level"] = str(policy.get("side_effect_level") or "none").strip().lower()
    if policy["side_effect_level"] not in {"none", "low", "high"}:
        policy["side_effect_level"] = "none"
    if not isinstance(policy.get("requires"), dict):
        policy["requires"] = {}
    if not isinstance(policy.get("result_contract"), dict):
        policy["result_contract"] = deepcopy(DEFAULT_TOOL_POLICY["result_contract"])
    if not isinstance(policy.get("retry_policy"), dict):
        policy["retry_policy"] = deepcopy(DEFAULT_TOOL_POLICY["retry_policy"])
    if not isinstance(policy.get("fallback"), dict):
        policy["fallback"] = deepcopy(DEFAULT_TOOL_POLICY["fallback"])
    if not isinstance(policy.get("quality_signals"), dict):
        policy["quality_signals"] = deepcopy(DEFAULT_TOOL_POLICY["quality_signals"])
    policy["retry_policy"]["max_retries"] = max(0, int(policy["retry_policy"].get("max_retries") or 0))
    policy["retry_policy"]["backoff_ms"] = max(0, int(policy["retry_policy"].get("backoff_ms") or 0))
    policy["retry_policy"]["retry_on"] = _as_str_list(policy["retry_policy"].get("retry_on"))
    policy["fallback"]["on_empty_capabilities"] = _as_str_list(policy["fallback"].get("on_empty_capabilities"))
    policy["fallback"]["on_transient_error_capabilities"] = _as_str_list(
        policy["fallback"].get("on_transient_error_capabilities")
    )
    policy["fallback"]["fallback_candidates"] = _as_str_list(policy["fallback"].get("fallback_candidates"))
    policy["quality_signals"]["min_hits"] = max(0, int(policy["quality_signals"].get("min_hits") or 0))
    policy["quality_signals"]["min_text_length"] = max(0, int(policy["quality_signals"].get("min_text_length") or 0))
    policy["quality_signals"]["require_sources"] = bool(policy["quality_signals"].get("require_sources"))
    policy["allows_goal_injection"] = bool(policy.get("allows_goal_injection", True))
    return policy


def build_tool_policy(tool_name: str, metadata: Dict[str, Any] | None) -> Dict[str, Any]:
    meta = metadata if isinstance(metadata, dict) else {}
    meta_policy_fields = {
        key: deepcopy(meta[key])
        for key in (
            "capabilities",
            "side_effect_level",
            "requires",
            "result_contract",
            "retry_policy",
            "fallback",
            "quality_signals",
            "allows_goal_injection",
        )
        if key in meta
    }
    merged = _deep_merge_dict(DEFAULT_TOOL_POLICY, TOOL_POLICY_OVERRIDES.get(str(tool_name).strip(), {}))
    merged = _deep_merge_dict(merged, meta_policy_fields)
    return normalize_tool_policy(merged)


def classify_error_kind(error_text: str) -> str:
    txt = str(error_text or "").lower()
    if not txt:
        return "unknown_error"
    transient_markers = (
        "timeout",
        "timed out",
        "connection refused",
        "failed to establish a new connection",
        "max retries exceeded",
        "temporarily unavailable",
        "rate limit",
        "429",
    )
    if any(marker in txt for marker in transient_markers):
        return "transient_error"
    permanent_markers = ("validationerror", "unknown tool", "tool_not_allowed", "invalid_step_schema")
    if any(marker in txt for marker in permanent_markers):
        return "permanent_error"
    return "unknown_error"


def fallback_candidates_for_capabilities(capabilities: Iterable[str]) -> List[str]:
    out: List[str] = []
    for cap in capabilities:
        for tool_name in CAPABILITY_FALLBACK_PRIORITY.get(str(cap).strip(), []):
            if tool_name not in out:
                out.append(tool_name)
    return out
