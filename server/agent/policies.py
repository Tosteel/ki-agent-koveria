from __future__ import annotations

import importlib
import re
from functools import lru_cache
from typing import Iterable, Set


GLOBAL_BASIC_TOOLS = {
    "file_read",
    "file_write",
    "rag_knowledgebase",
    "llm_text_chat",
    "llm_text_summarize",
    "llm_text_compose",
    "pdf_export",
    "ppt_export",
    "websearch_table",
    "websearch_videoanalyzer",
    "langsearch",
    "ebay_search",
    "web_fetch_page",
    "web_search_page",
    "web_crawl_site",
    "web_crawl_site_whitelist",
    "distance_check",
    "pricing_compute_quote",
    "booking_extract_facts",
    "booking_validate_completeness",
    "booking_booking_validate_completness",
    "booking_decision_engine",
    "booking_descision_enginge",
    "booking_reply_score",
    "booking_instruction_check",
    "mail_send",
    "mail_answer",
    "mail_fetch_inbox",
    "mail_fetch_unanswered",
    "mail_compose_clarification",
    "gmail_send_mail",
    "gmail_answer_mail",
    "gmail_fetch_inbox_mails",
    "gmail_fetch_unanswered_mails",
    "gmail_read_mail",
    "gmail_read_mail_thread",
    "calendar_check_availability",
    "calendar_create_event",
    "calendar_update_event",
    "calendar_propose_slots",
    "calendar_hold_event",
    "customer_support_reply_score",
    "customer_support_review_ticket_create",
    "customer_support_review_ticket_update",
    "customer_support_policy_check",
    "assistent_profile_create",
    "assistent_profile_get",
    "assistent_profile_update",
    "assistent_profile_check",
    "skills_list",
    "startup_matchup_step_1_workshop_analysis",
    "startup_matchup_step_2_company_profile",
    "startup_matchup_step_3_gap_analysis",
    "startup_matchup_step_4_startup_search",
    "startup_matchup_step_4_1_startup_structuring",
    "startup_matchup_step_5_startup_ranking",
    "startup_matchup_step_6_startup_deep_research",
    "startup_matchup_step_7_startup_profiles",
    "startup_matchup_step_8_final_report",
    "startup_matchup_step_9_pdf_report",
}

GLOBAL_COMPETITIVE_ANALYSIS_TOOLS = {
    # Reserved for future competitive analysis tools.
}

# Backward compatibility: legacy imports still use these names.
BASIC_TOOLS = GLOBAL_BASIC_TOOLS
COMPETITIVE_ANALYSIS_TOOLS = GLOBAL_COMPETITIVE_ANALYSIS_TOOLS
OFFER_FLOW: Set[str] = set()

ASSISTANT_POLICY_MODULES = {
    "booking-assistant": "server.assistants.booking_assistant.policies",
    "booking-assistant-v2": "server.assistants.booking_assistant_v2.policies",
    "booking-assistant-v3": "server.assistants.booking_assistant_v3.policies",
    "mail-assistant": "server.assistants.mail_assistant.policies",
}


def _normalize_tools(values: Iterable[object]) -> Set[str]:
    out: Set[str] = set()
    for raw in values:
        item = str(raw or "").strip()
        if item:
            out.add(item)
    return out


def _extract_assistant_id(goal: str) -> str:
    text = str(goal or "")
    if not text:
        return ""
    pattern = re.compile(r"^\s*assistant_id\s*[:=]\s*([a-z0-9_-]+)\s*$", re.IGNORECASE | re.MULTILINE)
    match = pattern.search(text)
    if match:
        return str(match.group(1) or "").strip().lower()
    return ""


@lru_cache(maxsize=16)
def _load_assistant_tools(assistant_id: str) -> Set[str]:
    assistant_key = str(assistant_id or "").strip().lower()
    if not assistant_key:
        return set()

    module_name = ASSISTANT_POLICY_MODULES.get(assistant_key)
    if not module_name:
        # Fallback mapping: server.assistants/<assistant_name>/policies.py
        module_name = f"server.assistants.{assistant_key.replace('-', '_')}.policies"
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return set()

    raw = getattr(module, "ALLOWED_TOOLS", None)
    if isinstance(raw, (set, list, tuple)):
        return _normalize_tools(raw)
    return set()


def tools_allowed(tool_name: str, goal: str = "") -> bool:
    name = str(tool_name or "").strip()
    if not name:
        return False
    if name.startswith("agent_"):
        return True

    assistant_id = _extract_assistant_id(goal)
    if assistant_id:
        scoped_tools = _load_assistant_tools(assistant_id)
        if scoped_tools:
            return name in scoped_tools

    return bool(name in GLOBAL_BASIC_TOOLS or name in GLOBAL_COMPETITIVE_ANALYSIS_TOOLS)
