from __future__ import annotations

from typing import Dict, List


TOOL_PRIORITIES: List[Dict[str, object]] = [
    {
        "task_de": "Websuche/Internet-Recherche",
        "task_en": "internet/web research",
        "primary": "langsearch",
        "fallbacks": ["websearch_table", "search_multitable", "web_search_page", "web_crawl_site"],
    },
    {
        "task_de": "Videoanalyse aus Prompt",
        "task_en": "video analysis from prompt",
        "primary": "websearch_videoanalyzer",
        "fallbacks": [],
    },
    {
        "task_de": "Produktsuche",
        "task_en": "product search",
        "primary": "ebay_search",
        "fallbacks": [],
    },
    {
        "task_de": "Mail senden (neue E-Mail)",
        "task_en": "send email (new message)",
        "primary": "mail_send",
        "fallbacks": [],
    },
    {
        "task_de": "Mail beantworten (bestehende Inbox-Mail)",
        "task_en": "reply to existing inbox email",
        "primary": "mail_answer",
        "fallbacks": [],
    },
    {
        "task_de": "Inbox abrufen",
        "task_en": "fetch inbox",
        "primary": "mail_fetch_inbox",
        "fallbacks": [],
    },
    {
        "task_de": "Unbeantwortete Mails abrufen",
        "task_en": "fetch unanswered mails",
        "primary": "mail_fetch_unanswered",
        "fallbacks": [],
    },
]


def planner_priority_guidance(provider: str) -> str:
    p = str(provider or "").strip().lower()
    is_de = p in {"ionos", ""}

    lines: List[str] = []
    if is_de:
        lines.append("Verwende die folgende Tool-Priorisierung (Primary zuerst, dann Fallbacks):")
        for rule in TOOL_PRIORITIES:
            task = str(rule.get("task_de") or "").strip()
            primary = str(rule.get("primary") or "").strip()
            fallbacks = [str(x).strip() for x in (rule.get("fallbacks") or []) if str(x).strip()]
            if not task or not primary:
                continue
            if fallbacks:
                lines.append(f"- {task}: {primary} -> {', '.join(fallbacks)}")
            else:
                lines.append(f"- {task}: {primary}")
    else:
        lines.append("Use the following tool priority order (primary first, then fallbacks):")
        for rule in TOOL_PRIORITIES:
            task = str(rule.get("task_en") or "").strip()
            primary = str(rule.get("primary") or "").strip()
            fallbacks = [str(x).strip() for x in (rule.get("fallbacks") or []) if str(x).strip()]
            if not task or not primary:
                continue
            if fallbacks:
                lines.append(f"- {task}: {primary} -> {', '.join(fallbacks)}")
            else:
                lines.append(f"- {task}: {primary}")

    return "\n".join(lines)
