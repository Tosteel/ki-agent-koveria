from __future__ import annotations

import json
from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .models import SearchGenerateJsonRequest
from .search_multitable import SearchService


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


def register(registry: ToolRegistry) -> None:
    def tool_search_multitable(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = SearchGenerateJsonRequest(**args)
        service = SearchService(ctx.settings.search_base_url, ctx.api_key)
        result = service.search_generate_json(user_prompt=req.user_prompt)
        result["text"] = _search_result_to_text(req.user_prompt, result)
        return result

    registry.register("search_multitable", tool_search_multitable, request_model=SearchGenerateJsonRequest)
