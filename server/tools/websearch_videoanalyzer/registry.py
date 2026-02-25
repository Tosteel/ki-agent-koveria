from __future__ import annotations

import json
from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .models import VideoAnalyzeSyncRequest, VideoAnalyzeSyncResponse
from .websearch_videoanalyzer import VideoAnalyzerService


def _video_result_to_text(prompt: str, result: Dict[str, Any]) -> str:
    if not isinstance(result, dict):
        return str(result or "")

    for key in ("text", "answer", "summary", "content", "markdown"):
        val = result.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    for key in ("rows", "results", "items", "data"):
        val = result.get(key)
        if isinstance(val, list) and val:
            lines = [f"Video Analysis Prompt: {prompt}", ""]
            for i, item in enumerate(val, start=1):
                if isinstance(item, dict):
                    lines.append(f"[{i}] " + ", ".join(f"{k}={v}" for k, v in item.items()))
                else:
                    lines.append(f"[{i}] {item}")
            return "\n".join(lines).strip()

    return json.dumps(result, ensure_ascii=False, indent=2)


def register(registry: ToolRegistry) -> None:
    def tool_websearch_videoanalyzer(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = VideoAnalyzeSyncRequest(**args)
        service = VideoAnalyzerService(ctx.settings.video_analyzer_base_url, ctx.api_key)
        result = service.analyze_json_sync_from_prompt(prompt=req.prompt)
        result["text"] = _video_result_to_text(req.prompt, result)
        return result

    registry.register(
        "websearch_videoanalyzer",
        tool_websearch_videoanalyzer,
        request_model=VideoAnalyzeSyncRequest,
        response_model=VideoAnalyzeSyncResponse,
    )
    registry.register(
        "websearch_videoanalizer",
        tool_websearch_videoanalyzer,
        request_model=VideoAnalyzeSyncRequest,
        response_model=VideoAnalyzeSyncResponse,
    )
