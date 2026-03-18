from __future__ import annotations

import json
from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .filesystem import read_text, write_text
from .models import FileReadRequest, FileReadResponse, FileWriteRequest, FileWriteResponse


def register(registry: ToolRegistry) -> None:
    def tool_read_file(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = FileReadRequest(**args)
        content = read_text(
            ctx.settings.user_work_dir(ctx.user_id),
            req.path,
            encoding=req.encoding,
            uploads_dir=ctx.settings.user_dir(ctx.user_id) / "uploads",
        )
        return FileReadResponse(path=req.path, content=content).model_dump()

    def tool_write_file(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = FileWriteRequest(**args)
        content = req.content
        if isinstance(content, (dict, list)):
            content_text = json.dumps(content, ensure_ascii=False, indent=2)
        elif isinstance(content, str):
            content_text = content
        else:
            content_text = str(content)
        n = write_text(
            ctx.settings.user_work_dir(ctx.user_id),
            req.path,
            content_text,
            encoding=req.encoding,
            overwrite=req.overwrite,
        )
        return FileWriteResponse(path=req.path, bytes_written=n).model_dump()

    registry.register(
        "file_read",
        tool_read_file,
        request_model=FileReadRequest,
        response_model=FileReadResponse,
    )
    registry.register(
        "file_write",
        tool_write_file,
        request_model=FileWriteRequest,
        response_model=FileWriteResponse,
    )
