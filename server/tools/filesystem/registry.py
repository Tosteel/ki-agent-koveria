from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .filesystem import read_text, write_text
from .models import FileReadRequest, FileReadResponse, FileWriteRequest, FileWriteResponse


def register(registry: ToolRegistry) -> None:
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

    registry.register("read_file", tool_read_file, request_model=FileReadRequest)
    registry.register("write_file", tool_write_file, request_model=FileWriteRequest)
