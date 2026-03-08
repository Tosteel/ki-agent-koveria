from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .document_import import parse_product_document
from .models import CompetitiveDocumentImportRequest, CompetitiveDocumentImportResponse


TOOL_NAME = "competitive_parse_document"


def register(registry: ToolRegistry) -> None:
    def tool_competitive_parse_document(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = CompetitiveDocumentImportRequest(**args)
        parsed = parse_product_document(
            path=req.path,
            max_chars=req.max_chars,
            user_root=ctx.settings.user_dir(ctx.user_id),
            work_root=ctx.settings.user_work_dir(ctx.user_id),
        )
        return CompetitiveDocumentImportResponse(parsed_doc=parsed).model_dump()

    registry.register(TOOL_NAME, tool_competitive_parse_document, request_model=CompetitiveDocumentImportRequest)
