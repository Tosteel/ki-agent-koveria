from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry
from server.assistants.booking_assistant_v3.models import (
    BookingAssistantV3PendingApplyRequest,
    BookingAssistantV3ReviewActionResponse,
    BookingAssistantV3ReviewsResponse,
    BookingAssistantV3PendingNextResponse,
    BookingAssistantV3StatusResponse,
)
from server.assistants.booking_assistant_v3.service import (
    apply_pending_action,
    get_pending_next,
    get_reviews,
    get_status,
)

from .models import (
    BookingAssistantV3PendingApplyToolRequest,
    BookingAssistantV3PendingNextToolRequest,
    BookingAssistantV3ReviewsToolRequest,
    BookingAssistantV3StatusToolRequest,
)


def register(registry: ToolRegistry) -> None:
    def tool_status(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = BookingAssistantV3StatusToolRequest(**args)
        result = get_status(user_id=ctx.user_id, settings=ctx.settings, since=req.since)
        return BookingAssistantV3StatusResponse(**result.model_dump()).model_dump()

    def tool_reviews(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = BookingAssistantV3ReviewsToolRequest(**args)
        result = get_reviews(user_id=ctx.user_id, settings=ctx.settings, status=req.status, kind=req.kind)
        return BookingAssistantV3ReviewsResponse(**result.model_dump()).model_dump()

    def tool_pending_next(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = BookingAssistantV3PendingNextToolRequest(**args)
        result = get_pending_next(user_id=ctx.user_id, settings=ctx.settings, queue=req.queue)
        return BookingAssistantV3PendingNextResponse(**result.model_dump()).model_dump()

    def tool_pending_apply(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = BookingAssistantV3PendingApplyToolRequest(**args)
        result = apply_pending_action(
            user_id=ctx.user_id,
            settings=ctx.settings,
            api_key=ctx.api_key,
            review_id=req.review_id,
            req=BookingAssistantV3PendingApplyRequest(
                provider=req.provider,
                action=req.action,
                edited_body=req.edited_body,
                subject=req.subject,
                reason=req.reason,
                send_to_customer=req.send_to_customer,
            ),
        )
        return BookingAssistantV3ReviewActionResponse(**result.model_dump()).model_dump()

    registry.register("booking_assistant_v3_status", tool_status, request_model=BookingAssistantV3StatusToolRequest, response_model=BookingAssistantV3StatusResponse)
    registry.register("booking_assistant_v3_reviews", tool_reviews, request_model=BookingAssistantV3ReviewsToolRequest, response_model=BookingAssistantV3ReviewsResponse)
    registry.register("booking_assistant_v3_pending_next", tool_pending_next, request_model=BookingAssistantV3PendingNextToolRequest, response_model=BookingAssistantV3PendingNextResponse)
    registry.register("booking_assistant_v3_pending_apply", tool_pending_apply, request_model=BookingAssistantV3PendingApplyToolRequest, response_model=BookingAssistantV3ReviewActionResponse)
