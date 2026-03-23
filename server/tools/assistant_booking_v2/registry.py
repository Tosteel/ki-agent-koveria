from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry
from server.assistants.booking_assistant_v2.models import (
    BookingAssistantV2ApproveRequest,
    BookingAssistantV2CounterofferRequest,
    BookingAssistantV2PendingApplyRequest,
    BookingAssistantV2RejectRequest,
    BookingAssistantV2ReviewActionResponse,
    BookingAssistantV2ReviewsResponse,
    BookingAssistantV2PendingNextResponse,
    BookingAssistantV2StatusMeetingResponse,
)
from server.assistants.booking_assistant_v2.service import (
    apply_pending_action,
    approve_review,
    counteroffer_review,
    get_pending_next,
    get_reviews,
    get_status_meeting,
    reject_review,
)

from .models import (
    BookingAssistantV2PendingApplyToolRequest,
    BookingAssistantV2PendingNextToolRequest,
    BookingAssistantV2ReviewApproveToolRequest,
    BookingAssistantV2ReviewCounterofferToolRequest,
    BookingAssistantV2ReviewRejectToolRequest,
    BookingAssistantV2ReviewsToolRequest,
    BookingAssistantV2StatusMeetingToolRequest,
)


def register(registry: ToolRegistry) -> None:
    def tool_status_meeting(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = BookingAssistantV2StatusMeetingToolRequest(**args)
        result = get_status_meeting(user_id=ctx.user_id, settings=ctx.settings, since=req.since)
        return BookingAssistantV2StatusMeetingResponse(**result.model_dump()).model_dump()

    def tool_reviews(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = BookingAssistantV2ReviewsToolRequest(**args)
        result = get_reviews(user_id=ctx.user_id, settings=ctx.settings, status=req.status)
        return BookingAssistantV2ReviewsResponse(**result.model_dump()).model_dump()

    def tool_pending_next(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        BookingAssistantV2PendingNextToolRequest(**args)
        result = get_pending_next(user_id=ctx.user_id, settings=ctx.settings)
        return BookingAssistantV2PendingNextResponse(**result.model_dump()).model_dump()

    def tool_pending_apply(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = BookingAssistantV2PendingApplyToolRequest(**args)
        result = apply_pending_action(
            user_id=ctx.user_id,
            settings=ctx.settings,
            api_key=ctx.api_key,
            review_id=req.review_id,
            req=BookingAssistantV2PendingApplyRequest(
                provider=req.provider,
                action=req.action,
                edited_body=req.edited_body,
                subject=req.subject,
                reason=req.reason,
                send_to_customer=req.send_to_customer,
            ),
        )
        return BookingAssistantV2ReviewActionResponse(**result.model_dump()).model_dump()

    def tool_review_approve(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = BookingAssistantV2ReviewApproveToolRequest(**args)
        result = approve_review(
            user_id=ctx.user_id,
            settings=ctx.settings,
            api_key=ctx.api_key,
            review_id=req.review_id,
            req=BookingAssistantV2ApproveRequest(
                provider=req.provider,
                edited_body=req.edited_body,
                subject=req.subject,
            ),
        )
        return BookingAssistantV2ReviewActionResponse(**result.model_dump()).model_dump()

    def tool_review_reject(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = BookingAssistantV2ReviewRejectToolRequest(**args)
        result = reject_review(
            user_id=ctx.user_id,
            settings=ctx.settings,
            api_key=ctx.api_key,
            review_id=req.review_id,
            req=BookingAssistantV2RejectRequest(
                provider=req.provider,
                edited_body=req.edited_body,
                subject=req.subject,
                reason=req.reason,
                send_to_customer=req.send_to_customer,
            ),
        )
        return BookingAssistantV2ReviewActionResponse(**result.model_dump()).model_dump()

    def tool_review_counteroffer(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = BookingAssistantV2ReviewCounterofferToolRequest(**args)
        result = counteroffer_review(
            user_id=ctx.user_id,
            settings=ctx.settings,
            api_key=ctx.api_key,
            review_id=req.review_id,
            req=BookingAssistantV2CounterofferRequest(
                provider=req.provider,
                edited_body=req.edited_body,
                subject=req.subject,
            ),
        )
        return BookingAssistantV2ReviewActionResponse(**result.model_dump()).model_dump()

    registry.register(
        "booking_assistant_v2_status_meeting",
        tool_status_meeting,
        request_model=BookingAssistantV2StatusMeetingToolRequest,
        response_model=BookingAssistantV2StatusMeetingResponse,
    )
    registry.register(
        "booking_assistant_v2_reviews",
        tool_reviews,
        request_model=BookingAssistantV2ReviewsToolRequest,
        response_model=BookingAssistantV2ReviewsResponse,
    )
    registry.register(
        "booking_assistant_v2_pending_next",
        tool_pending_next,
        request_model=BookingAssistantV2PendingNextToolRequest,
        response_model=BookingAssistantV2PendingNextResponse,
    )
    registry.register(
        "booking_assistant_v2_pending_apply",
        tool_pending_apply,
        request_model=BookingAssistantV2PendingApplyToolRequest,
        response_model=BookingAssistantV2ReviewActionResponse,
    )
    registry.register(
        "booking_assistant_v2_review_approve",
        tool_review_approve,
        request_model=BookingAssistantV2ReviewApproveToolRequest,
        response_model=BookingAssistantV2ReviewActionResponse,
    )
    registry.register(
        "booking_assistant_v2_review_reject",
        tool_review_reject,
        request_model=BookingAssistantV2ReviewRejectToolRequest,
        response_model=BookingAssistantV2ReviewActionResponse,
    )
    registry.register(
        "booking_assistant_v2_review_counteroffer",
        tool_review_counteroffer,
        request_model=BookingAssistantV2ReviewCounterofferToolRequest,
        response_model=BookingAssistantV2ReviewActionResponse,
    )
