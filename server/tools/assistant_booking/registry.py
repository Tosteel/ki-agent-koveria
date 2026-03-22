from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry
from server.assistants.booking_assistant.models import (
    BookingAssistantApproveRequest,
    BookingAssistantCounterofferRequest,
    BookingAssistantPendingApplyRequest,
    BookingAssistantRejectRequest,
    BookingAssistantReviewActionResponse,
    BookingAssistantReviewsResponse,
    BookingAssistantPendingNextResponse,
    BookingAssistantStatusMeetingResponse,
)
from server.assistants.booking_assistant.service import (
    apply_pending_action,
    approve_review,
    counteroffer_review,
    get_pending_next,
    get_reviews,
    get_status_meeting,
    reject_review,
)

from .models import (
    BookingAssistantPendingApplyToolRequest,
    BookingAssistantPendingNextToolRequest,
    BookingAssistantReviewApproveToolRequest,
    BookingAssistantReviewCounterofferToolRequest,
    BookingAssistantReviewRejectToolRequest,
    BookingAssistantReviewsToolRequest,
    BookingAssistantStatusMeetingToolRequest,
)


def register(registry: ToolRegistry) -> None:
    def tool_booking_assistant_status_meeting(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = BookingAssistantStatusMeetingToolRequest(**args)
        result = get_status_meeting(
            user_id=ctx.user_id,
            settings=ctx.settings,
            since=req.since,
        )
        return BookingAssistantStatusMeetingResponse(**result.model_dump()).model_dump()

    def tool_booking_assistant_reviews(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = BookingAssistantReviewsToolRequest(**args)
        result = get_reviews(
            user_id=ctx.user_id,
            settings=ctx.settings,
            status=req.status,
        )
        return BookingAssistantReviewsResponse(**result.model_dump()).model_dump()

    def tool_booking_assistant_pending_next(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        BookingAssistantPendingNextToolRequest(**args)
        result = get_pending_next(
            user_id=ctx.user_id,
            settings=ctx.settings,
        )
        return BookingAssistantPendingNextResponse(**result.model_dump()).model_dump()

    def tool_booking_assistant_pending_apply(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = BookingAssistantPendingApplyToolRequest(**args)
        result = apply_pending_action(
            user_id=ctx.user_id,
            settings=ctx.settings,
            api_key=ctx.api_key,
            review_id=req.review_id,
            req=BookingAssistantPendingApplyRequest(
                provider=req.provider,
                action=req.action,
                edited_body=req.edited_body,
                subject=req.subject,
                reason=req.reason,
                send_to_customer=req.send_to_customer,
            ),
        )
        return BookingAssistantReviewActionResponse(**result.model_dump()).model_dump()

    def tool_booking_assistant_review_approve(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = BookingAssistantReviewApproveToolRequest(**args)
        result = approve_review(
            user_id=ctx.user_id,
            settings=ctx.settings,
            api_key=ctx.api_key,
            review_id=req.review_id,
            req=BookingAssistantApproveRequest(
                provider=req.provider,
                edited_body=req.edited_body,
                subject=req.subject,
            ),
        )
        return BookingAssistantReviewActionResponse(**result.model_dump()).model_dump()

    def tool_booking_assistant_review_reject(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = BookingAssistantReviewRejectToolRequest(**args)
        result = reject_review(
            user_id=ctx.user_id,
            settings=ctx.settings,
            api_key=ctx.api_key,
            review_id=req.review_id,
            req=BookingAssistantRejectRequest(
                provider=req.provider,
                edited_body=req.edited_body,
                subject=req.subject,
                reason=req.reason,
                send_to_customer=req.send_to_customer,
            ),
        )
        return BookingAssistantReviewActionResponse(**result.model_dump()).model_dump()

    def tool_booking_assistant_review_counteroffer(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = BookingAssistantReviewCounterofferToolRequest(**args)
        result = counteroffer_review(
            user_id=ctx.user_id,
            settings=ctx.settings,
            api_key=ctx.api_key,
            review_id=req.review_id,
            req=BookingAssistantCounterofferRequest(
                provider=req.provider,
                edited_body=req.edited_body,
                subject=req.subject,
            ),
        )
        return BookingAssistantReviewActionResponse(**result.model_dump()).model_dump()

    registry.register(
        "booking_assistant_status_meeting",
        tool_booking_assistant_status_meeting,
        request_model=BookingAssistantStatusMeetingToolRequest,
        response_model=BookingAssistantStatusMeetingResponse,
    )
    registry.register(
        "booking_assistant_reviews",
        tool_booking_assistant_reviews,
        request_model=BookingAssistantReviewsToolRequest,
        response_model=BookingAssistantReviewsResponse,
    )
    registry.register(
        "booking_assistant_pending_next",
        tool_booking_assistant_pending_next,
        request_model=BookingAssistantPendingNextToolRequest,
        response_model=BookingAssistantPendingNextResponse,
    )
    registry.register(
        "booking_assistant_pending_apply",
        tool_booking_assistant_pending_apply,
        request_model=BookingAssistantPendingApplyToolRequest,
        response_model=BookingAssistantReviewActionResponse,
    )
    registry.register(
        "booking_assistant_review_approve",
        tool_booking_assistant_review_approve,
        request_model=BookingAssistantReviewApproveToolRequest,
        response_model=BookingAssistantReviewActionResponse,
    )
    registry.register(
        "booking_assistant_review_reject",
        tool_booking_assistant_review_reject,
        request_model=BookingAssistantReviewRejectToolRequest,
        response_model=BookingAssistantReviewActionResponse,
    )
    registry.register(
        "booking_assistant_review_counteroffer",
        tool_booking_assistant_review_counteroffer,
        request_model=BookingAssistantReviewCounterofferToolRequest,
        response_model=BookingAssistantReviewActionResponse,
    )
