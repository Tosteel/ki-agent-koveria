from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .customer_support import customer_support_review_ticket_create, customer_support_policy_check, customer_support_reply_score, customer_support_review_ticket_update
from .models import (
    CreateReviewTicketRequest,
    CreateReviewTicketResponse,
    PolicyCheckRequest,
    PolicyCheckResponse,
    ScoreReplyRequest,
    ScoreReplyResponse,
    UpdateReviewTicketRequest,
    UpdateReviewTicketResponse,
)


def register(registry: ToolRegistry) -> None:
    def tool_score_reply(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = ScoreReplyRequest(**args)
        result = customer_support_reply_score(
            user_message=req.user_message,
            draft_reply=req.draft_reply,
            knowledge_evidence=req.knowledge_evidence,
            require_actionable=req.require_actionable,
        )
        return ScoreReplyResponse(**result).model_dump()

    def tool_create_review_ticket(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = CreateReviewTicketRequest(**args)
        result = customer_support_review_ticket_create(
            user_dir=ctx.settings.user_dir(ctx.user_id),
            title=req.title,
            user_message=req.user_message,
            draft_reply=req.draft_reply,
            score=req.score,
            reasons=req.reasons,
            priority=req.priority,
            metadata=req.metadata,
        )
        return CreateReviewTicketResponse(**result).model_dump()

    def tool_update_review_ticket(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = UpdateReviewTicketRequest(**args)
        result = customer_support_review_ticket_update(
            user_dir=ctx.settings.user_dir(ctx.user_id),
            ticket_id=req.ticket_id,
            status=req.status,
            reviewer_note=req.reviewer_note,
            assignee=req.assignee,
            draft_reply=req.draft_reply,
            score=req.score,
            resolution=req.resolution,
        )
        return UpdateReviewTicketResponse(**result).model_dump()

    def tool_policy_check(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = PolicyCheckRequest(**args)
        result = customer_support_policy_check(
            text=req.text,
            policy_profile=req.policy_profile,
            strict_mode=req.strict_mode,
        )
        return PolicyCheckResponse(**result).model_dump()

    registry.register(
        "customer_support_reply_score",
        tool_score_reply,
        request_model=ScoreReplyRequest,
        response_model=ScoreReplyResponse,
    )
    registry.register(
        "customer_support_review_ticket_create",
        tool_create_review_ticket,
        request_model=CreateReviewTicketRequest,
        response_model=CreateReviewTicketResponse,
    )
    registry.register(
        "customer_support_review_ticket_update",
        tool_update_review_ticket,
        request_model=UpdateReviewTicketRequest,
        response_model=UpdateReviewTicketResponse,
    )
    registry.register(
        "customer_support_policy_check",
        tool_policy_check,
        request_model=PolicyCheckRequest,
        response_model=PolicyCheckResponse,
    )

