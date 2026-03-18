from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

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


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    @router.post("/tools/customer-support/score-reply", response_model=ScoreReplyResponse)
    def score_reply_route(
        req: ScoreReplyRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> ScoreReplyResponse:
        ensure_user_dirs(s, user_id)
        result = customer_support_reply_score(
            user_message=req.user_message,
            draft_reply=req.draft_reply,
            knowledge_evidence=req.knowledge_evidence,
            require_actionable=req.require_actionable,
        )
        return ScoreReplyResponse(**result)

    @router.post("/tools/customer-support/review-ticket/create", response_model=CreateReviewTicketResponse)
    def create_review_ticket_route(
        req: CreateReviewTicketRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> CreateReviewTicketResponse:
        ensure_user_dirs(s, user_id)
        result = customer_support_review_ticket_create(
            user_dir=s.user_dir(user_id),
            title=req.title,
            user_message=req.user_message,
            draft_reply=req.draft_reply,
            score=req.score,
            reasons=req.reasons,
            priority=req.priority,
            metadata=req.metadata,
        )
        return CreateReviewTicketResponse(**result)

    @router.post("/tools/customer-support/review-ticket/update", response_model=UpdateReviewTicketResponse)
    def update_review_ticket_route(
        req: UpdateReviewTicketRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> UpdateReviewTicketResponse:
        ensure_user_dirs(s, user_id)
        result = customer_support_review_ticket_update(
            user_dir=s.user_dir(user_id),
            ticket_id=req.ticket_id,
            status=req.status,
            reviewer_note=req.reviewer_note,
            assignee=req.assignee,
            draft_reply=req.draft_reply,
            score=req.score,
            resolution=req.resolution,
        )
        return UpdateReviewTicketResponse(**result)

    @router.post("/tools/customer-support/policy-check", response_model=PolicyCheckResponse)
    def policy_check_route(
        req: PolicyCheckRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> PolicyCheckResponse:
        ensure_user_dirs(s, user_id)
        result = customer_support_policy_check(
            text=req.text,
            policy_profile=req.policy_profile,
            strict_mode=req.strict_mode,
        )
        return PolicyCheckResponse(**result)

    return router

