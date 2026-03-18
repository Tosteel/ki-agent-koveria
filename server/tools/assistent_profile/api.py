from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .assistent_profile import (
    assistent_profile_check,
    assistent_profile_create,
    assistent_profile_get,
    assistent_profile_update,
)
from .models import (
    AssistentProfileCheckRequest,
    AssistentProfileCheckResponse,
    AssistentProfileCreateRequest,
    AssistentProfileGetRequest,
    AssistentProfileResponse,
    AssistentProfileUpdateRequest,
)


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    @router.post("/tools/assistent-profile/create", response_model=AssistentProfileResponse)
    def create_route(
        req: AssistentProfileCreateRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> AssistentProfileResponse:
        ensure_user_dirs(s, user_id)
        result = assistent_profile_create(
            user_dir=s.user_dir(user_id),
            assistent_profile_name=req.assistent_profile_name,
            codename=req.codename,
            instructions=req.instructions,
            rules=req.rules,
        )
        return AssistentProfileResponse(**result)

    @router.post("/tools/assistent-profile/get", response_model=AssistentProfileResponse)
    def get_route(
        req: AssistentProfileGetRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> AssistentProfileResponse:
        ensure_user_dirs(s, user_id)
        result = assistent_profile_get(
            user_dir=s.user_dir(user_id),
            assistent_profile_name=req.assistent_profile_name,
        )
        return AssistentProfileResponse(**result)

    @router.post("/tools/assistent-profile/update", response_model=AssistentProfileResponse)
    def update_route(
        req: AssistentProfileUpdateRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> AssistentProfileResponse:
        ensure_user_dirs(s, user_id)
        result = assistent_profile_update(
            user_dir=s.user_dir(user_id),
            assistent_profile_name=req.assistent_profile_name,
            codename=req.codename,
            instructions_add=req.instructions_add,
            rules_patch=req.rules_patch,
            raw_patch=req.raw_patch,
        )
        return AssistentProfileResponse(**result)

    @router.post("/tools/assistent-profile/check", response_model=AssistentProfileCheckResponse)
    def check_route(
        req: AssistentProfileCheckRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> AssistentProfileCheckResponse:
        ensure_user_dirs(s, user_id)
        result = assistent_profile_check(
            user_dir=s.user_dir(user_id),
            assistent_profile_name=req.assistent_profile_name,
            action=req.action,
            context_text=req.context_text,
            context=req.context,
        )
        return AssistentProfileCheckResponse(**result)

    return router

