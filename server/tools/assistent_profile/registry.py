from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

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


def register(registry: ToolRegistry) -> None:
    def tool_assistent_profile_create(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = AssistentProfileCreateRequest(**args)
        result = assistent_profile_create(
            user_dir=ctx.settings.user_dir(ctx.user_id),
            assistent_profile_name=req.assistent_profile_name,
            codename=req.codename,
            instructions=req.instructions,
            rules=req.rules,
        )
        return AssistentProfileResponse(**result).model_dump()

    def tool_assistent_profile_get(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = AssistentProfileGetRequest(**args)
        result = assistent_profile_get(
            user_dir=ctx.settings.user_dir(ctx.user_id),
            assistent_profile_name=req.assistent_profile_name,
        )
        return AssistentProfileResponse(**result).model_dump()

    def tool_assistent_profile_update(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = AssistentProfileUpdateRequest(**args)
        result = assistent_profile_update(
            user_dir=ctx.settings.user_dir(ctx.user_id),
            assistent_profile_name=req.assistent_profile_name,
            codename=req.codename,
            instructions_add=req.instructions_add,
            rules_patch=req.rules_patch,
            raw_patch=req.raw_patch,
        )
        return AssistentProfileResponse(**result).model_dump()

    def tool_assistent_profile_check(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = AssistentProfileCheckRequest(**args)
        result = assistent_profile_check(
            user_dir=ctx.settings.user_dir(ctx.user_id),
            assistent_profile_name=req.assistent_profile_name,
            action=req.action,
            context_text=req.context_text,
            context=req.context,
        )
        return AssistentProfileCheckResponse(**result).model_dump()

    registry.register(
        "assistent_profile_create",
        tool_assistent_profile_create,
        request_model=AssistentProfileCreateRequest,
        response_model=AssistentProfileResponse,
    )
    registry.register(
        "assistent_profile_get",
        tool_assistent_profile_get,
        request_model=AssistentProfileGetRequest,
        response_model=AssistentProfileResponse,
    )
    registry.register(
        "assistent_profile_update",
        tool_assistent_profile_update,
        request_model=AssistentProfileUpdateRequest,
        response_model=AssistentProfileResponse,
    )
    registry.register(
        "assistent_profile_check",
        tool_assistent_profile_check,
        request_model=AssistentProfileCheckRequest,
        response_model=AssistentProfileCheckResponse,
    )

