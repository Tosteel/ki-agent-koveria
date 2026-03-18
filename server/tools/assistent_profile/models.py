from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class AssistentProfileCreateRequest(BaseModel):
    assistent_profile_name: str = Field(..., min_length=1)
    codename: str = ""
    instructions: List[str] = Field(default_factory=list)
    rules: Dict[str, Any] = Field(default_factory=dict)


class AssistentProfileGetRequest(BaseModel):
    assistent_profile_name: str = Field(..., min_length=1)


class AssistentProfileUpdateRequest(BaseModel):
    assistent_profile_name: str = Field(..., min_length=1)
    codename: str = ""
    instructions_add: List[str] = Field(default_factory=list)
    rules_patch: Dict[str, Any] = Field(default_factory=dict)
    raw_patch: Dict[str, Any] = Field(default_factory=dict)


class AssistentProfileCheckRequest(BaseModel):
    assistent_profile_name: str = Field(..., min_length=1)
    action: str = Field(..., min_length=1)
    context_text: str = ""
    context: Dict[str, Any] = Field(default_factory=dict)


class AssistentProfileResponse(BaseModel):
    ok: bool = True
    assistent_profile_name: str = ""
    path: str = ""
    profile: Dict[str, Any] = Field(default_factory=dict)
    text: str = ""


class AssistentProfileCheckResponse(BaseModel):
    ok: bool = True
    assistent_profile_name: str = ""
    action: str = ""
    allowed: bool = True
    reasons: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    matched_rules: List[str] = Field(default_factory=list)
    text: str = ""

