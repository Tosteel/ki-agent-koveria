from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class ListSkillsRequest(BaseModel):
    include_descriptions: bool = True


class SkillItem(BaseModel):
    name: str
    description: str = ""


class ListSkillsResponse(BaseModel):
    count: int
    skills: List[SkillItem] = Field(default_factory=list)
    text: str

