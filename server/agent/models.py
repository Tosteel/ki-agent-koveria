from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field


class AgentStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str = Field(..., min_length=1)
    args: Dict[str, Any] = Field(default_factory=dict)


class AgentToolPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: List[AgentStep] = Field(default_factory=list)
