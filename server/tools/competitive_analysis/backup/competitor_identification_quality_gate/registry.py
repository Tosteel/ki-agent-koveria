from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .competitor_identification_quality_gate import run_competitor_identification_quality_gate
from .models import (
    CompetitorIdentificationQualityGateRequest,
    CompetitorIdentificationQualityGateResponse,
)


TOOL_NAME = "competitor_identification_quality_gate"


def register(registry: ToolRegistry) -> None:
    def tool_competitor_identification_quality_gate(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = CompetitorIdentificationQualityGateRequest(**args)
        cleaned, report = run_competitor_identification_quality_gate(
            competitor_list=req.competitor_list,
            competitor_list_path=req.competitor_list_path,
            product_profile=req.product_profile,
            product_profile_path=req.product_profile_path,
            provider=req.provider,
            min_relevance_score=req.min_relevance_score,
            drop_generic_listing_pages=req.drop_generic_listing_pages,
            drop_weak_unknown_candidates=req.drop_weak_unknown_candidates,
            drop_manufacturer_nodes_without_model_signal=req.drop_manufacturer_nodes_without_model_signal,
            dedupe_by_name_and_domain=req.dedupe_by_name_and_domain,
            enable_llm_snippet_validation=req.enable_llm_snippet_validation,
            llm_min_keep_confidence=req.llm_min_keep_confidence,
            max_llm_checks=req.max_llm_checks,
            user_root=ctx.settings.user_dir(ctx.user_id),
            work_root=ctx.settings.user_work_dir(ctx.user_id),
        )
        return CompetitorIdentificationQualityGateResponse(
            competitor_list=cleaned,
            quality_report=report,
        ).model_dump()

    registry.register(TOOL_NAME, tool_competitor_identification_quality_gate, request_model=CompetitorIdentificationQualityGateRequest)
