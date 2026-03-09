from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .competitor_products import extract_competitor_products
from .models import CompetitorProductsRequest, CompetitorProductsResponse


TOOL_NAME = "competitor_products"


def register(registry: ToolRegistry) -> None:
    def tool_competitor_products(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = CompetitorProductsRequest(**args)
        result = extract_competitor_products(
            competitor_search_results=req.competitor_search_results,
            competitor_search_results_path=req.competitor_search_results_path,
            product_profile=req.product_profile,
            product_profile_path=req.product_profile_path,
            provider=req.provider,
            per_query_results=req.per_query_results,
            top_products_per_company=req.top_products_per_company,
            max_queries_per_company=req.max_queries_per_company,
            semantic_weight=req.semantic_weight,
            feature_match_weight=req.feature_match_weight,
            performance_similarity_weight=req.performance_similarity_weight,
            price_weight=req.price_weight,
            emit_all_candidates=req.emit_all_candidates,
            manufacturer_domain_only=req.manufacturer_domain_only,
            verbose_terminal=req.verbose_terminal,
            user_root=ctx.settings.user_dir(ctx.user_id),
            work_root=ctx.settings.user_work_dir(ctx.user_id),
        )
        return CompetitorProductsResponse(competitor_products_results=result).model_dump()

    registry.register(TOOL_NAME, tool_competitor_products, request_model=CompetitorProductsRequest)
