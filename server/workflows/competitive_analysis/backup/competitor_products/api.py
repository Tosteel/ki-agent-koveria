from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .competitor_products import extract_competitor_products
from .models import CompetitorProductsRequest, CompetitorProductsResponse


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    #@router.post('/competitive/competitors/products', response_model=CompetitorProductsResponse)
    def competitive_competitor_products(
        req: CompetitorProductsRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> CompetitorProductsResponse:
        ensure_user_dirs(s, user_id)
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
            user_root=s.user_dir(user_id),
            work_root=s.user_work_dir(user_id),
        )
        return CompetitorProductsResponse(competitor_products_results=result)

    return router
