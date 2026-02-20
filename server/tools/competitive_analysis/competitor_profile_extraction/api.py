from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .competitor_profile_extraction import (
    extract_competitor_profiles,
    merge_competitor_profiles_parts,
    verify_competitor_source_registry,
)
from .models import (
    CompetitorProfileExtractionRequest,
    CompetitorProfileExtractionResponse,
    CompetitorProfileMergeRequest,
    CompetitorProfileMergeResponse,
    SourceRegistryVerifyRequest,
    SourceRegistryVerifyResponse,
)


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    @router.post('/competitive/competitors/profiles', response_model=CompetitorProfileExtractionResponse)
    def competitive_competitor_profiles(
        req: CompetitorProfileExtractionRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> CompetitorProfileExtractionResponse:
        ensure_user_dirs(s, user_id)
        result = extract_competitor_profiles(
            competitor_list=req.competitor_list,
            competitor_list_path=req.competitor_list_path,
            source_registry=req.source_registry,
            source_registry_path=req.source_registry_path,
            provider=req.provider,
            max_competitors=req.max_competitors,
            max_pages_per_competitor=req.max_pages_per_competitor,
            offset=req.offset,
            limit=req.limit,
            verbose_progress=req.verbose_progress,
            registry_first=req.registry_first,
            min_active_sources_for_search=req.min_active_sources_for_search,
            user_root=s.user_dir(user_id),
            work_root=s.user_work_dir(user_id),
        )
        return CompetitorProfileExtractionResponse(competitor_profiles=result)

    @router.post('/competitive/competitors/profiles/merge', response_model=CompetitorProfileMergeResponse)
    def competitive_competitor_profiles_merge(
        req: CompetitorProfileMergeRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> CompetitorProfileMergeResponse:
        ensure_user_dirs(s, user_id)
        result = merge_competitor_profiles_parts(
            part_paths=req.part_paths,
            provider=req.provider,
            user_root=s.user_dir(user_id),
            work_root=s.user_work_dir(user_id),
        )
        return CompetitorProfileMergeResponse(competitor_profiles=result)

    @router.post('/competitive/competitors/sources/verify', response_model=SourceRegistryVerifyResponse)
    def competitive_competitor_source_registry_verify(
        req: SourceRegistryVerifyRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> SourceRegistryVerifyResponse:
        ensure_user_dirs(s, user_id)
        result = verify_competitor_source_registry(
            competitor_list=req.competitor_list,
            competitor_list_path=req.competitor_list_path,
            source_registry=req.source_registry,
            source_registry_path=req.source_registry_path,
            max_urls_per_competitor=req.max_urls_per_competitor,
            timeout_seconds=req.timeout_seconds,
            include_fallbacks=req.include_fallbacks,
            user_root=s.user_dir(user_id),
            work_root=s.user_work_dir(user_id),
        )
        return SourceRegistryVerifyResponse(source_registry=result)

    return router
