from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .competitor_profile_extraction import (
    extract_competitor_profiles,
    merge_competitor_profiles_parts,
    verify_competitor_source_registry,
)
from .models import (
    CompetitorSourceRegistry,
    CompetitorProfileExtractionRequest,
    CompetitorProfileExtractionResponse,
    CompetitorProfileMergeRequest,
    CompetitorProfileMergeResponse,
    SourceRegistryVerifyRequest,
    SourceRegistryVerifyResponse,
)


TOOL_NAME = "competitive_extract_competitor_profiles"
MERGE_TOOL_NAME = "competitive_merge_competitor_profiles"
VERIFY_SOURCES_TOOL_NAME = "competitive_verify_competitor_source_registry"


def register(registry: ToolRegistry) -> None:
    def tool_competitor_profile_extraction(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = CompetitorProfileExtractionRequest(**args)
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
            user_root=ctx.settings.user_dir(ctx.user_id),
            work_root=ctx.settings.user_work_dir(ctx.user_id),
        )
        return CompetitorProfileExtractionResponse(competitor_profiles=result).model_dump()

    registry.register(TOOL_NAME, tool_competitor_profile_extraction, request_model=CompetitorProfileExtractionRequest)

    def tool_competitor_profile_merge(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = CompetitorProfileMergeRequest(**args)
        result = merge_competitor_profiles_parts(
            part_paths=req.part_paths,
            provider=req.provider,
            user_root=ctx.settings.user_dir(ctx.user_id),
            work_root=ctx.settings.user_work_dir(ctx.user_id),
        )
        return CompetitorProfileMergeResponse(competitor_profiles=result).model_dump()

    registry.register(MERGE_TOOL_NAME, tool_competitor_profile_merge, request_model=CompetitorProfileMergeRequest)

    def tool_competitor_source_registry_verify(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = SourceRegistryVerifyRequest(**args)
        result: CompetitorSourceRegistry = verify_competitor_source_registry(
            competitor_list=req.competitor_list,
            competitor_list_path=req.competitor_list_path,
            source_registry=req.source_registry,
            source_registry_path=req.source_registry_path,
            max_urls_per_competitor=req.max_urls_per_competitor,
            timeout_seconds=req.timeout_seconds,
            include_fallbacks=req.include_fallbacks,
            user_root=ctx.settings.user_dir(ctx.user_id),
            work_root=ctx.settings.user_work_dir(ctx.user_id),
        )
        return SourceRegistryVerifyResponse(source_registry=result).model_dump()

    registry.register(VERIFY_SOURCES_TOOL_NAME, tool_competitor_source_registry_verify, request_model=SourceRegistryVerifyRequest)
