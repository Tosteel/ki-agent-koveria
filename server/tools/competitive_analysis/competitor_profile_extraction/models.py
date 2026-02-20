from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class MappedFeature(BaseModel):
    schema_feature: str
    raw_name: str = ""
    value: str = ""
    unit: str = ""
    normalized_value: float | int | str | None = None
    normalized_unit: str = ""
    source_url: str = ""
    evidence: str = ""


class PriceInfo(BaseModel):
    raw: str
    value: float | int | None = None
    currency: str = ""
    package: str = ""
    source_url: str = ""


class SourceEvidence(BaseModel):
    url: str
    title: str = ""
    retrieved_at: str = ""
    excerpt: str = ""


class DataQuality(BaseModel):
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    completeness: float = Field(default=0.0, ge=0.0, le=1.0)
    freshness_days: int | None = None
    notes: List[str] = Field(default_factory=list)


class CompetitorProfile(BaseModel):
    name: str
    url: str
    cluster: str = "unknown"
    status: str = "usable"  # usable | weak | empty
    mapped_features: List[MappedFeature] = Field(default_factory=list)
    prices: List[PriceInfo] = Field(default_factory=list)
    packages: List[str] = Field(default_factory=list)
    sources: List[SourceEvidence] = Field(default_factory=list)
    data_quality: DataQuality = Field(default_factory=DataQuality)


class CompetitorProfiles(BaseModel):
    schema_version: str = "1.0"
    provider: str = "openai"
    target_feature_schema: List[str] = Field(default_factory=list)
    competitor_profiles: List[CompetitorProfile] = Field(default_factory=list)
    extraction_warnings: List[str] = Field(default_factory=list)
    batch_offset: int = 0
    batch_limit: int | None = None
    batch_total_candidates: int = 0
    processed_count: int = 0


class CompetitorProfileExtractionRequest(BaseModel):
    competitor_list: Optional[Dict[str, Any]] = None
    competitor_list_path: Optional[str] = None
    source_registry: Optional[Dict[str, Any]] = None
    source_registry_path: Optional[str] = None
    provider: str = "openai"
    max_competitors: int = Field(default=10, ge=1, le=50)
    max_pages_per_competitor: int = Field(default=3, ge=1, le=10)
    offset: int = Field(default=0, ge=0)
    limit: Optional[int] = Field(default=None, ge=1, le=50)
    verbose_progress: bool = True
    registry_first: bool = True
    min_active_sources_for_search: int = Field(default=2, ge=0, le=10)

    @model_validator(mode="after")
    def _validate_input(self) -> "CompetitorProfileExtractionRequest":
        if not self.competitor_list and not (self.competitor_list_path or "").strip():
            raise ValueError("Either competitor_list or competitor_list_path must be provided.")
        return self


class CompetitorProfileExtractionResponse(BaseModel):
    competitor_profiles: CompetitorProfiles


class CompetitorProfileMergeRequest(BaseModel):
    part_paths: List[str] = Field(default_factory=list)
    provider: str = "openai"

    @model_validator(mode="after")
    def _validate_input(self) -> "CompetitorProfileMergeRequest":
        if not self.part_paths:
            raise ValueError("part_paths must not be empty.")
        return self


class CompetitorProfileMergeResponse(BaseModel):
    competitor_profiles: CompetitorProfiles


class SourceRegistryEntry(BaseModel):
    url: str
    kind: str = "fallback"  # primary | fallback
    priority: int = 50
    active: bool = True
    last_checked_at: str = ""
    last_status: str = ""
    last_error: str = ""
    title_hint: str = ""


class SourceRegistryCompetitor(BaseModel):
    name: str
    entries: List[SourceRegistryEntry] = Field(default_factory=list)


class CompetitorSourceRegistry(BaseModel):
    schema_version: str = "1.0"
    competitors: List[SourceRegistryCompetitor] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class SourceRegistryVerifyRequest(BaseModel):
    competitor_list: Optional[Dict[str, Any]] = None
    competitor_list_path: Optional[str] = None
    source_registry: Optional[Dict[str, Any]] = None
    source_registry_path: Optional[str] = None
    max_urls_per_competitor: int = Field(default=6, ge=1, le=20)
    timeout_seconds: int = Field(default=25, ge=5, le=90)
    include_fallbacks: bool = True

    @model_validator(mode="after")
    def _validate_input(self) -> "SourceRegistryVerifyRequest":
        if not self.source_registry and not (self.source_registry_path or "").strip():
            if not self.competitor_list and not (self.competitor_list_path or "").strip():
                raise ValueError("Either source_registry/source_registry_path or competitor_list/competitor_list_path must be provided.")
        return self


class SourceRegistryVerifyResponse(BaseModel):
    source_registry: CompetitorSourceRegistry


__all__ = [
    "CompetitorProfileExtractionRequest",
    "CompetitorProfileExtractionResponse",
    "CompetitorProfileMergeRequest",
    "CompetitorProfileMergeResponse",
    "SourceRegistryVerifyRequest",
    "SourceRegistryVerifyResponse",
    "CompetitorSourceRegistry",
    "CompetitorProfiles",
    "CompetitorProfile",
]
