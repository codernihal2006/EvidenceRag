from pydantic import BaseModel, Field
from .chunk import SearchResult, Evidence


class CitationCheck(BaseModel):
    citation: str
    claim: str
    supported: bool


class CitationValidation(BaseModel):
    supported: bool
    grounding_score: float = Field(ge=0, le=1)
    citations: list[CitationCheck] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    answer: str
    sources: list[Evidence]
    dense_results: list[SearchResult]
    bm25_results: list[SearchResult]
    fused_results: list[SearchResult]
    reranked_results: list[SearchResult]
    validation: CitationValidation
    timings_ms: dict[str, float]
