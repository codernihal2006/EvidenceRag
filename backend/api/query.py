import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from ..config import settings
from ..generation.generator import generate_answer
from ..generation.citation_validator import validate_citations
from ..models.chunk import Evidence

router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    question: str = Field(min_length=2)
    dense_top_k: int | None = None
    bm25_top_k: int | None = None
    rrf_k: int | None = None
    reranker_top_k: int | None = None


@router.post("")
def query(request: QueryRequest):
    from ..main import retriever
    if not retriever.chunks:
        raise HTTPException(400, "Index a PDF before asking a question")
    start = time.perf_counter()
    dense, bm25, fused, reranked, timings = retriever.search(request.question, request.dense_top_k or settings.dense_top_k,
        request.bm25_top_k or settings.bm25_top_k, request.rrf_k or settings.rrf_k, request.reranker_top_k or settings.reranker_top_k)
    evidence = [Evidence(citation_id=str(i), chunk=r.chunk) for i, r in enumerate(reranked, 1)]
    generation_start = time.perf_counter(); answer = generate_answer(request.question, evidence); timings["generation"] = (time.perf_counter() - generation_start) * 1000
    validation_start = time.perf_counter(); validation = validate_citations(request.question, answer, evidence); timings["validation"] = (time.perf_counter() - validation_start) * 1000
    timings["total"] = (time.perf_counter() - start) * 1000
    from ..models.response import QueryResponse
    return QueryResponse(answer=answer, sources=evidence, dense_results=dense, bm25_results=bm25, fused_results=fused, reranked_results=reranked, validation=validation, timings_ms=timings)
