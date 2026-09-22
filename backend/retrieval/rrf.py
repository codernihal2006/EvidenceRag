from ..models.chunk import SearchResult


def rrf_fusion(dense_results: list[SearchResult], bm25_results: list[SearchResult], k: int = 60) -> list[SearchResult]:
    """Fuse ranked lists without comparing incomparable dense/BM25 scores."""
    by_id: dict[str, SearchResult] = {}
    scores: dict[str, float] = {}
    for results in (dense_results, bm25_results):
        for rank, result in enumerate(results, 1):
            by_id[result.chunk.chunk_id] = result
            scores[result.chunk.chunk_id] = scores.get(result.chunk.chunk_id, 0) + 1 / (k + rank)
    ranked = sorted(by_id, key=lambda cid: scores[cid], reverse=True)
    return [SearchResult(chunk=by_id[cid].chunk, score=scores[cid], rank=i, source="rrf") for i, cid in enumerate(ranked, 1)]
