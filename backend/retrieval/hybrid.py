import time
from ..models.chunk import SearchResult
from .bm25 import BM25Index
from .dense import DenseRetriever
from .rrf import rrf_fusion
from .reranker import Reranker


class HybridRetriever:
    def __init__(self, dense: DenseRetriever, bm25: BM25Index, reranker: Reranker) -> None:
        self.dense, self.bm25, self.reranker = dense, bm25, reranker
        self.chunks = []

    def rebuild(self, chunks) -> None:
        self.chunks = chunks
        self.bm25.rebuild(chunks)
        self.dense.upsert(chunks)

    def delete(self, document_id: str) -> None:
        self.chunks = [c for c in self.chunks if c.document_id != document_id]
        self.bm25.rebuild(self.chunks)
        self.dense.delete(document_id)

    def search(self, query: str, dense_k: int, bm25_k: int, rrf_k: int, rerank_k: int) -> tuple[list[SearchResult], list[SearchResult], list[SearchResult], list[SearchResult], dict[str, float]]:
        timings = {}
        start = time.perf_counter(); dense = self.dense.search(query, dense_k); timings["dense"] = (time.perf_counter() - start) * 1000
        start = time.perf_counter(); bm25 = self.bm25.search(query, bm25_k); timings["bm25"] = (time.perf_counter() - start) * 1000
        start = time.perf_counter(); fused = rrf_fusion(dense, bm25, rrf_k); timings["rrf"] = (time.perf_counter() - start) * 1000
        start = time.perf_counter(); reranked = self.reranker.rerank(query, fused[:max(rerank_k, 15)], rerank_k); timings["rerank"] = (time.perf_counter() - start) * 1000
        return dense, bm25, fused, reranked, timings
