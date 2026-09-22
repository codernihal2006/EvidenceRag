from ..models.chunk import SearchResult


class Reranker:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = None
        self._attempted = False

    def _load(self) -> None:
        if self._attempted:
            return
        self._attempted = True
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
        except Exception:
            self.model = None

    def rerank(self, query: str, candidates: list[SearchResult], top_k: int = 5) -> list[SearchResult]:
        if not candidates:
            return []
        self._load()
        if self.model is None:
            return candidates[:top_k]
        scores = self.model.predict([(query, c.chunk.text) for c in candidates])
        ranked = sorted(zip(scores, candidates), key=lambda x: float(x[0]), reverse=True)
        return [SearchResult(chunk=c.chunk, score=float(score), rank=i, source="reranker") for i, (score, c) in enumerate(ranked[:top_k], 1)]
