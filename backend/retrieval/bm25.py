import math
from collections import Counter
from ..models.chunk import Chunk, SearchResult
from ..utils.text import tokenize


class BM25Index:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.tokens: list[list[str]] = []
        self.idf: dict[str, float] = {}
        self.avgdl = 0.0

    def rebuild(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.tokens = [tokenize(c.text) for c in chunks]
        self.avgdl = sum(map(len, self.tokens)) / max(1, len(self.tokens))
        document_frequency = Counter(token for terms in self.tokens for token in set(terms))
        self.idf = {term: math.log(1 + (len(chunks) - freq + 0.5) / (freq + 0.5)) for term, freq in document_frequency.items()}

    def search(self, query: str, top_k: int = 15) -> list[SearchResult]:
        q = tokenize(query)
        scored = []
        for i, terms in enumerate(self.tokens):
            counts = Counter(terms)
            score = 0.0
            for term in q:
                if term not in counts:
                    continue
                tf = counts[term]
                score += self.idf.get(term, 0) * (tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * len(terms) / max(1, self.avgdl)))
            scored.append((score, self.chunks[i]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [SearchResult(chunk=c, score=s, rank=n, source="bm25") for n, (s, c) in enumerate(scored[:top_k], 1) if s > 0]
