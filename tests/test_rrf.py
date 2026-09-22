from evidencerag.backend.models.chunk import Chunk, SearchResult
from evidencerag.backend.retrieval.rrf import rrf_fusion


def result(name, rank, source):
    c = Chunk(chunk_id=name, document_id="d", document_name="x.pdf", page_number=1, text=name, chunk_index=rank)
    return SearchResult(chunk=c, score=1, rank=rank, source=source)


def test_rrf_rewards_overlap():
    fused = rrf_fusion([result("A", 1, "dense"), result("B", 2, "dense"), result("C", 3, "dense")],
                       [result("B", 1, "bm25"), result("C", 2, "bm25"), result("D", 3, "bm25")], k=60)
    assert [r.chunk.chunk_id for r in fused][:2] == ["B", "C"]
