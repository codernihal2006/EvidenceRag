from evidencerag.backend.generation.citation_validator import validate_citations
from evidencerag.backend.models.chunk import Chunk, Evidence


def test_citation_validator_rejects_unknown_ids():
    evidence = [Evidence(citation_id="1", chunk=Chunk(chunk_id="c", document_id="d", document_name="a.pdf", page_number=2, text="Evidence", chunk_index=0))]
    check = validate_citations("q", "The answer is true [9].", evidence)
    assert not check.supported
    assert check.unsupported_claims
