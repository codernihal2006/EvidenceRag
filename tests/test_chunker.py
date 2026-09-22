from evidencerag.backend.ingestion.pdf_parser import ParsedDocument, ParsedPage
from evidencerag.backend.ingestion.chunker import chunk_document


def test_chunk_preserves_page_and_section():
    doc = ParsedDocument(name="paper.pdf", pages=[ParsedPage(page_number=7, title="Methods", text="Methods\n\nThis is useful evidence about enzyme immobilization." * 20)])
    chunks = chunk_document(doc, "doc1", chunk_size=30, overlap=5)
    assert chunks and all(c.page_number == 7 for c in chunks)
    assert chunks[0].document_name == "paper.pdf"
