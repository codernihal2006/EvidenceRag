import re
from .pdf_parser import ParsedDocument
from ..models.chunk import Chunk


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]


def chunk_document(document: ParsedDocument, document_id: str, chunk_size: int = 900, overlap: int = 120) -> list[Chunk]:
    chunks: list[Chunk] = []
    index = 0
    for page in document.pages:
        if not page.text:
            continue
        section = page.title
        paragraphs = _paragraphs(page.text)
        buffer: list[str] = []
        words = 0
        for paragraph in paragraphs:
            if len(paragraph.split()) < 16 and not paragraph.endswith("."):
                section = paragraph
            candidate = " ".join(buffer + [paragraph])
            if buffer and len(candidate.split()) > chunk_size:
                text = " ".join(buffer).strip()
                chunks.append(Chunk(chunk_id=f"{document_id}_{index}", document_id=document_id,
                    document_name=document.name, page_number=page.page_number, text=text,
                    section=section, chunk_index=index))
                index += 1
                tail = text.split()[-overlap:]
                buffer, words = [" ".join(tail)], len(tail)
            buffer.append(paragraph)
            words += len(paragraph.split())
        if buffer:
            chunks.append(Chunk(chunk_id=f"{document_id}_{index}", document_id=document_id,
                document_name=document.name, page_number=page.page_number, text=" ".join(buffer).strip(),
                section=section, chunk_index=index))
            index += 1
    return chunks
