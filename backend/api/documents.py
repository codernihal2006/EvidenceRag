import shutil
from pathlib import Path
from uuid import uuid4
from fastapi import APIRouter, File, UploadFile, HTTPException
from ..config import settings
from ..ingestion.pdf_parser import parse_pdf
from ..ingestion.chunker import chunk_document
from ..models.document import DocumentRecord

router = APIRouter(prefix="/documents", tags=["documents"])


def get_runtime():
    from ..main import store, retriever
    return store, retriever


@router.get("")
def list_documents():
    store, _ = get_runtime()
    return list(store.documents.values())


@router.post("/upload")
def upload_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a PDF file")
    store, retriever = get_runtime()
    if any(d.document_name == file.filename for d in store.documents.values()):
        raise HTTPException(409, "That document is already indexed")
    doc_id = uuid4().hex
    path = settings.uploads_dir / f"{doc_id}.pdf"
    with path.open("wb") as target:
        shutil.copyfileobj(file.file, target)
    try:
        parsed = parse_pdf(path)
        chunks = chunk_document(parsed, doc_id, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            raise ValueError("No text chunks were extracted")
        record = DocumentRecord(document_id=doc_id, document_name=file.filename, pages=len(parsed.pages), chunks=len(chunks), path=str(path))
        store.add(record, chunks)
        retriever.rebuild(store.chunks)
        return record
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(422, str(exc)) from exc


@router.delete("/{document_id}")
def delete_document(document_id: str):
    store, retriever = get_runtime()
    record = store.delete(document_id)
    if not record:
        raise HTTPException(404, "Document not found")
    retriever.delete(document_id)
    return {"deleted": document_id}
