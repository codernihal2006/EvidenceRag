from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int
    text: str
    section: str = ""
    chunk_index: int


class SearchResult(BaseModel):
    chunk: Chunk
    score: float
    rank: int = 0
    source: str = ""


class Evidence(BaseModel):
    citation_id: str
    chunk: Chunk
