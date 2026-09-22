from pydantic import BaseModel


class DocumentRecord(BaseModel):
    document_id: str
    document_name: str
    pages: int
    chunks: int
    status: str = "Indexed"
    path: str = ""
