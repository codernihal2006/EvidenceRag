import json
from pathlib import Path
from .models.document import DocumentRecord
from .models.chunk import Chunk


class DocumentStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)
        self.index_file = path / "documents.json"
        self.documents: dict[str, DocumentRecord] = {}
        self.chunks: list[Chunk] = []
        self._load()

    def _load(self):
        if self.index_file.exists():
            data = json.loads(self.index_file.read_text())
            self.documents = {d["document_id"]: DocumentRecord(**d) for d in data.get("documents", [])}
            self.chunks = [Chunk(**c) for c in data.get("chunks", [])]

    def save(self):
        self.index_file.write_text(json.dumps({"documents": [d.model_dump() for d in self.documents.values()], "chunks": [c.model_dump() for c in self.chunks]}, indent=2))

    def add(self, record: DocumentRecord, chunks: list[Chunk]):
        self.documents[record.document_id] = record
        self.chunks.extend(chunks)
        self.save()

    def delete(self, document_id: str):
        record = self.documents.pop(document_id, None)
        self.chunks = [c for c in self.chunks if c.document_id != document_id]
        self.save()
        if record and record.path:
            Path(record.path).unlink(missing_ok=True)
        return record
