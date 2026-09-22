from ..models.chunk import Chunk, SearchResult


class DenseRetriever:
    def __init__(self, persist_dir, model_name: str) -> None:
        self.persist_dir = str(persist_dir)
        self.model_name = model_name
        self.collection = None
        self._fallback = False

    def _ensure(self) -> None:
        if self.collection is not None or self._fallback:
            return
        try:
            import chromadb
            self.client = chromadb.PersistentClient(path=self.persist_dir)
            self.collection = self.client.get_or_create_collection("evidencerag_chunks")
        except Exception:
            self._fallback = True

    def upsert(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self._ensure()
        if self.collection is None:
            return
        try:
            from google import genai
            from ..config import settings
            client = genai.Client(api_key=settings.gemini_api_key)
            embeddings = [client.models.embed_content(model=self.model_name, contents=c.text).embeddings[0].values for c in chunks]
            self.collection.upsert(ids=[c.chunk_id for c in chunks], documents=[c.text for c in chunks], embeddings=embeddings,
                metadatas=[c.model_dump() | {"text": c.text} for c in chunks])
        except Exception:
            self._fallback = True

    def delete(self, document_id: str) -> None:
        self._ensure()
        if self.collection is not None:
            self.collection.delete(where={"document_id": document_id})

    def search(self, query: str, top_k: int = 15) -> list[SearchResult]:
        self._ensure()
        if self.collection is None:
            return []
        try:
            from google import genai
            from ..config import settings
            client = genai.Client(api_key=settings.gemini_api_key)
            vector = client.models.embed_content(model=self.model_name, contents=query).embeddings[0].values
            result = self.collection.query(query_embeddings=[vector], n_results=top_k, include=["documents", "metadatas", "distances"])
            output = []
            for rank, (meta, distance) in enumerate(zip(result["metadatas"][0], result["distances"][0]), 1):
                output.append(SearchResult(chunk=Chunk(**{k: meta[k] for k in Chunk.model_fields}), score=1 / (1 + distance), rank=rank, source="dense"))
            return output
        except Exception:
            return []
