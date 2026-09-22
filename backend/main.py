from fastapi import FastAPI
from .config import settings
from .store import DocumentStore
from .retrieval.bm25 import BM25Index
from .retrieval.dense import DenseRetriever
from .retrieval.reranker import Reranker
from .retrieval.hybrid import HybridRetriever
from .api import health, documents, query
from .utils.logging import configure_logging

configure_logging()
store = DocumentStore(settings.data_dir)
retriever = HybridRetriever(DenseRetriever(settings.chroma_dir, settings.embedding_model), BM25Index(), Reranker(settings.reranker_model))
retriever.rebuild(store.chunks)
app = FastAPI(title="EvidenceRAG", description="Hybrid RAG with verified citations")
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(query.router)
