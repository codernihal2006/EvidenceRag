from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    embedding_model: str = "models/text-embedding-004"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    data_dir: Path = Path("data")
    chunk_size: int = 900
    chunk_overlap: int = 120
    dense_top_k: int = 15
    bm25_top_k: int = 15
    rrf_k: int = 60
    reranker_top_k: int = 5
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"


settings = Settings()
settings.uploads_dir.mkdir(parents=True, exist_ok=True)
settings.chroma_dir.mkdir(parents=True, exist_ok=True)
