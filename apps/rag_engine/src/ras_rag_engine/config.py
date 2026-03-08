"""RAG engine configuration via environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class RAGConfig(BaseSettings):
    model_config = {"env_prefix": "CHAT_"}

    # Chat LLM
    llm_max_tokens: int = 4096
    llm_context_window: int = 40960
    llm_temperature: float = 0.5

    # Embedding
    embed_dimensions: int = 1024
    embed_task_prefix: str = ""

    # Reranker
    rerank_enabled: bool = True
    rerank_candidates: int = 30
    rerank_instruction: str = (
        "Given a user question about historical JMBRAS and Swettenham journal documents, "
        "judge whether the document passage is relevant"
    )

    # Bedrock
    bedrock_region: str = "eu-west-2"
    bedrock_chat_model_id: str = "qwen.qwen3-235b-a22b-2507-v1:0"
    bedrock_embed_model_id: str = "amazon.titan-embed-text-v2:0"
    bedrock_rerank_region: str = "eu-central-1"  # Cohere Rerank not available in all regions
    bedrock_rerank_model_id: str = "amazon.rerank-v1:0"

    # Extended thinking (0 = disabled)
    llm_thinking_budget: int = 2048

    # Retrieval
    retrieval_top_k: int = 15

    # Database
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "raskl_rag"
    db_user: str = "raskl"
    db_password: str = "raskl"
    database_dsn: str = ""  # If set, overrides constructed DSN

    # Web search
    web_search_enabled: bool = True

    # Audio (AWS Transcribe temp storage)
    transcribe_s3_bucket: str = ""  # S3 bucket for temporary transcription files
    transcribe_vocabulary_name: str = ""  # AWS Transcribe custom vocabulary name

    # Image serving
    s3_bucket: str = ""  # S3 bucket for image assets (Lambda mode)
    data_dir: str = "data/out"  # Local asset root (local mode)
    api_base_url: str = "http://localhost:8000"  # Base URL for image URLs in LLM context

    # API server
    api_port: int = 8000
    api_key: str = ""  # If set, requires Bearer token auth

    @property
    def dsn(self) -> str:
        if self.database_dsn:
            return self.database_dsn
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
