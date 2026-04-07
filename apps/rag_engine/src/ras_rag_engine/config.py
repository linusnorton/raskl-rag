"""RAG engine configuration via environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class RAGConfig(BaseSettings):
    model_config = {"env_prefix": "CHAT_"}

    # Chat LLM
    llm_max_tokens: int = 4096
    llm_context_window: int = 65536
    llm_temperature: float = 0.4

    # Embedding
    embed_dimensions: int = 1024
    embed_task_prefix: str = ""

    # Reranker
    rerank_enabled: bool = True
    rerank_candidates: int = 50
    rerank_relevance_score: float = 0.4
    rerank_instruction: str = (
        "Identify passages that contain specific names, dates, or events relevant to the query. Ensure that both detailed narratives and concise references are considered equally if they contain factual answers."
    )


    # Provider selection ("bedrock" or "model_studio")
    llm_provider: str = "bedrock"

    # Bedrock
    bedrock_region: str = "eu-west-2"
    bedrock_chat_model_id: str = "qwen.qwen3-235b-a22b-2507-v1:0"
    bedrock_embed_region: str = "eu-west-1"  # Cohere Embed v4 not available in eu-west-2
    bedrock_embed_model_id: str = "eu.cohere.embed-v4:0"
    bedrock_rerank_region: str = "eu-central-1"  # Cohere Rerank not available in all regions
    bedrock_rerank_model_id: str = "cohere.rerank-v3-5:0"

    # Model Studio (Alibaba Cloud)
    model_studio_api_key: str = ""
    model_studio_base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    model_studio_chat_model_id: str = "qwen3.5-397b-a17b"
    model_studio_embed_model_id: str = "text-embedding-v4"
    model_studio_rerank_model_id: str = "qwen3-rerank"

    # Extended thinking (0 = disabled)
    llm_thinking_budget: int = 2048

    # Retrieval
    retrieval_top_k: int = 20

    # Diversity — cap chunks per document in rerank candidates (0 = disabled)
    diversity_max_per_doc: int = 5

    # Database
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "raskl_rag"
    db_user: str = "raskl"
    db_password: str = "raskl"
    database_dsn: str = ""  # If set, overrides constructed DSN

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
