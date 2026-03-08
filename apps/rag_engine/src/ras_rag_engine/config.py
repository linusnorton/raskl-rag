"""RAG engine configuration via environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class RAGConfig(BaseSettings):
    model_config = {"env_prefix": "CHAT_"}

    # Provider selection
    llm_provider: str = "vllm"  # "vllm" or "bedrock"
    embed_provider: str = "sentence-transformers"  # "sentence-transformers" or "bedrock"
    rerank_provider: str = "qwen3"  # "qwen3", "cross-encoder", or "bedrock"

    # Chat LLM (vLLM backend)
    llm_base_url: str = "http://localhost:8002/v1"
    llm_model: str = "Qwen/Qwen3-30B-A3B-GPTQ-Int4"
    llm_max_tokens: int = 4096
    llm_context_window: int = 40960
    llm_temperature: float = 0.5

    # Embedding (sentence-transformers backend)
    embed_model: str = "./models/Qwen--Qwen3-Embedding-8B"
    embed_dimensions: int = 1024
    embed_task_prefix: str = "Represent this query for retrieving relevant passages: "
    embed_device: str = "cpu"  # CHAT_EMBED_DEVICE — "cpu" or "cuda"

    # Reranker (local backends)
    rerank_model: str = "./models/Qwen--Qwen3-Reranker-8B"
    rerank_enabled: bool = True
    rerank_candidates: int = 30
    rerank_backend: str = "qwen3"  # CHAT_RERANK_BACKEND — "qwen3" or "cross-encoder"
    rerank_device: str = "cpu"  # CHAT_RERANK_DEVICE — "cpu" or "cuda"
    rerank_instruction: str = (
        "Given a user question about historical JMBRAS and Swettenham journal documents, "
        "judge whether the document passage is relevant"
    )

    # Bedrock configuration (used when provider = "bedrock")
    bedrock_region: str = "eu-west-2"
    bedrock_chat_model_id: str = "qwen.qwen3-235b-a22b-2507-v1:0"
    bedrock_embed_model_id: str = "amazon.titan-embed-text-v2:0"
    bedrock_rerank_region: str = "eu-central-1"  # Cohere Rerank not available in all regions
    bedrock_rerank_model_id: str = "amazon.rerank-v1:0"

    # Extended thinking (Bedrock only, 0 = disabled)
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

    # API server
    api_port: int = 8000
    api_key: str = ""  # If set, requires Bearer token auth

    @property
    def dsn(self) -> str:
        if self.database_dsn:
            return self.database_dsn
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
