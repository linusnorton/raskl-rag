"""Chunker/indexer configuration via environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class ChunkerConfig(BaseSettings):
    model_config = {"env_prefix": "CHUNKER_"}

    # Paths
    data_dir: Path = Path("data")

    # Restitching
    restitch_enabled: bool = True

    # Chunking
    max_chunk_tokens: int = 512
    min_chunk_tokens: int = 0

    # Embedding
    embed_batch_size: int = 32
    embed_dimensions: int = 1024
    embed_task_prefix: str = ""

    # Provider selection ("bedrock" or "model_studio")
    llm_provider: str = "bedrock"

    # Bedrock
    bedrock_region: str = "eu-west-2"
    bedrock_embed_model_id: str = "amazon.titan-embed-text-v2:0"

    # Model Studio (Alibaba Cloud)
    model_studio_api_key: str = ""
    model_studio_base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    model_studio_embed_model_id: str = "text-embedding-v4"

    # Database
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "raskl_rag"
    db_user: str = "raskl"
    db_password: str = "raskl"
    database_dsn: str = ""  # If set, overrides constructed DSN

    @property
    def dsn(self) -> str:
        if self.database_dsn:
            return self.database_dsn
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
