"""Admin app configuration via environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class AdminConfig(BaseSettings):
    model_config = {"env_prefix": "ADMIN_"}

    secret_key: str = "change-me-in-production"
    open_webui_url: str = "http://localhost:3000"
    database_dsn: str = "postgresql://raskl:raskl@localhost:5432/raskl_rag"
    s3_bucket: str = ""
    session_expiry_hours: int = 24
    port: int = 8001
