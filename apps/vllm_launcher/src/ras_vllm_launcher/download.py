"""Download models via huggingface_hub."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download
from rich.console import Console

MODELS = {
    "embed": "Qwen/Qwen3-Embedding-8B",
    "rerank": "Qwen/Qwen3-Reranker-8B",
    "chat": "cyankiwi/Qwen3.5-27B-AWQ-4bit",
    "embed-light": "BAAI/bge-m3",
    "rerank-light": "BAAI/bge-reranker-v2-m3",
    "chat-light": "Qwen/Qwen3-8B-AWQ",
}

# Only download Q4_K_M quant + metadata for GGUF repos.
GGUF_ALLOW_PATTERNS = ["*Q4_K_M*", "*.json", "*.txt", "tokenizer*"]

console = Console()


def download_model(role: str, target_dir: Path) -> Path:
    """Download a single model to target_dir/{org}--{name}."""
    repo_id = MODELS[role]
    local_dir = target_dir / repo_id.replace("/", "--")
    console.print(f"[bold]Downloading {repo_id}[/bold] → {local_dir}")
    kwargs: dict = {"repo_id": repo_id, "local_dir": str(local_dir)}
    if "GGUF" in repo_id:
        kwargs["allow_patterns"] = GGUF_ALLOW_PATTERNS
    snapshot_download(**kwargs)
    console.print(f"[green]Done:[/green] {local_dir}")
    return local_dir


def download_all(target_dir: Path) -> None:
    """Download all standard (heavy) models."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for role in ("embed", "rerank", "chat"):
        download_model(role, target_dir)


def download_light(target_dir: Path) -> None:
    """Download all lightweight models (BGE-M3, BGE-Reranker, Qwen3.5-9B-GGUF)."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for role in ("embed-light", "rerank-light", "chat-light"):
        download_model(role, target_dir)
