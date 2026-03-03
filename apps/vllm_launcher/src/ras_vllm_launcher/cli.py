"""CLI for the vLLM launcher skeleton app."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import click
import httpx
from rich.console import Console

console = Console()
PID_DIR = Path("data/pids")
LOG_DIR = Path("data/logs")

DEFAULT_OCR_MODEL = "deepseek-ai/DeepSeek-OCR"
DEFAULT_OCR_PORT = 8000


def _find_uv() -> str:
    """Find the uv binary."""
    import shutil

    path = shutil.which("uv")
    if path:
        return path
    # Common install location
    candidate = Path.home() / ".local" / "bin" / "uv"
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError("uv not found on PATH or in ~/.local/bin")


def _start_server(role: str, model: str, port: int, extra_args: list[str] | None = None) -> int:
    """Start a vLLM server subprocess with logging. Returns PID.

    Uses ``uv run --package ras-vllm-launcher --extra gpu`` so that vllm
    is available without polluting the main venv.
    """
    PID_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_file = LOG_DIR / f"{role}.log"
    uv = _find_uv()
    cmd = [
        uv, "run", "--package", "ras-vllm-launcher", "--extra", "gpu",
        "vllm", "serve", model, "--port", str(port),
    ]
    if extra_args:
        cmd.extend(extra_args)

    console.print(f"[bold green]Starting {role}:[/] {model} on port {port}")
    console.print(f"  Command: {' '.join(cmd)}")
    console.print(f"  Log: {log_file}")

    log_fh = open(log_file, "w")
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)

    pid_file = PID_DIR / f"{role}.json"
    pid_file.write_text(json.dumps({"pid": proc.pid, "model": model, "port": port}))
    console.print(f"  PID {proc.pid} → {pid_file}")
    return proc.pid


@click.group()
def cli() -> None:
    """ras-vllm-launcher: manage vLLM model servers."""


@cli.command()
@click.option("--model", default=DEFAULT_OCR_MODEL, show_default=True, help="DeepSeek-OCR model name")
@click.option("--port", default=DEFAULT_OCR_PORT, show_default=True, type=int, help="Server port")
@click.option("--gpu-memory-utilization", default=0.85, show_default=True, type=float, help="GPU memory fraction")
@click.option("--max-model-len", default=8192, show_default=True, type=int, help="Max model context length")
@click.option("--wait/--no-wait", default=True, help="Wait for server to be ready before returning")
def ocr(model: str, port: int, gpu_memory_utilization: float, max_model_len: int, wait: bool) -> None:
    """Start the DeepSeek-OCR server for document extraction."""
    extra_args = [
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
    ]
    pid = _start_server("ocr", model, port, extra_args)

    if wait:
        url = f"http://localhost:{port}/v1/models"
        console.print(f"\n[bold]Waiting for server to be ready...[/]")
        for i in range(120):  # up to 2 minutes
            try:
                resp = httpx.get(url, timeout=2)
                if resp.status_code == 200:
                    console.print(f"[bold green]Server ready![/] http://localhost:{port}")
                    return
            except (httpx.ConnectError, httpx.TimeoutException):
                pass
            time.sleep(1)
            if (i + 1) % 10 == 0:
                console.print(f"  Still waiting... ({i + 1}s)")

        console.print(f"[yellow]Server not ready after 120s. Check log: data/logs/ocr.log[/]")


DEFAULT_EMBED_MODEL = "./models/Qwen--Qwen3-Embedding-8B"
DEFAULT_EMBED_PORT = 8001


@cli.command()
@click.option("--model", default=DEFAULT_EMBED_MODEL, show_default=True, help="Embedding model path or name")
@click.option("--port", default=DEFAULT_EMBED_PORT, show_default=True, type=int, help="Server port")
@click.option("--gpu-memory-utilization", default=0.85, show_default=True, type=float, help="GPU memory fraction")
@click.option("--max-model-len", default=8192, show_default=True, type=int, help="Max model context length")
@click.option("--wait/--no-wait", default=True, help="Wait for server to be ready before returning")
def embed(model: str, port: int, gpu_memory_utilization: float, max_model_len: int, wait: bool) -> None:
    """Start the embedding server (Qwen3-Embedding)."""
    extra_args = [
        "--runner", "pooling",
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
    ]
    pid = _start_server("embed", model, port, extra_args)

    if wait:
        url = f"http://localhost:{port}/v1/models"
        console.print(f"\n[bold]Waiting for server to be ready...[/]")
        for i in range(120):
            try:
                resp = httpx.get(url, timeout=2)
                if resp.status_code == 200:
                    console.print(f"[bold green]Server ready![/] http://localhost:{port}")
                    return
            except (httpx.ConnectError, httpx.TimeoutException):
                pass
            time.sleep(1)
            if (i + 1) % 10 == 0:
                console.print(f"  Still waiting... ({i + 1}s)")

        console.print(f"[yellow]Server not ready after 120s. Check log: data/logs/embed.log[/]")


DEFAULT_CHAT_MODEL = "Qwen/Qwen3-30B-A3B-GPTQ-Int4"
DEFAULT_CHAT_PORT = 8002


@cli.command()
@click.option("--model", default=DEFAULT_CHAT_MODEL, show_default=True, help="Chat model name or path")
@click.option("--tokenizer", default=None, help="Tokenizer name or path (required for GGUF models)")
@click.option("--port", default=DEFAULT_CHAT_PORT, show_default=True, type=int, help="Server port")
@click.option("--gpu-memory-utilization", default=0.85, show_default=True, type=float, help="GPU memory fraction")
@click.option("--max-model-len", default=40960, show_default=True, type=int, help="Max model context length")
@click.option("--wait/--no-wait", default=True, help="Wait for server to be ready before returning")
def chat(model: str, tokenizer: str | None, port: int, gpu_memory_utilization: float, max_model_len: int, wait: bool) -> None:
    """Start the chat LLM server with tool calling."""
    extra_args = [
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_coder",
        "--reasoning-parser", "qwen3",
    ]
    if tokenizer:
        extra_args.extend(["--tokenizer", tokenizer])
    pid = _start_server("chat", model, port, extra_args)

    if wait:
        url = f"http://localhost:{port}/v1/models"
        console.print(f"\n[bold]Waiting for server to be ready...[/]")
        for i in range(180):  # up to 3 minutes for larger model
            try:
                resp = httpx.get(url, timeout=2)
                if resp.status_code == 200:
                    console.print(f"[bold green]Server ready![/] http://localhost:{port}")
                    return
            except (httpx.ConnectError, httpx.TimeoutException):
                pass
            time.sleep(1)
            if (i + 1) % 10 == 0:
                console.print(f"  Still waiting... ({i + 1}s)")

        console.print(f"[yellow]Server not ready after 180s. Check log: data/logs/chat.log[/]")


@cli.command()
@click.option("--chat-model", required=True, help="Chat model name")
@click.option("--embed-model", default=None, help="Embedding model name")
@click.option("--rerank-model", default=None, help="Reranking model name")
@click.option("--ports", default="8000,8001,8002", help="Comma-separated ports")
def up(chat_model: str, embed_model: str | None, rerank_model: str | None, ports: str) -> None:
    """Start all vLLM serve subprocesses (chat + optional embed/rerank)."""
    port_list = [int(p.strip()) for p in ports.split(",")]

    models = [("chat", chat_model, port_list[0])]
    if embed_model and len(port_list) > 1:
        models.append(("embed", embed_model, port_list[1]))
    if rerank_model and len(port_list) > 2:
        models.append(("rerank", rerank_model, port_list[2]))

    for role, model, port in models:
        _start_server(role, model, port)


@cli.command()
def health() -> None:
    """Check health of running vLLM servers."""
    if not PID_DIR.exists():
        console.print("[yellow]No pid directory found.[/]")
        return
    for pid_file in PID_DIR.glob("*.json"):
        info = json.loads(pid_file.read_text())
        port = info["port"]
        role = pid_file.stem
        try:
            resp = httpx.get(f"http://localhost:{port}/v1/models", timeout=5)
            console.print(f"[green]✓[/] {role} (port {port}): {resp.status_code}")
        except httpx.ConnectError:
            console.print(f"[red]✗[/] {role} (port {port}): connection refused")


@cli.command()
def down() -> None:
    """Stop running vLLM servers via pidfiles."""
    if not PID_DIR.exists():
        console.print("[yellow]No pid directory found.[/]")
        return
    for pid_file in PID_DIR.glob("*.json"):
        info = json.loads(pid_file.read_text())
        pid = info["pid"]
        role = pid_file.stem
        try:
            os.kill(pid, signal.SIGTERM)
            console.print(f"[green]Stopped[/] {role} (PID {pid})")
        except ProcessLookupError:
            console.print(f"[yellow]Already stopped[/] {role} (PID {pid})")
        pid_file.unlink()


@cli.command()
@click.option(
    "--role",
    type=click.Choice(["embed", "rerank", "chat", "all", "embed-light", "rerank-light", "chat-light", "all-light"]),
    default="all",
    help="Which model(s) to download",
)
@click.option("--target-dir", default="models", type=click.Path(path_type=Path), help="Directory to download into")
def download(role: str, target_dir: Path) -> None:
    """Download embedding/reranker models from HuggingFace."""
    from .download import download_all, download_light, download_model

    if role == "all":
        download_all(target_dir)
    elif role == "all-light":
        download_light(target_dir)
    else:
        download_model(role, target_dir)


if __name__ == "__main__":
    cli()
