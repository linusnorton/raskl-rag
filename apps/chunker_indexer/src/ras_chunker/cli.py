"""CLI for the chunker/indexer."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .config import ChunkerConfig

console = Console()


@click.group()
def cli() -> None:
    """ras-chunker: chunk and index text blocks for RAG."""
    from dotenv import load_dotenv

    load_dotenv()


@cli.command()
@click.option("--doc-id", required=True, help="Document ID to index")
@click.option("--data-dir", default="data", type=click.Path(path_type=Path), help="Data directory")
def index(doc_id: str, data_dir: Path) -> None:
    """Embed and index a document into PostgreSQL/pgvector."""
    from .pipeline import run_index

    config = ChunkerConfig(data_dir=data_dir)
    run_index(doc_id, config)


@cli.command()
@click.option("--data-dir", default="data", type=click.Path(path_type=Path), help="Data directory")
def index_all(data_dir: Path) -> None:
    """Index all processed documents."""
    from .pipeline import run_index_all

    config = ChunkerConfig(data_dir=data_dir)
    run_index_all(config)


@cli.command()
@click.option("--doc-id", required=True, help="Document ID to preview")
@click.option("--data-dir", default="data", type=click.Path(path_type=Path), help="Data directory")
def plan(doc_id: str, data_dir: Path) -> None:
    """Dry-run: show chunk plan without embedding or indexing."""
    from .pipeline import load_and_chunk

    config = ChunkerConfig(data_dir=data_dir)
    output, chunks = load_and_chunk(doc_id, config)

    table = Table(title=f"Chunk Plan: {output.meta.source_filename}")
    table.add_column("#", style="dim", width=4)
    table.add_column("Pages", width=8)
    table.add_column("Heading", max_width=40)
    table.add_column("Tokens", justify="right", width=7)
    table.add_column("Blocks", justify="right", width=7)
    table.add_column("Preview", max_width=60)

    total_tokens = 0
    for chunk in chunks:
        page_range = f"{chunk.start_page}-{chunk.end_page}" if chunk.start_page != chunk.end_page else str(chunk.start_page)
        preview = chunk.text[:80].replace("\n", " ") + ("..." if len(chunk.text) > 80 else "")
        table.add_row(
            str(chunk.chunk_index),
            page_range,
            chunk.section_heading or "",
            str(chunk.token_count),
            str(len(chunk.block_ids)),
            preview,
        )
        total_tokens += chunk.token_count

    console.print(table)
    console.print(f"\nTotal: [bold]{len(chunks)}[/bold] chunks, [bold]{total_tokens}[/bold] tokens")


@cli.command("init-db")
def init_db() -> None:
    """Create database tables and indexes."""
    from .db import init_schema

    config = ChunkerConfig()
    console.print("Initializing database schema ...")
    init_schema(config)
    console.print("[green]Database schema created.[/green]")


if __name__ == "__main__":
    cli()
