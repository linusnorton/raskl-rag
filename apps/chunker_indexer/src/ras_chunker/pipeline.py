"""Orchestrate the load → restitch → chunk → embed → index pipeline."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from .chunker import chunk_blocks
from .config import ChunkerConfig
from .db import get_connection, upsert_chunks, upsert_document, upsert_figures
from .embedder import embed_chunks
from .loader import DocprocOutput, find_doc_dir
from .restitch import restitch
from .schema import Chunk, FigureMeta

console = Console()


def load_and_chunk(doc_id: str, config: ChunkerConfig) -> tuple[DocprocOutput, list[Chunk]]:
    """Load docproc output, restitch, and chunk. No DB or embedding needed."""
    doc_dir = find_doc_dir(config.data_dir, doc_id)
    output = DocprocOutput(doc_dir)
    console.print(f"Loaded [bold]{output.meta.source_filename}[/bold]: {len(output.blocks)} blocks")

    if config.restitch_enabled:
        blocks = restitch(output)
        console.print(f"Restitched: {len(blocks)} blocks (from {len(output.blocks)})")
    else:
        # Convert to stitched blocks without merging
        from .restitch import _best_text, _collect_footnote_refs
        from .schema import StitchedBlock

        blocks = [
            StitchedBlock(
                block_ids=[b.block_id],
                doc_id=b.doc_id,
                start_page=b.page_num_1,
                end_page=b.page_num_1,
                text=_best_text(b),
                block_type=b.block_type,
                section_path=b.section_path,
                lang=b.lang,
                reading_order=b.reading_order,
                footnote_refs=_collect_footnote_refs(b.block_id, output),
            )
            for b in sorted(output.blocks, key=lambda b: (b.page_num_1, b.reading_order))
        ]

    chunks = chunk_blocks(blocks, output, config)
    console.print(f"Chunked: {len(chunks)} chunks")
    return output, chunks


def _build_figures(output: DocprocOutput) -> list[FigureMeta]:
    """Build FigureMeta list from docproc output, computing relative asset paths."""
    figures = []
    doc_dir_str = str(output.doc_dir) + "/"
    for fig in output.figures:
        if not fig.asset_jpg_path:
            continue
        # Compute relative path: strip the doc_dir prefix
        asset_path = fig.asset_jpg_path.replace(doc_dir_str, "") if fig.asset_jpg_path else ""
        thumb_path = fig.asset_thumb_path.replace(doc_dir_str, "") if fig.asset_thumb_path else ""
        figures.append(
            FigureMeta(
                figure_id=fig.figure_id,
                doc_id=fig.doc_id,
                page_num=fig.page_num_1,
                caption=fig.caption_text_clean,
                asset_path=asset_path,
                thumb_path=thumb_path,
            )
        )
    return figures


def _index_figures(figures: list[FigureMeta], config: ChunkerConfig) -> list[list[float] | None]:
    """Embed figure captions and return embeddings (None for empty captions)."""
    from .providers import get_embed_provider

    provider = get_embed_provider(config)
    embeddings: list[list[float] | None] = []

    # Collect figures with non-empty captions for batch embedding
    captioned = [(i, fig) for i, fig in enumerate(figures) if fig.caption.strip()]
    if captioned:
        texts = [fig.caption for _, fig in captioned]
        vecs = provider.embed(texts)
        vec_map = {i: vec for (i, _), vec in zip(captioned, vecs)}
    else:
        vec_map = {}

    for i in range(len(figures)):
        embeddings.append(vec_map.get(i))

    return embeddings


def run_index(doc_id: str, config: ChunkerConfig, *, s3_prefix: str = "", version: int = 0) -> None:
    """Full pipeline: load → restitch → chunk → embed → index."""
    output, chunks = load_and_chunk(doc_id, config)

    if not chunks:
        console.print("[yellow]No chunks to index.[/yellow]")
        return

    # Store S3 prefix in metadata
    if s3_prefix:
        output.meta.s3_prefix = s3_prefix

    # Embed
    console.print(f"Embedding {len(chunks)} chunks via Bedrock ({config.bedrock_embed_model_id}) ...")
    embeddings = embed_chunks(chunks, config)
    console.print(f"[green]Embedded {len(embeddings)} chunks[/green]")

    # Build and embed figures
    figures = _build_figures(output)
    fig_embeddings: list[list[float] | None] = []
    if figures:
        console.print(f"Embedding {len(figures)} figures ...")
        fig_embeddings = _index_figures(figures, config)
        console.print(f"[green]Embedded {len(figures)} figures[/green]")

    # Index
    console.print("Writing to database ...")
    with get_connection(config) as conn:
        upsert_document(conn, output.meta)
        upsert_chunks(conn, chunks, embeddings, doc_id=doc_id, version=version)
        upsert_figures(conn, figures, fig_embeddings)
        conn.commit()
    console.print(f"[green]Indexed {len(chunks)} chunks + {len(figures)} figures for {doc_id}[/green]")


def run_index_all(config: ChunkerConfig) -> None:
    """Index all documents found in data/out/."""
    out_dir = config.data_dir / "out"
    if not out_dir.is_dir():
        console.print(f"[red]No output directory: {out_dir}[/red]")
        return

    doc_ids = sorted(d.name for d in out_dir.iterdir() if d.is_dir() and (d / "documents.jsonl").exists())
    console.print(f"Found {len(doc_ids)} documents to index")

    for doc_id in doc_ids:
        console.print(f"\n[bold]--- {doc_id} ---[/bold]")
        run_index(doc_id, config)
