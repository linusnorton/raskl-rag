"""Batched embedding via Bedrock."""

from __future__ import annotations

from tqdm import tqdm

from .config import ChunkerConfig
from .providers import get_embed_provider
from .schema import Chunk


def embed_chunks(chunks: list[Chunk], config: ChunkerConfig) -> list[list[float]]:
    """Generate embeddings for a list of chunks.

    Returns a list of float vectors, one per chunk, in the same order.
    """
    provider = get_embed_provider(config)
    texts = [c.text for c in chunks]
    all_embeddings: list[list[float]] = []

    batch_size = config.embed_batch_size
    for batch_start in tqdm(range(0, len(texts), batch_size), desc="Embedding", unit="batch"):
        batch = texts[batch_start : batch_start + batch_size]
        batch_embeddings = provider.embed(batch)
        all_embeddings.extend(batch_embeddings)

    return all_embeddings
