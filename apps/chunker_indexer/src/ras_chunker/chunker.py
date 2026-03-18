"""Heading-based semantic chunking with footnote inlining.

Algorithm:
1. Walk stitched blocks in order; each ``heading`` block starts a new chunk.
2. Accumulate paragraph / list_item blocks until ``max_chunk_tokens`` exceeded.
3. Inline footnotes at chunk bottom via footnote_refs → footnotes lookup.
4. Merge tiny trailing chunks (< ``min_chunk_tokens``) into previous chunk.
5. Generate deterministic chunk_id from doc_id + chunk_index.
"""

from __future__ import annotations

import hashlib

from .config import ChunkerConfig
from .loader import DocprocOutput
from .schema import Chunk, StitchedBlock

# Rough token estimate: 1 token ≈ 4 characters (for English text).
_CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _make_chunk_id(doc_id: str, chunk_index: int) -> str:
    raw = f"{doc_id}:chunk:{chunk_index}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _build_footnote_map(output: DocprocOutput) -> dict[str, tuple[int, str, str]]:
    """Map footnote_id → (footnote_number, cleaned text, footnote_type)."""
    result: dict[str, tuple[int, str, str]] = {}
    for fn in output.footnotes:
        text = fn.text_clean if fn.text_clean else fn.text_raw
        fn_type = getattr(fn, "footnote_type", "explanatory")
        result[fn.footnote_id] = (fn.footnote_number, text.strip(), fn_type)
    return result


def _collect_overlap_blocks(chunk: _RawChunk, overlap_tokens: int) -> list[StitchedBlock]:
    """Walk blocks backwards to collect up to overlap_tokens worth of blocks."""
    if overlap_tokens <= 0 or not chunk.blocks:
        return []
    collected: list[StitchedBlock] = []
    tokens = 0
    for block in reversed(chunk.blocks):
        bt = _estimate_tokens(block.text)
        if tokens + bt > overlap_tokens and collected:
            break
        collected.append(block)
        tokens += bt
    collected.reverse()
    return collected


def _format_chunk_text(heading: str | None, body_parts: list[str], footnote_texts: list[str]) -> str:
    """Assemble final chunk text from heading, body, and footnotes."""
    parts: list[str] = []
    if heading:
        parts.append(heading)
        parts.append("")  # blank line after heading
    parts.extend(body_parts)

    if footnote_texts:
        parts.append("")
        parts.append("---")
        parts.append("Footnotes:")
        parts.extend(footnote_texts)

    return "\n".join(parts).strip()


def chunk_blocks(
    blocks: list[StitchedBlock],
    output: DocprocOutput,
    config: ChunkerConfig,
) -> list[Chunk]:
    """Split stitched blocks into semantic chunks."""
    footnote_map = _build_footnote_map(output)

    # Accumulate raw chunks (before merging tiny ones)
    raw_chunks: list[_RawChunk] = []
    current = _RawChunk()

    for block in blocks:
        # Skip boilerplate types
        if block.block_type in ("header", "footer", "page_number"):
            continue

        # Headings start a new chunk (flush the current one)
        if block.block_type == "heading":
            if current.has_content():
                raw_chunks.append(current)
            current = _RawChunk(heading=block.text.strip(), heading_block=block)
            continue

        # Footnote blocks are handled via inline refs, skip standalone
        if block.block_type == "footnote":
            continue

        # Check if adding this block would exceed max tokens
        block_tokens = _estimate_tokens(block.text)
        if current.has_content() and current.tokens + block_tokens > config.max_chunk_tokens:
            overlap = _collect_overlap_blocks(current, config.overlap_tokens)
            raw_chunks.append(current)
            current = _RawChunk(heading=current.heading, overlap_blocks=overlap)

        current.add_block(block)

    # Flush last chunk
    if current.has_content():
        raw_chunks.append(current)

    # Merge tiny trailing chunks into previous
    merged = _merge_small_chunks(raw_chunks, config.min_chunk_tokens)

    # Build final Chunk objects
    doc_id = output.doc_id
    chunks: list[Chunk] = []
    for i, raw in enumerate(merged):
        # Collect footnote texts for blocks in this chunk
        all_refs = sorted({ref for b in raw.blocks for ref in b.footnote_refs})
        footnote_texts = []
        for fid in all_refs:
            if fid not in footnote_map:
                continue
            fn_num, fn_text, fn_type = footnote_map[fid]
            if fn_type in ("citation", "mixed"):
                footnote_texts.append(f"[{fn_num}] [cites:] {fn_text}")
            else:
                footnote_texts.append(f"[{fn_num}] {fn_text}")

        body_parts = [b.text for b in raw.overlap_blocks] + [b.text for b in raw.blocks]
        text = _format_chunk_text(raw.heading, body_parts, footnote_texts)
        token_count = _estimate_tokens(text)
        block_ids = [bid for b in raw.blocks for bid in b.block_ids]
        if raw.heading_block:
            block_ids = list(raw.heading_block.block_ids) + block_ids

        chunks.append(
            Chunk(
                chunk_id=_make_chunk_id(doc_id, i),
                doc_id=doc_id,
                chunk_index=i,
                start_page=raw.start_page,
                end_page=raw.end_page,
                section_heading=raw.heading,
                text=text,
                block_ids=block_ids,
                token_count=token_count,
            )
        )

    return chunks


class _RawChunk:
    """Mutable accumulator for building a chunk."""

    def __init__(
        self,
        heading: str | None = None,
        heading_block: StitchedBlock | None = None,
        overlap_blocks: list[StitchedBlock] | None = None,
    ) -> None:
        self.heading = heading
        self.heading_block = heading_block
        self.blocks: list[StitchedBlock] = []
        self.overlap_blocks: list[StitchedBlock] = overlap_blocks or []
        self.tokens: int = 0

    def add_block(self, block: StitchedBlock) -> None:
        self.blocks.append(block)
        self.tokens += _estimate_tokens(block.text)

    def has_content(self) -> bool:
        return bool(self.blocks) or self.heading is not None

    @property
    def start_page(self) -> int:
        pages = [b.start_page for b in self.blocks]
        pages.extend(b.start_page for b in self.overlap_blocks)
        if self.heading_block:
            pages.append(self.heading_block.start_page)
        return min(pages) if pages else 1

    @property
    def end_page(self) -> int:
        pages = [b.end_page for b in self.blocks]
        pages.extend(b.end_page for b in self.overlap_blocks)
        if self.heading_block:
            pages.append(self.heading_block.end_page)
        return max(pages) if pages else 1


def _merge_small_chunks(raw_chunks: list[_RawChunk], min_tokens: int) -> list[_RawChunk]:
    """Merge chunks smaller than min_tokens into the previous chunk."""
    if not raw_chunks:
        return []

    merged: list[_RawChunk] = [raw_chunks[0]]
    for chunk in raw_chunks[1:]:
        if chunk.tokens < min_tokens and merged:
            prev = merged[-1]
            prev.blocks.extend(chunk.blocks)
            prev.tokens += chunk.tokens
            # Keep heading from the new chunk if prev didn't have one
            if chunk.heading and not prev.heading:
                prev.heading = chunk.heading
                prev.heading_block = chunk.heading_block
        else:
            merged.append(chunk)

    return merged
