"""Cross-page paragraph restitching.

When a paragraph is split across a page boundary, the docproc pipeline emits
two separate TextBlockRecord entries.  This module merges them back together
using simple heuristics:

1. The previous block is a ``paragraph`` that does NOT end with sentence-ending
   punctuation (``.``, ``!``, ``?``, ``"``, ``'``, ``)``).
2. The next block is a ``paragraph`` on the very next page whose cleaned text
   starts with a lowercase letter.

Merges are applied in-place so chains (page 4 → 5 → 6) work naturally.
"""

from __future__ import annotations

import re

from .loader import DocprocOutput, _TextBlock
from .schema import StitchedBlock

_SENTENCE_END = re.compile(r'[.!?"\'\)]\s*$')


def _best_text(block: _TextBlock) -> str:
    return block.text_clean if block.text_clean else block.text_raw


def _collect_footnote_refs(block_id: str, output: DocprocOutput) -> list[int]:
    """Return sorted list of footnote numbers referenced by a block."""
    return sorted({r.footnote_number for r in output.footnote_refs if r.parent_block_id == block_id})


def restitch(output: DocprocOutput) -> list[StitchedBlock]:
    """Merge cross-page paragraph continuations and return StitchedBlocks."""
    # Sort blocks by (page, reading_order)
    blocks = sorted(output.blocks, key=lambda b: (b.page_num_1, b.reading_order))

    if not blocks:
        return []

    # Build initial stitched blocks (one per input block)
    stitched: list[StitchedBlock] = []
    for b in blocks:
        stitched.append(
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
        )

    # Forward scan: merge paragraph continuations
    merged_indices: set[int] = set()

    for i in range(len(stitched) - 1):
        if i in merged_indices:
            continue

        prev = stitched[i]
        j = i + 1

        # Keep merging as long as the chain continues
        while j < len(stitched) and j not in merged_indices:
            curr = stitched[j]

            if not _should_merge(prev, curr):
                break

            # Merge curr into prev
            prev.text = prev.text.rstrip() + " " + curr.text.lstrip()
            prev.end_page = curr.end_page
            prev.block_ids.extend(curr.block_ids)
            prev.footnote_refs = sorted(set(prev.footnote_refs) | set(curr.footnote_refs))
            merged_indices.add(j)
            j += 1

    return [s for i, s in enumerate(stitched) if i not in merged_indices]


def _should_merge(prev: StitchedBlock, curr: StitchedBlock) -> bool:
    """Decide whether curr should be merged into prev."""
    if prev.block_type != "paragraph" or curr.block_type != "paragraph":
        return False

    # Must be on consecutive pages
    if curr.start_page != prev.end_page + 1:
        return False

    # curr must be the first block on its page (reading_order == 0)
    if curr.reading_order != 0:
        return False

    # prev must NOT end with sentence-ending punctuation
    prev_text = prev.text.rstrip()
    if _SENTENCE_END.search(prev_text):
        return False

    # curr must start with a lowercase letter
    curr_text = curr.text.lstrip()
    if not curr_text or not curr_text[0].islower():
        return False

    return True
