"""Diagnostic script to test each RAG component (embedding, retrieval, reranking, LLM) independently.

Usage:
    # Test Bedrock stack (as deployed)
    CHAT_LLM_PROVIDER=bedrock CHAT_EMBED_PROVIDER=bedrock CHAT_RERANK_PROVIDER=bedrock \
    CHAT_BEDROCK_EMBED_MODEL_ID=amazon.titan-embed-text-v2:0 \
    CHAT_EMBED_TASK_PREFIX="" CHAT_EMBED_DIMENSIONS=1024 \
    CHAT_DATABASE_DSN="<neon_dsn>" \
    uv run python scripts/diagnose_rag.py

    # Test with specific query
    ... uv run python scripts/diagnose_rag.py --query "What dates did Swettenham go to Singapore?"
"""

import argparse
import json
import logging
import time

logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s %(message)s")

from ras_rag_engine.config import RAGConfig as ChatConfig
from ras_rag_engine.providers import get_embed_provider, get_llm_provider, get_rerank_provider
from ras_rag_engine.retriever import RETRIEVE_SQL, RetrievedChunk, embed_query, retrieve
from ras_rag_engine.reranker import _doc_text, rerank
from ras_rag_engine.tools import format_chunks_for_context
from ras_rag_engine.agent import SYSTEM_PROMPT

import psycopg
from pgvector.psycopg import register_vector


DEFAULT_QUERY = "What dates did Swettenham go to Singapore?"

# The chunk we KNOW should be in results (page 111, "6th October. Reached Singapore at 1 P.M.")
TARGET_CHUNK_ID = "537e2d43ab9e4c9e99b3e3a6"
# Also the chunk on pages 110-111 mentioning Singapore trip
TARGET_CHUNK_ID_2 = "35778e10a07a34ed4a13b5de"


def test_embedding(config: ChatConfig, query: str):
    """Test embedding quality - check vector similarity to known-good chunks."""
    print("\n" + "=" * 80)
    print("STEP 1: EMBEDDING")
    print("=" * 80)
    print(f"Provider: {config.embed_provider}")
    if config.embed_provider == "bedrock":
        print(f"Model: {config.bedrock_embed_model_id}")
    else:
        print(f"Model: {config.embed_model}")
    print(f"Task prefix: '{config.embed_task_prefix}'")
    print(f"Dimensions: {config.embed_dimensions}")

    t0 = time.time()
    vec = embed_query(query, config)
    dt = time.time() - t0
    print(f"Embedding took {dt:.2f}s, vector dim={len(vec)}")
    print(f"Vector norm: {sum(x**2 for x in vec) ** 0.5:.4f}")
    print(f"First 5 values: {vec[:5]}")

    # Check cosine distance to our target chunks directly
    vec_str = "[" + ",".join(str(x) for x in vec) + "]"
    with psycopg.connect(config.dsn) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                """SELECT chunk_id, start_page, end_page,
                          embedding <=> %(vec)s::vector AS cosine_dist,
                          left(text, 100) AS preview
                   FROM chunks
                   WHERE doc_id = 'swettenham-journal-1874-1876-bbfb9df1239d'
                   ORDER BY embedding <=> %(vec)s::vector
                   LIMIT 20""",
                {"vec": vec_str},
            )
            rows = cur.fetchall()

    print(f"\nTop 20 Swettenham chunks by vector similarity:")
    target_found = False
    for i, (cid, sp, ep, dist, preview) in enumerate(rows):
        marker = ""
        if cid == TARGET_CHUNK_ID:
            marker = " <<<< TARGET (6th Oct reached Singapore)"
            target_found = True
        elif cid == TARGET_CHUNK_ID_2:
            marker = " <<<< TARGET2 (going to Singapore)"
        print(f"  {i+1:2d}. dist={dist:.4f} pp.{sp}-{ep} {preview[:80]}...{marker}")

    if not target_found:
        # Check where it actually ranks
        with psycopg.connect(config.dsn) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute(
                    """WITH ranked AS (
                        SELECT chunk_id, ROW_NUMBER() OVER (ORDER BY embedding <=> %(vec)s::vector) AS rank,
                               embedding <=> %(vec)s::vector AS dist
                        FROM chunks
                    )
                    SELECT rank, dist FROM ranked WHERE chunk_id = %(target)s""",
                    {"vec": vec_str, "target": TARGET_CHUNK_ID},
                )
                row = cur.fetchone()
                if row:
                    print(f"\n  WARNING: Target chunk ranks #{row[0]} overall (dist={row[1]:.4f})")
                else:
                    print(f"\n  ERROR: Target chunk not found in database!")

    return vec


def test_retrieval(config: ChatConfig, query: str):
    """Test hybrid retrieval (vector + FTS + RRF) - what chunks come back?"""
    print("\n" + "=" * 80)
    print("STEP 2: HYBRID RETRIEVAL (before reranking)")
    print("=" * 80)

    # Temporarily disable reranking to see raw retrieval
    orig_rerank = config.rerank_enabled
    config.rerank_enabled = False
    fetch_k = 30  # Same as rerank_candidates default

    t0 = time.time()
    chunks = retrieve(query, config, top_k=fetch_k)
    dt = time.time() - t0
    config.rerank_enabled = orig_rerank

    print(f"Retrieved {len(chunks)} chunks in {dt:.2f}s")
    print(f"\nAll {len(chunks)} chunks by RRF score:")
    target_rank = None
    for i, c in enumerate(chunks):
        marker = ""
        if c.chunk_id == TARGET_CHUNK_ID:
            marker = " <<<< TARGET"
            target_rank = i + 1
        elif c.chunk_id == TARGET_CHUNK_ID_2:
            marker = " <<<< TARGET2"
        src = c.source_filename[:40] if c.source_filename else "?"
        print(f"  {i+1:2d}. RRF={c.score:.4f} pp.{c.start_page}-{c.end_page} [{src}] {c.text[:60]}...{marker}")

    if target_rank:
        print(f"\n  Target chunk at rank #{target_rank}")
    else:
        print(f"\n  WARNING: Target chunk NOT in top {fetch_k} retrieval results!")

    return chunks


def test_reranking(config: ChatConfig, query: str, chunks: list[RetrievedChunk]):
    """Test reranking - how does it re-order the chunks?"""
    print("\n" + "=" * 80)
    print("STEP 3: RERANKING")
    print("=" * 80)
    print(f"Provider: {config.rerank_provider}")
    if config.rerank_provider == "bedrock":
        print(f"Model: {config.bedrock_rerank_model_id} (region: {config.bedrock_rerank_region})")

    if not chunks:
        print("  No chunks to rerank!")
        return []

    t0 = time.time()
    reranked = rerank(query, list(chunks), config, top_k=config.retrieval_top_k)
    dt = time.time() - t0

    print(f"Reranked to top {len(reranked)} in {dt:.2f}s")
    print(f"\nReranked chunks:")
    target_rank = None
    for i, c in enumerate(reranked):
        marker = ""
        if c.chunk_id == TARGET_CHUNK_ID:
            marker = " <<<< TARGET"
            target_rank = i + 1
        elif c.chunk_id == TARGET_CHUNK_ID_2:
            marker = " <<<< TARGET2"
        src = c.source_filename[:40] if c.source_filename else "?"
        print(f"  {i+1:2d}. score={c.score:.4f} pp.{c.start_page}-{c.end_page} [{src}] {c.text[:60]}...{marker}")

    if target_rank:
        print(f"\n  Target chunk at rank #{target_rank} after reranking")
    else:
        print(f"\n  WARNING: Target chunk NOT in top {config.retrieval_top_k} after reranking!")

        # Check if it was in pre-rerank set
        target_in_candidates = any(c.chunk_id == TARGET_CHUNK_ID for c in chunks)
        if target_in_candidates:
            print(f"  It WAS in the {len(chunks)} candidates but was ranked out by the reranker")
        else:
            print(f"  It was NOT even in the {len(chunks)} retrieval candidates")

    return reranked


def test_llm(config: ChatConfig, query: str, chunks: list[RetrievedChunk]):
    """Test LLM response quality - does it correctly cite the context?"""
    print("\n" + "=" * 80)
    print("STEP 4: LLM RESPONSE")
    print("=" * 80)
    print(f"Provider: {config.llm_provider}")
    if config.llm_provider == "bedrock":
        print(f"Model: {config.bedrock_chat_model_id}")
    else:
        print(f"Model: {config.llm_model}")
    print(f"Temperature: {config.llm_temperature}")

    if not chunks:
        print("  No chunks to test LLM with!")
        return

    context = format_chunks_for_context(chunks)
    system = SYSTEM_PROMPT + "\n\n---\nCONTEXT:\n" + context

    print(f"\nContext ({len(chunks)} chunks):")
    for i, c in enumerate(chunks):
        marker = ""
        if c.chunk_id == TARGET_CHUNK_ID:
            marker = " <<<< TARGET"
        elif c.chunk_id == TARGET_CHUNK_ID_2:
            marker = " <<<< TARGET2"
        print(f"  [{i+1}] pp.{c.start_page}-{c.end_page} {c.text[:60]}...{marker}")

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": query},
    ]

    llm = get_llm_provider(config)
    input_tokens = llm.count_tokens(messages)
    max_tokens = min(config.llm_max_tokens, config.llm_context_window - input_tokens - 64)
    print(f"\nInput tokens (est): {input_tokens}, max_tokens: {max_tokens}")

    t0 = time.time()
    result = llm.chat_completion(messages, max_tokens=max_tokens, temperature=config.llm_temperature)
    dt = time.time() - t0

    content = result.get("content") or ""
    reasoning = result.get("reasoning") or ""

    print(f"\nLLM response in {dt:.2f}s:")
    if reasoning:
        print(f"\n--- REASONING ---")
        print(reasoning[:2000])
        if len(reasoning) > 2000:
            print(f"... ({len(reasoning)} chars total)")
    print(f"\n--- CONTENT ---")
    print(content)

    # Check for the hallucination
    if "november" in content.lower() or "1st november" in content.lower():
        print("\n  !! HALLUCINATION DETECTED: Response mentions November but source says 6th October !!")
    if "6th october" in content.lower() or "6 october" in content.lower():
        print("\n  CORRECT: Response mentions 6th October")


def main():
    parser = argparse.ArgumentParser(description="Diagnose RAG pipeline")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Query to test")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM test")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embedding details")
    args = parser.parse_args()

    config = ChatConfig()

    print("Configuration:")
    print(f"  LLM provider: {config.llm_provider}")
    print(f"  Embed provider: {config.embed_provider}")
    print(f"  Rerank provider: {config.rerank_provider}")
    print(f"  DSN: {config.dsn[:40]}...")
    print(f"  Query: {args.query}")

    # Step 1: Embedding
    if not args.skip_embed:
        test_embedding(config, args.query)

    # Step 2: Raw retrieval
    raw_chunks = test_retrieval(config, args.query)

    # Step 3: Reranking
    reranked = test_reranking(config, args.query, raw_chunks)

    # Step 4: LLM
    if not args.skip_llm:
        test_llm(config, args.query, reranked)

    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
