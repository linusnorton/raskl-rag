
# RAG Quality Investigation — Findings & Action Items

## Root Cause

The Neon database was indexed from a **different docproc run** than the local database.
The re-run produced different text block boundaries from DeepSeek OCR, which caused the
chunker to create an orphaned 198-char chunk containing the critical sentence
"6th October. Reached Singapore at 1 P.M." — too small to be retrieved by either
vector or full-text search.

In the local DB, that sentence is part of a 2131-char chunk that also includes
"I hope to be in Singapore by 1 P.M. to-morrow", making it easily retrievable
(reranker scores it #2 at 0.94).

The model differences (Titan Embed vs BGE-M3, Cohere Rerank vs BGE-Reranker, Qwen3-235B
vs Qwen3-8B) are **secondary** to the missing chunk. The LLM hallucinated "1st November"
because it never saw the "6th October" evidence and confused dates from the adjacent
Fourth Journal header "(23rd October to 4th November, 1875)".

## Immediate Fix: Re-index Neon from local DB

Write a migration script that:
1. Exports chunk text + metadata from local PostgreSQL (the known-good chunks)
2. Re-embeds with Bedrock Titan Embed v2 (must run from an env with Bedrock access)
3. Upserts into Neon (deleting old chunks per doc, inserting new)

This needs to handle the page_offset difference (local has 0, Neon expects -3 for
Swettenham). Either apply the offset during migration or keep as-is and update the
document record.

---

## Suggestions for RAG & Agentic RAG Improvements

### 1. Chunk quality guardrails

**Problem:** A 198-char orphan chunk is invisible to both vector and FTS retrieval.
Small chunks lack enough text for distinctive embeddings and match few FTS terms.

**Recommendation:** Add a post-chunking merge pass. After the chunker produces chunks,
merge any chunk below a configurable minimum (e.g. 50 tokens / ~200 chars) into its
**neighbour** — preferring the preceding chunk (append) since trailing fragments
typically belong to the previous section. This is different from `min_chunk_tokens`
which prevents small chunks from forming but can merge in the wrong direction.

**Implementation:** A new `merge_tiny_tails()` step in `pipeline.py` after
`chunk_blocks()`, before embedding.

### 2. Deterministic indexing pipeline

**Problem:** Re-running docproc on the same PDF can produce different text block
boundaries (DeepSeek OCR is non-deterministic), leading to different chunks and broken
retrieval for specific queries.

**Recommendations:**
- **Pin the JSONL artifacts.** Once docproc output is validated, treat the JSONL files
  as immutable inputs to the chunker. Store them in S3 alongside the PDF.
- **Add a `chunks_hash` to the documents table** so you can detect when re-indexing
  would change chunk boundaries. Warn/abort if chunks differ unexpectedly.
- **For the S3-triggered Lambda pipeline:** docproc should write JSONL to S3, and the
  chunker should read from S3 — not regenerate from the PDF each time.

### 3. Embedding model upgrade: Titan → Cohere Embed Multilingual v3

**Problem:** Amazon Titan Embed Text v2 is a general-purpose model. For historical
multilingual documents with archaic language, a specialised multilingual model performs
better. The Cohere model also properly supports asymmetric search with
`input_type=search_query` vs `search_document`.

**Recommendation:** Switch to `cohere.embed-multilingual-v3` on Bedrock. The provider
code already handles it — just change the model ID in `variables.tf`. Requires
re-indexing all documents.

**Trade-off:** Cohere Embed has higher per-request cost than Titan. Batching (96/request)
mitigates this for indexing.

### 4. Enable extended thinking for Qwen3 on Bedrock

**Problem:** Without thinking/reasoning mode, Qwen3 doesn't carefully parse dates and
facts from context passages. Locally, Qwen3's thinking mode helped it reason through
"the section heading says 3rd September to 6th October... he left for Singapore on the
5th... arrived 6th October."

**Recommendation:** Enable extended thinking via Bedrock's
`additionalModelRequestFields`. In `bedrock_llm.py`, add:

```python
kwargs["additionalModelRequestFields"] = {
    "thinking": {"type": "enabled", "budget_tokens": 2048}
}
```

This requires updating the token budget calculation in `agent.py` to account for
thinking tokens.

**Trade-off:** Higher latency and cost per request. Consider making it configurable
via `CHAT_LLM_THINKING_BUDGET`.

### 5. Retrieval diagnostics / observability

**Problem:** When the system gives a wrong answer, there's no easy way to see what
chunks were retrieved, how they scored, and what the reranker did.

**Recommendations:**
- **Structured logging:** Log the full retrieval pipeline per query:
  query → embedding time → top-30 RRF scores → rerank scores → final top-10.
  Use structured JSON logs that can be queried in CloudWatch.
- **Debug endpoint:** Add a `/debug` route to the Gradio app (behind a flag) that
  shows the retrieval results alongside the chat response. This existed implicitly
  in the local stack via console logs but is invisible in Lambda.
- **Eval harness:** Build a small set of known Q&A pairs (like "What date did
  Swettenham reach Singapore?" → "6th October 1875") and run them as a regression
  test after re-indexing. This would have caught this issue immediately.

### 6. FTS query strategy: AND-with-fallback

**Problem:** The OR-based FTS query (`'date' | 'swettenham' | 'go' | 'singapor'`)
matches very broadly — the target chunk ranked #182 because almost every Swettenham
chunk matches at least one of those terms.

**Recommendation:** Consider a **two-pass FTS strategy**:
1. First try AND-based matching (the PostgreSQL default). If it returns >= N results
   (e.g. 10), use those.
2. Fall back to OR-based if AND returns too few.

This would dramatically improve precision for queries where all terms appear together.
The current OR-only approach was chosen because some queries have terms that don't
all appear, but AND-with-fallback gets the best of both worlds.

### 7. Hybrid search weight tuning

**Problem:** The RRF formula uses a fixed k=60 and equal weighting for vector and FTS.
For this historical corpus, FTS may deserve higher weight since the text is OCR'd and
keyword matching on dates/names is very reliable.

**Recommendation:** Make the RRF weights configurable:
```python
rrf_score = alpha * (1/(k + vrank)) + (1 - alpha) * (1/(k + trank))
```
Where `alpha` is a config parameter (default 0.5). Allow tuning based on eval results.

### 8. Context window token counting

**Problem:** The Bedrock token counting is a rough estimate (`chars/3.5 + 100`).
This can cause either wasted context window (too conservative) or unexpected
truncation (too aggressive).

**Recommendation:** For Bedrock Converse API, use the response's
`usage.inputTokens` / `usage.outputTokens` from the first non-streaming call to
calibrate. Or use `bedrock-runtime` `count_tokens` API if available for the model.

### 9. Reranker domain hint

**Problem:** The local Qwen3 reranker uses a domain-specific instruction:
"Given a user question about historical JMBRAS and Swettenham journal documents,
judge whether the document passage is relevant." The Bedrock Cohere reranker gets
no such instruction.

**Recommendation:** Cohere Rerank v3.5 doesn't support custom instructions, but
you can prepend domain context to the query:
```python
augmented_query = f"Historical JMBRAS journal research question: {query}"
```
This gives the reranker a hint about the domain without changing the API.

### 10. Chunk overlap / sliding window

**Problem:** Hard chunk boundaries mean context at the edge of one chunk may be needed
to understand the next chunk. The "6th October" sentence was at a boundary.

**Recommendation:** Add configurable chunk overlap (e.g. 1-2 sentences from the end
of chunk N are prepended to chunk N+1). This provides context continuity at boundaries
and is standard practice in RAG systems. It increases storage by ~10-20% but
significantly improves retrieval for boundary-adjacent content.
