
# RAG Quality — Remaining Action Items

## Retrieval diagnostics / observability

**Problem:** When the system gives a wrong answer, there's no easy way to see what
chunks were retrieved, how they scored, and what the reranker did.

**Recommendations:**
- **Structured logging:** Log the full retrieval pipeline per query:
  query → embedding time → top-30 RRF scores → rerank scores → final top-10.
  Use structured JSON logs that can be queried in CloudWatch.
- **Debug endpoint:** Add a `/debug` route to the Gradio app (behind a flag) that
  shows the retrieval results alongside the chat response.
- **Eval harness:** Build a small set of known Q&A pairs and run them as a regression
  test after re-indexing.

## Hybrid search weight tuning

**Problem:** The RRF formula uses a fixed k=60 and equal weighting for vector and FTS.
For this historical corpus, FTS may deserve higher weight since the text is OCR'd and
keyword matching on dates/names is very reliable.

**Recommendation:** Make the RRF weights configurable:
```python
rrf_score = alpha * (1/(k + vrank)) + (1 - alpha) * (1/(k + trank))
```
Where `alpha` is a config parameter (default 0.5). Allow tuning based on eval results.

## Context window token counting

**Problem:** The Bedrock token counting is a rough estimate (`chars/3.5 + 100`).
This can cause either wasted context window (too conservative) or unexpected
truncation (too aggressive).

**Recommendation:** Use `bedrock-runtime` `count_tokens` API if available, or
calibrate from response `usage.inputTokens` / `usage.outputTokens`.

## Chunk overlap / sliding window

**Problem:** Hard chunk boundaries mean context at the edge of one chunk may be needed
to understand the next chunk.

**Recommendation:** Add configurable chunk overlap (e.g. 1-2 sentences from the end
of chunk N are prepended to chunk N+1). Increases storage by ~10-20% but improves
retrieval for boundary-adjacent content.
