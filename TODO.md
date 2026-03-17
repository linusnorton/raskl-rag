
# Tuning

 - Thinking budget might need to be increased?
 - Drop temperature to 0.4?
 - System prompt

# RAG Quality — Remaining Action Items

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

Aliba cloud keys in github
Voice/transcribe
Check google search / tools