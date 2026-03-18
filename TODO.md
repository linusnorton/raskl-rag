
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

## Journal split / duplication

~~The journals were originally published together in a collection. The PDFs have been divided up by journal but some journals start on the same page that a preceeding one finishes. Can we reliably detect and filter out the preceeding and following articles?~~

**Implemented:** Page-level filtering (drops blocks outside `page_range_label`) + title-based trimming (drops preceding-article content on shared first pages). Applied in the chunker pipeline. See `apps/chunker_indexer/README.md` D12.

**Remaining:**
- Last-page trimming: content from the *next* article on the shared last page is not yet filtered (would need the next article's title or a content-boundary heuristic)
- Deeper content detection for documents without `page_range_label` (20/118 docs)


## Document duplication

How could we detect duplicate documents in different PDFs.

## Filtering

Find out what mechanisms the agent has to filter information (should this be the embedding search or the full text search?). Can we ask it to find articles between x and y date or discussing particular locations?



