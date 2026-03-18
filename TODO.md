
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

The journals were originally published together in a collection. The PDFs have been divided up by journal but some journals start on the same page that a preceeding one finishes. Can we reliably detect and filter out the preceeding and following articles? E.g. file:///home/linus/Downloads/Stirling-RedWhiteFlag-1925%20(1).pdf


## Document duplication

How could we detect duplicate documents in different PDFs.

## Document type updates

journal_article
front_matter (often only contains the contents, but sometimes other interesting content is included such as a list of members, staff, patrons, etc.)
obituary
editorial ("foreword" and "preface" also fit here)
annual_report
agm_minutes
biographical_notes (short biographical notes on the journal contributors)
secondary_source (histories, biographies, etc.)
primary_source (first-hand accounts related to MBRAS studies)
mbras_monograph (book-sized articles, published separately as books by MBRAS)
mbras_reprint (older, out of print books printed by MBRAS. Can be primary or secondary sources)
index (MBRAS occasionally publishes an index to its articles. These sometimes include useful glossaries and other interesting bits)

Check that the agent can handle queries like "Find me primary sources on ..."

## Filtering

Find out what mechanisms the agent has to filter information (should this be the embedding search or the full text search?). Can we ask it to find articles between x and y date or discussing particular locations?



