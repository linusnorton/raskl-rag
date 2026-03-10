# ras-docproc — PDF Processing Pipeline

## What this app does

`ras-docproc` takes raw PDF files — academic journal articles from JMBRAS (Journal of the Malayan
Branch of the Royal Asiatic Society) and related collections — and produces structured JSONL files
ready for downstream indexing and search. It handles two very different PDF types: clean
born-digital PDFs and messy historical scans of documents printed between the 1870s and the
present day.

## How it fits into the pipeline

```
PDF files
    │
    ▼
ras-docproc  ← this app
    │
    ▼
data/out/{doc_id}/
  documents.jsonl
  text_blocks.jsonl
  footnotes.jsonl
  footnote_refs.jsonl
  figures.jsonl
  pages.jsonl
  ...
    │
    ▼
ras-chunker-indexer  (chunks + embeds + stores in PostgreSQL)
    │
    ▼
ras-rag-engine  (OpenAI-compatible RAG API → Open WebUI)
```

## Architecture overview

The pipeline runs seventeen stages in sequence. Each stage reads from the previous stage's output
and writes an augmented version forward. The final stage writes JSONL files to disk.

| # | Stage | What it does |
|---|-------|-------------|
| 1 | Inventory | Discovers the PDF, computes its SHA256 hash, generates a `doc_id` |
| 2 | Extract (Qwen3 VL) | Extracts structured text blocks from the PDF via Bedrock |
| 3 | Extract (MuPDF) | Low-level extraction: fonts, span data, images, raw metadata |
| 4 | Extract Metadata | Parses the JSTOR/MUSE cover page for title, author, year, DOI, journal pages |
| 4c | Classify Doc Type | Classifies the document as `primary_source` or `journal_article` via LLM |
| 5 | Normalize Text | NFKC normalization, dehyphenation, superscript cleanup |
| 6 | Detect Boilerplate | Removes platform headers, running footers, page numbers |
| 7 | Detect Content Area | Computes the usable text area per page |
| 8 | Detect Language | Identifies the language of each text block (English, Malay, Chinese, Arabic) |
| 9 | OCR Cleanup | For English blocks: removes spurious diacritics and non-Latin characters |
| 10 | Detect Footnotes | Finds footnote blocks in the bottom zone of each page |
| 11 | Link Footnote Refs | Links superscript numbers in body text to the corresponding footnotes |
| 12 | Footnote Ref Markup | Inserts `[ref:N]` markers in the cleaned text |
| 13 | Detect Rotation | Flags pages with vertical text and suggests rotation angle |
| 14 | Detect Figures | Extracts embedded images, produces JPG + 200×200px thumbnails |
| 15 | Detect Captions | Matches "Fig. N" / "Plate N" text to figure records |
| 16 | Export JSONL | Writes all records to `data/out/{doc_id}/` |

## Key design decisions

### D1 — Extraction via Qwen3 VL (Bedrock)

**What we chose:** A single extraction backend using the Qwen3-VL-235B vision-language model via
AWS Bedrock's Converse API. Each page is rendered as a PNG at 300 DPI, sent to the model, and
the response is parsed into structured text blocks. Pages are processed in parallel (default 10
workers). Since Qwen3 VL returns Markdown text without spatial coordinates, blocks are assigned
full-page degenerate bounding boxes.

**Why a single backend:** Earlier versions supported Docling (for clean PDFs) and DeepSeek-OCR
(for messy scans via local vLLM). In practice, Qwen3 VL handles both clean and messy PDFs well
enough that maintaining multiple backends added complexity without benefit. The legacy backends
were removed.

### D2 — Page offset correction

**What we chose:** JSTOR and MUSE PDFs include a cover/metadata page before the article begins.
PDF page 1 is the JSTOR cover, not journal page 1. The pipeline computes:

```
page_offset = last_page_number_on_cover − total_pages_in_pdf
```

This offset is applied once during the Metadata stage (Stage 4), adjusting the page numbers of
every block before export.

**Fallback detection:** If the cover page doesn't contain a parseable page range, a fallback
algorithm samples running page numbers from the second half of the document (to avoid roman
numerals and unnumbered plates). If 3+ pages agree on the same offset, it is adopted.

**Why baked in, not applied at display time:** If the offset were applied later, every downstream
consumer (the chunker, the chat UI, the citation renderer) would need to know about it and apply it
consistently. A single correction at the source means all downstream page numbers are simply
correct. The `page_offset` value is also stored in `documents.jsonl` as a record of what was
applied — not for re-application.

**Why this matters:** The chat interface cites documents by page number. A reader going to the
source needs the page number to match what they see in the journal, not the PDF viewer.

### D3 — Boilerplate detection

**What we chose:** A two-phase approach.

- **Phase 1 (zone-based):** Any block whose vertical centre falls in the top 10% or bottom 20%
  of the page is classified as a header or footer candidate.
- **Phase 2 (frequency-based):** Any normalised line of text that appears on 30% or more of pages
  is classified as boilerplate (e.g. "Downloaded from JSTOR", running journal title, page numbers).

Both phases feed a removal step that also strips known platform strings ("PROJECT MUSE", "JSTOR").

**Why:** Boilerplate content appears in almost every chunk if not removed. This pollutes the
vector embeddings (making all chunks look similar) and confuses the LLM, which sees hundreds of
copies of "JSTOR: Journal of the Malayan Branch of the Royal Asiatic Society" across different
passages.

### D4 — Footnote detection and linking

**What we chose:** A dual approach combining regex patterns with font-size analysis from MuPDF.

**Footnote detection:** Blocks in the bottom 28% of the page (below the 72% mark) that start
with a numeric pattern are classified as footnotes. When a block has a full-page bounding box
(y0 ≈ 0, y1 ≈ page height) — as produced by Qwen3 VL — the zone check is skipped and
classification relies purely on text pattern matching. This prevents footnotes from being
missed when the extraction backend assigns degenerate bboxes to all blocks.

- `1 Some text` — number then space
- `(1) Some text` — parentheses
- `1. Some text` — number then period
- `1) Some text` — number then bracket

**Multi-footnote splitting:** Qwen3 VL often merges consecutive footnotes into a single text
block (e.g. `"35 Weld to CO 273/105. 36 Smith (1990: 12)."`). The detection stage splits
these by finding ascending sequential numbers separated by `. ` boundaries, validating that the
numbers are close together (gap < 3× count). Each sub-footnote gets its own `FootnoteRecord`
with independent classification.

**Superscript linking:** Two independent methods run in parallel:
1. **Regex:** Patterns like `word43` (digit immediately after a letter), `word.43`, `word. 45`
   (space-separated after sentence punctuation), `(43)`, `[43]`
2. **MuPDF span analysis:** Spans where `font_size < 70% of the median font size on that page` and
   the span contains only digits are treated as superscripts.

Cross-page (endnote-style) linking is supported: if a footnote number has no match on the same
page, it looks across the whole document.

**Why both methods:** Historical typesetting varied enormously. 1870s journals used superscript
type; 1950s journals used parenthetical references; digitised PDFs sometimes have font metadata
that allows superscript detection but sometimes do not. The dual approach maximises coverage
without requiring manual configuration per document.

### D5 — Footnote classification

**What we chose:** Each detected footnote is classified as `citation`, `explanatory`, or `mixed`
using regex-based heuristics in `classify_footnote_type()`. The classification is stored in the
`footnote_type` field of `FootnoteRecord` and exported to `footnotes.jsonl`.

**Pattern categories:** Scholarly author-year references (`Gullick (1992: 246-8)`), archival
sources (`CO 273/105`, `IOR`, `BL/APAC/`), ibid/cross-references, see/cf. references,
newspaper citations, official documents (`'s journal/diary` — matches both straight and curly
apostrophes, with or without trailing comma), URLs, and personal communications.

**Citation vs mixed:** After removing all citation pattern matches from the text, if ≤8 words
remain, it's a pure `citation`; otherwise `mixed`. Footnotes with no citation patterns are
`explanatory`.

**Why:** Downstream, the chunker annotates citation footnotes with `[cites:]` markers so the
RAG system can distinguish between what a secondary author claims and what a primary source
records. This solves the "citation of citation" problem where the system attributes a claim to
the paper author when the footnote actually cites an external primary source.

### D6 — Deterministic, content-based IDs

**What we chose:**

```
doc_id  = slug(filename) + "-" + sha256(file)[:12]
block_id = sha256(doc_id + ":" + page + ":" + bbox + ":" + kind + ":" + text_hash)[:24]
```

Example doc_id: `abdullah-2011-jmbras-84-1-1-22-a3b1c2d3e4f5`

**Why:** The pipeline can be re-run on the same PDF (e.g. after a bug fix) and produce identical
IDs for unchanged content. The chunker/indexer can detect this and skip re-embedding blocks that
haven't changed. If the PDF content changes, the hash changes and new IDs are produced automatically.

This also means IDs are meaningful at a glance (the slug prefix) while being collision-resistant
(the hash suffix).

### D7 — AWS Lambda deployment

**What we chose:** An S3-triggered Lambda handler (`lambda_handler.py`) that:

1. Downloads the uploaded PDF from S3 (`uploads/*.pdf`)
2. Runs the full pipeline using the **Qwen3 VL** backend (Bedrock, no GPU needed)
3. Uploads versioned JSONL output to `processed/{doc_id}/v{N}/` with a `_meta.json` manifest
4. Diffs against the previous version (if any) using `diff.py`
5. Sends an HTML email via AWS SES with a summary of changes (`notify.py`)

Version numbers auto-increment. Each version is a complete snapshot of the pipeline output.

The `processed/{doc_id}/v{N}/documents.jsonl` upload triggers the chunker Lambda downstream
(see `apps/chunker_indexer/README.md`).

**Why versioned output:** Re-processing the same PDF (e.g. after a pipeline improvement) should
not silently overwrite previous results. Versioning lets us compare outputs and track quality
improvements over time.

**`[cloud]` dependency:** The Lambda handler, Qwen3 VL backend, and SES notifications require
`boto3`, which is an optional `[cloud]` extra in `pyproject.toml`.

### D8 — Document type classification

**What we chose:** An LLM-based classification stage (Stage 4c) that labels each document as
`primary_source` or `journal_article`. The classifier sends the first 2 pages of extracted text
plus existing metadata (title, author, year, publication, filename) to Qwen3 VL with a zero-shot
prompt. The LLM returns a JSON object with `document_type`, `confidence`, and `reasoning`.

**Why these two types:** The JMBRAS corpus contains two fundamentally different kinds of documents:
peer-reviewed academic papers (secondary sources) and colonial-era journals, diaries, and
correspondence (primary sources). This distinction maps to the primary/secondary source taxonomy
from library science. The RAG system uses this label to (1) annotate context headers with
`[Primary Source]` or `[Journal Article]` tags, (2) guide the LLM to prefer primary sources for
factual claims about historical events, and (3) display source type in citation output.

**Why a string field (not enum):** Using `str | None` means new types (e.g. `book_chapter`,
`government_report`) can be added without schema migration. The `None` default provides graceful
degradation for documents processed before this stage was added.

## Output format

All output goes to `data/out/{doc_id}/`:

| File | Contents |
|------|----------|
| `documents.jsonl` | One record: title, author, year, DOI, page range, `page_offset`, `document_type` |
| `pages.jsonl` | One record per page: dimensions, rotation, content bounding box |
| `text_blocks.jsonl` | Body text blocks: raw text, cleaned text, language, bbox, block type |
| `removed_blocks.jsonl` | Blocks removed as boilerplate (for debugging) |
| `footnotes.jsonl` | Detected footnotes: number, page, text |
| `footnote_refs.jsonl` | Links between body refs and footnote records |
| `figures.jsonl` | Extracted images with paths to JPG and thumbnail assets |
| `plates.jsonl` | Multi-figure plate pages (if present) |

Assets (images) go to `data/out/{doc_id}/assets/`.

## Configuration

All variables use the `DOCPROC_` prefix. The most commonly needed ones:

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCPROC_OUT_DIR` | `data` | Root output directory |
| `DOCPROC_BOILERPLATE_THRESHOLD` | `0.3` | Fraction of pages a line must appear on to be boilerplate |
| `DOCPROC_FOOTER_ZONE_TOP` | `0.80` | Top of footer zone (fraction of page height from top) |
| `DOCPROC_HEADER_ZONE_BOTTOM` | `0.10` | Bottom of header zone (fraction of page height from top) |
| `DOCPROC_FOOTNOTE_ZONE_TOP` | `0.72` | Top of footnote zone (fraction of page height from top) |
| `DOCPROC_MIN_LANG_CHARS` | `30` | Blocks shorter than this get `lang="unknown"` |
| `DOCPROC_FORCE` | `false` | Re-process even if output already exists for this SHA256 |
| `DOCPROC_BEDROCK_REGION` | `eu-west-2` | AWS region for Bedrock (Qwen3 VL backend only) |
| `DOCPROC_BEDROCK_OCR_MODEL_ID` | `qwen.qwen3-vl-235b-a22b` | Bedrock model ID (Qwen3 VL backend only) |
| `DOCPROC_QWEN3VL_DPI` | `300` | Page rendering DPI (Qwen3 VL backend only) |
| `DOCPROC_QWEN3VL_MAX_TOKENS` | `4096` | Max tokens per page (Qwen3 VL backend only) |
| `DOCPROC_QWEN3VL_MAX_WORKERS` | `10` | Max parallel Bedrock calls (Qwen3 VL backend only) |
| `NOTIFY_EMAIL` | — | Recipient email for SES diff notifications (Lambda only) |
| `SENDER_EMAIL` | — | Sender email for SES diff notifications (Lambda only) |

See root `CLAUDE.md` for all CLI commands.

## Testing

Tests live in `apps/docproc/tests/` and are end-to-end by design: they run the full pipeline on
real PDFs and assert on the output.

```bash
uv run --package ras-docproc pytest apps/docproc/tests/ -v
```

Three test PDFs are used:

| PDF | Why it's used |
|-----|---------------|
| `docs/messy/Swettenham Journal 1874-1876.pdf` | 151-page JSTOR scan — tests boilerplate detection, metadata, figures |
| `docs/clean/Abdullah (2011) JMBRAS 84(1), 1-22.pdf` | 23-page born-digital — tests footnote detection and linking |
| `docs/clean/Aznan (2020) JMBRAS 93(1), 119-131.pdf` | 14 pages, page 12 has vertical text — tests rotation detection |

Key test coverage:
- All 8 JSONL output files are produced and valid
- Boilerplate blocks are removed and logged to `removed_blocks.jsonl`
- Footnotes are detected and not leaked into `text_blocks.jsonl`
- `[ref:N]` markup appears in `text_clean` fields
- Vertical text triggers a rotation suggestion
- Running the pipeline twice on the same PDF produces identical `doc_id` and `block_id` values
