# raskl-rag

A document processing pipeline for historical JMBRAS (Journal of the Malaysian Branch of the Royal Asiatic Society) and Swettenham PDFs. Extracts structured text, figures, footnotes, and metadata into JSONL records suitable for RAG indexing.

## Document Processing

- Per page bounding box detection
- Document meta data processing from first page
- Figure/image extraction in original form, jpeg and thumbnail*
- Diacritic OCR error correction
- Foot note extraction and reference tags

\* note that there is a heuristic detection to skip whole page image data where the document has the original scan as an image background.

Process:
1. Inventory
2. Text extraction (Docling or DeepSeek-OCR)
3. MuPDF
4. Metadata
5. Normalize
6. Boilerplate
7. Content area
8. Language
9. OCR cleanup
10. Footnotes
11. Footnote refs
12. Ref markup
13. Rotation
14. Figures (filtered)
15. Captions
16. Export

### Extraction Backends

The pipeline supports two text extraction backends:

- **Docling** (default) — Works well for born-digital "clean" PDFs with selectable text
- **DeepSeek-OCR** — Vision-language model that dramatically outperforms Docling/RapidOCR on scanned historical "messy" PDFs

The backend is auto-detected based on the PDF path (`docs/messy/` → DeepSeek, otherwise → Docling), or can be set explicitly with `--backend`.

## Setup

```bash
# Requires Python 3.11+ and uv
uv sync
```

## Running docproc

```bash
# Process a clean PDF (uses Docling backend by default)
uv run ras-docproc run --pdf "docs/clean/Abdullah (2011) JMBRAS 84(1), 1-22.pdf"

# Process a messy/scanned PDF (auto-detects DeepSeek backend)
uv run ras-docproc run --pdf "docs/messy/Swettenham Journal 1874-1876.pdf" --max-pages 30

# Explicitly choose a backend
uv run ras-docproc run --pdf "docs/messy/Swettenham Journal 1874-1876.pdf" --page-range 76 --backend deepseek
uv run ras-docproc run --pdf "docs/clean/Abdullah (2011) JMBRAS 84(1), 1-22.pdf" --backend docling

# Force re-processing
uv run ras-docproc run --pdf path/to/file.pdf --force
```

## Output Structure

```
data/out/{doc_id}/
├── documents.jsonl        # Document-level metadata
├── pages.jsonl            # Per-page metadata
├── text_blocks.jsonl      # All extracted text blocks
├── figures.jsonl          # Extracted figures with captions
├── footnotes.jsonl        # Detected footnotes
├── footnote_refs.jsonl    # Links between body text and footnotes
├── plates.jsonl           # Multi-figure plate pages
├── removed_blocks.jsonl   # Boilerplate / header / footer blocks
└── assets/                # Extracted images (original + JPG + thumbnails)
```

## HTML Debug Reports

```bash
uv run ras-docproc report --doc-id DOC_ID --pages 1,5,10
```

Generates an interactive HTML report with SVG overlays showing block boundaries, types, and extraction metadata.

## DeepSeek-OCR Setup

The DeepSeek backend requires a running vLLM server with the DeepSeek-OCR model. Use the launcher app:

```bash
# Install with GPU/vLLM support
uv sync --package ras-vllm-launcher --extra gpu

# Start the DeepSeek-OCR server (downloads model on first run, waits until ready)
uv run ras-vllm-launcher ocr

# Check server health
uv run ras-vllm-launcher health

# Stop the server
uv run ras-vllm-launcher down
```

Options:
```bash
uv run ras-vllm-launcher ocr --port 8001 --gpu-memory-utilization 0.8 --max-model-len 8192
```

If the docproc pipeline connects to a non-default URL, set `DOCPROC_VLLM_BASE_URL`:
```bash
DOCPROC_VLLM_BASE_URL=http://localhost:8001 uv run ras-docproc run --pdf ... --backend deepseek
```

## Chunker/Indexer

The chunker takes docproc JSONL output and:
1. **Re-stitches** paragraphs split across page boundaries
2. **Chunks** text into semantic sections based on headings (~512 tokens)
3. **Embeds** chunks via Qwen3-Embedding (served by vLLM)
4. **Indexes** chunks + vectors into PostgreSQL/pgvector

```bash
# Download embedding + reranker models
uv run ras-vllm-launcher download --role all

# Start PostgreSQL
docker compose up -d
uv run ras-chunker init-db

# Preview chunk boundaries (no DB/vLLM needed)
uv run ras-chunker plan --doc-id swettenham-journal-1874-1876-bbfb9df1239d

# Index a document (requires vLLM embedding server + PostgreSQL)
uv run ras-chunker index --doc-id swettenham-journal-1874-1876-bbfb9df1239d
```

## Other Apps

- **ras-chat-ui** — Gradio-based chat interface for RAG queries (skeleton)

## Testing

```bash
# Run all tests (excluding slow tests that need vLLM)
uv run --package ras-docproc pytest apps/docproc/tests/ -v -m "not slow"

# Run all tests including DeepSeek E2E (requires vLLM server running)
uv run --package ras-docproc pytest apps/docproc/tests/ -v

# Run only unit tests
uv run --package ras-docproc pytest apps/docproc/tests/test_utils.py -v

# Run only e2e tests
uv run --package ras-docproc pytest apps/docproc/tests/test_e2e_pipeline.py -v

# Run DeepSeek parser unit tests (no vLLM needed)
uv run --package ras-docproc pytest apps/docproc/tests/test_deepseek_extraction.py -v -m "not slow"
```
