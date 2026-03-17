"""Compare OCR quality, speed and cost across Qwen3-VL-235B (Bedrock) and Qwen3.5 models (Alibaba Model Studio)."""

from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import fitz
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DOCS_DIR = Path("docs")

# (pdf_path_relative_to_DOCS_DIR, 1-based page number, description)
TEST_PAGES: list[tuple[str, int, str]] = [
    # Clean text with footnotes
    ("clean/Abdullah (2011) JMBRAS 84(1), 1-22.pdf", 5, "clean text + footnotes"),
    ("clean/Blackburn (2015) JMBRAS 88(2), 51-76.pdf", 8, "clean text + footnotes"),
    # Title / metadata page
    ("clean/Abdullah (2011) JMBRAS 84(1), 1-22.pdf", 1, "title page"),
    ("clean/Gallop (2013) JMBRAS 86(2), 1-32.pdf", 1, "title page"),
    # Vertical / rotated text
    ("clean/Aznan (2020) JMBRAS 93(1), 119-131.pdf", 12, "vertical/rotated text"),
    # Mixed Malay/English
    ("clean/Aznan (2020) JMBRAS 93(1), 119-131.pdf", 3, "mixed Malay/English"),
    ("clean/Karuppannan & Yacob (2020) JMBRAS 93(1), 67-89.pdf", 5, "mixed Malay/English"),
    # Dense academic with citations
    ("clean/Borschberg (2017) JMBRAS 90(1), 29-60.pdf", 10, "dense citations"),
    ("clean/Rivers (2011) JMBRAS 84(1), 47-101.pdf", 20, "dense citations"),
    # Tables / structured data
    ("clean/Earl of Cranbrook (2013) JMBRAS 86(1), 79-112.pdf", 15, "tables/structured"),
    ("clean/Koike (2017) JMBRAS 90(1), 73-100.pdf", 10, "tables/structured"),
    # Messy JSTOR scans
    ("messy/Swettenham Journal 1874-1876.pdf", 10, "messy scan"),
    ("messy/Swettenham Journal 1874-1876.pdf", 50, "messy scan"),
    ("messy/Swettenham Journal 1874-1876.pdf", 100, "messy scan"),
    # Second messy PDF
    ("messy/The Origin of British Intervention by Khoo Kay Kim.pdf", 5, "messy scan"),
    ("messy/The Origin of British Intervention by Khoo Kay Kim.pdf", 20, "messy scan"),
    # Chinese / non-Latin
    ("clean/Chen (2016) JMBRAS 89(1), 123-135.pdf", 5, "Chinese/non-Latin"),
    # Maps / figures
    ("clean/Gullick (2014) JMBRAS 87(2), 47-89.pdf", 10, "maps/figures"),
    # Emily Innes — scanned 19th-century book (261pp)
    ("messy/The Chersonese With the Gilding Off by Emily Innes.pdf", 1, "Innes title page"),
    ("messy/The Chersonese With the Gilding Off by Emily Innes.pdf", 15, "Innes body text"),
    ("messy/The Chersonese With the Gilding Off by Emily Innes.pdf", 50, "Innes mid-book"),
    ("messy/The Chersonese With the Gilding Off by Emily Innes.pdf", 120, "Innes mid-book"),
    ("messy/The Chersonese With the Gilding Off by Emily Innes.pdf", 200, "Innes late-book"),
]

SYSTEM_PROMPT = """\
You are a document OCR engine. Convert the page image to clean Markdown text.

Rules:
- Preserve the reading order exactly as it appears on the page.
- Use **bold** and *italic* for emphasis where the original uses bold/italic.
- Use # for main headings, ## for subheadings.
- Use > for block quotes.
- Separate each distinct paragraph or entry with a blank line.
- In journals or diaries, treat each date entry (e.g. "6th October.", "Monday, 12th March.") \
as the start of a new paragraph — always insert a blank line before it.
- Preserve natural paragraph breaks from the source document; do not merge adjacent paragraphs \
into a single block of text.
- If the page has footnotes (small text at the bottom, often after a horizontal rule), \
separate them with --- and format each as a numbered line: 1. footnote text
- Convert superscript footnote references in body text to ^N notation (e.g. ^1, ^23).
- For images, illustrations, maps, plates, engravings, or figures on the page, emit a single line: \
![Figure](brief description). Do not describe the image in detail — just note what it depicts in a few words. \
If the illustration is rotated sideways on the page (i.e. you need to rotate the page 90° clockwise \
or counter-clockwise to view it correctly), add |rotate90cw or |rotate90ccw at the end of the \
description, e.g. ![Figure](a house with palm trees|rotate90cw). \
Only emit a figure tag for actual illustrations, engravings, maps, photographs, or plates — NOT for \
blank pages, empty pages, pages with only minor blemishes/specks, or decorative page borders.
- Do NOT add any commentary, explanation, or preamble. Output ONLY the Markdown text."""

MAX_TOKENS = 4096
DPI = 300
JPEG_QUALITY = 85
MAX_IMAGE_BYTES = 3_700_000
DPI_FALLBACKS = (200, 150)

MODELS = [
    {
        "name": "Qwen3-VL-235B (Bedrock)",
        "provider": "bedrock",
        "model_id": "qwen.qwen3-vl-235b-a22b",
        "input_cost_per_m": 0.53,
        "output_cost_per_m": 2.66,
    },
    {
        "name": "Qwen3.5-397B-A17B (Model Studio)",
        "provider": "model_studio",
        "model_id": "qwen3.5-397b-a17b",
        "input_cost_per_m": None,  # TBD
        "output_cost_per_m": None,
    },
    {
        "name": "Qwen3.5-122B-A10B (Model Studio)",
        "provider": "model_studio",
        "model_id": "qwen3.5-122b-a10b",
        "input_cost_per_m": None,  # TBD
        "output_cost_per_m": None,
    },
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class PageResult:
    pdf_name: str
    page_num: int
    description: str
    model_name: str
    provider: str
    markdown: str
    input_tokens: int
    output_tokens: int
    elapsed_seconds: float
    cost_usd: float | None
    error: str | None = None


# ---------------------------------------------------------------------------
# Page rendering (reuses logic from extract_qwen3vl.py)
# ---------------------------------------------------------------------------


def render_page(pdf_path: str, page_index: int) -> tuple[bytes, float, float]:
    """Render a PDF page to JPEG bytes. Returns (jpeg_bytes, page_width_pts, page_height_pts)."""
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_index]
        pw, ph = page.rect.width, page.rect.height
        for current_dpi in (DPI, *DPI_FALLBACKS):
            zoom = current_dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("jpeg", jpg_quality=JPEG_QUALITY)
            if len(img_bytes) <= MAX_IMAGE_BYTES:
                return img_bytes, pw, ph
        return img_bytes, pw, ph
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# Bedrock caller
# ---------------------------------------------------------------------------


def call_bedrock(model_id: str, image_bytes: bytes) -> tuple[str, int, int, float]:
    """Call Bedrock Converse API. Returns (markdown, input_tokens, output_tokens, elapsed_s)."""
    import boto3
    from botocore.config import Config as BotoConfig

    config = BotoConfig(read_timeout=120, retries={"max_attempts": 2})
    client = boto3.client("bedrock-runtime", region_name="eu-west-2", config=config)

    t0 = time.time()
    response = client.converse(
        modelId=model_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {"image": {"format": "jpeg", "source": {"bytes": image_bytes}}},
                    {"text": "Convert this page to Markdown following the system instructions."},
                ],
            },
        ],
        system=[{"text": SYSTEM_PROMPT}],
        inferenceConfig={"maxTokens": MAX_TOKENS, "temperature": 0.0},
    )
    elapsed = time.time() - t0

    text = response["output"]["message"]["content"][0]["text"]
    text = re.sub(r"^```(?:markdown)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)

    usage = response.get("usage", {})
    input_tokens = usage.get("inputTokens", 0)
    output_tokens = usage.get("outputTokens", 0)

    return text, input_tokens, output_tokens, elapsed


# ---------------------------------------------------------------------------
# Model Studio caller (OpenAI-compatible)
# ---------------------------------------------------------------------------


def call_model_studio(model_id: str, image_bytes: bytes) -> tuple[str, int, int, float]:
    """Call Alibaba Model Studio via OpenAI-compatible API. Returns (markdown, input_tokens, output_tokens, elapsed_s)."""
    from openai import OpenAI

    api_key = os.environ.get("MODEL_STUDIO_KEY")
    if not api_key:
        raise RuntimeError("MODEL_STUDIO_KEY not set in environment")

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )

    b64 = base64.b64encode(image_bytes).decode("utf-8")

    t0 = time.time()
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": "Convert this page to Markdown following the system instructions."},
                ],
            },
        ],
        max_tokens=MAX_TOKENS,
        temperature=0.0,
    )
    elapsed = time.time() - t0

    text = response.choices[0].message.content or ""
    # Strip thinking blocks (Qwen3.5 reasoning)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:markdown)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)

    usage = response.usage
    input_tokens = usage.prompt_tokens if usage else 0
    output_tokens = usage.completion_tokens if usage else 0

    return text, input_tokens, output_tokens, elapsed


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


def generate_html_report(results: list[PageResult], output_path: Path) -> None:
    """Generate an HTML comparison report."""
    # Group results by (pdf_name, page_num)
    from collections import OrderedDict

    pages: dict[tuple[str, int], list[PageResult]] = OrderedDict()
    for r in results:
        key = (r.pdf_name, r.page_num)
        pages.setdefault(key, []).append(r)

    model_names = [m["name"] for m in MODELS]

    # Summary stats per model
    summary: dict[str, dict] = {}
    for name in model_names:
        model_results = [r for r in results if r.model_name == name and r.error is None]
        if not model_results:
            summary[name] = {"count": 0, "avg_time": 0, "total_input": 0, "total_output": 0, "total_cost": None}
            continue
        total_time = sum(r.elapsed_seconds for r in model_results)
        total_input = sum(r.input_tokens for r in model_results)
        total_output = sum(r.output_tokens for r in model_results)
        costs = [r.cost_usd for r in model_results if r.cost_usd is not None]
        summary[name] = {
            "count": len(model_results),
            "avg_time": total_time / len(model_results),
            "total_input": total_input,
            "total_output": total_output,
            "total_cost": sum(costs) if costs else None,
        }

    # Build HTML
    html_parts = [
        """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>OCR Model Comparison</title>
<style>
body { font-family: system-ui, sans-serif; margin: 20px; background: #f5f5f5; }
h1 { color: #333; }
h2 { color: #555; border-bottom: 1px solid #ccc; padding-bottom: 4px; }
table { border-collapse: collapse; width: 100%; margin: 16px 0; }
th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
th { background: #f0f0f0; }
.page-section { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.columns { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
.column { border: 1px solid #e0e0e0; border-radius: 4px; padding: 12px; }
.column h3 { margin-top: 0; font-size: 14px; color: #666; }
.markdown-output { white-space: pre-wrap; font-family: monospace; font-size: 12px; line-height: 1.5;
  background: #fafafa; padding: 8px; border-radius: 4px; max-height: 600px; overflow-y: auto; }
.stats { font-size: 11px; color: #888; margin-top: 8px; }
.page-image { max-width: 300px; max-height: 400px; border: 1px solid #ddd; }
.error { color: #c00; font-style: italic; }
.page-header { display: flex; gap: 20px; align-items: flex-start; margin-bottom: 16px; }
</style></head><body>
<h1>OCR Model Comparison Report</h1>
"""
    ]

    # Summary table
    html_parts.append("<h2>Summary</h2><table><tr><th>Model</th><th>Pages</th><th>Avg Time/Page</th>"
                      "<th>Total Input Tokens</th><th>Total Output Tokens</th><th>Total Cost</th></tr>")
    for name in model_names:
        s = summary[name]
        cost_str = f"${s['total_cost']:.4f}" if s["total_cost"] is not None else "TBD"
        html_parts.append(
            f"<tr><td>{name}</td><td>{s['count']}</td><td>{s['avg_time']:.1f}s</td>"
            f"<td>{s['total_input']:,}</td><td>{s['total_output']:,}</td><td>{cost_str}</td></tr>"
        )
    html_parts.append("</table>")

    # Per-page sections
    for (pdf_name, page_num), page_results in pages.items():
        desc = page_results[0].description if page_results else ""
        html_parts.append(f'<div class="page-section">')
        html_parts.append(f"<h2>{pdf_name} — Page {page_num} ({desc})</h2>")

        # Try to embed page image
        pdf_path = DOCS_DIR / pdf_name
        if pdf_path.exists():
            try:
                img_bytes, _, _ = render_page(str(pdf_path), page_num - 1)
                b64_img = base64.b64encode(img_bytes).decode()
                html_parts.append(f'<div class="page-header">'
                                  f'<img class="page-image" src="data:image/jpeg;base64,{b64_img}" '
                                  f'alt="Page {page_num}"></div>')
            except Exception:
                pass

        html_parts.append('<div class="columns">')
        for model_name in model_names:
            matching = [r for r in page_results if r.model_name == model_name]
            html_parts.append('<div class="column">')
            html_parts.append(f"<h3>{model_name}</h3>")
            if matching:
                r = matching[0]
                if r.error:
                    html_parts.append(f'<div class="error">{r.error}</div>')
                else:
                    import html
                    html_parts.append(f'<div class="markdown-output">{html.escape(r.markdown)}</div>')
                    cost_str = f"${r.cost_usd:.4f}" if r.cost_usd is not None else "TBD"
                    html_parts.append(
                        f'<div class="stats">{r.elapsed_seconds:.1f}s | '
                        f'{r.input_tokens:,} in / {r.output_tokens:,} out | {cost_str}</div>'
                    )
            else:
                html_parts.append('<div class="error">No result</div>')
            html_parts.append("</div>")
        html_parts.append("</div></div>")

    html_parts.append("</body></html>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(html_parts))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Load existing results to allow incremental runs
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    json_path = reports_dir / "ocr_comparison.json"

    results: list[PageResult] = []
    existing_keys: set[tuple[str, int, str]] = set()
    if json_path.exists():
        for r in json.loads(json_path.read_text()):
            pr = PageResult(**r)
            results.append(pr)
            existing_keys.add((pr.pdf_name, pr.page_num, pr.model_name))
        print(f"Loaded {len(results)} existing results from {json_path}")

    total = len(TEST_PAGES) * len(MODELS)
    done = 0

    for pdf_rel, page_num, description in TEST_PAGES:
        pdf_path = DOCS_DIR / pdf_rel
        if not pdf_path.exists():
            print(f"SKIP {pdf_rel} p{page_num} — file not found")
            continue

        print(f"\n{'='*60}")
        print(f"Page: {pdf_rel} p{page_num} ({description})")
        print(f"{'='*60}")

        # Render once, reuse for all models
        img_bytes, pw, ph = render_page(str(pdf_path), page_num - 1)
        print(f"  Rendered: {len(img_bytes):,} bytes")

        for model in MODELS:
            done += 1
            model_name = model["name"]

            if (pdf_rel, page_num, model_name) in existing_keys:
                print(f"  [{done}/{total}] {model_name}... CACHED")
                continue

            print(f"  [{done}/{total}] {model_name}...", end=" ", flush=True)

            try:
                if model["provider"] == "bedrock":
                    md, in_tok, out_tok, elapsed = call_bedrock(model["model_id"], img_bytes)
                elif model["provider"] == "model_studio":
                    md, in_tok, out_tok, elapsed = call_model_studio(model["model_id"], img_bytes)
                else:
                    raise ValueError(f"Unknown provider: {model['provider']}")

                cost = None
                if model["input_cost_per_m"] is not None and model["output_cost_per_m"] is not None:
                    cost = (in_tok * model["input_cost_per_m"] + out_tok * model["output_cost_per_m"]) / 1_000_000

                results.append(PageResult(
                    pdf_name=pdf_rel,
                    page_num=page_num,
                    description=description,
                    model_name=model_name,
                    provider=model["provider"],
                    markdown=md,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    elapsed_seconds=elapsed,
                    cost_usd=cost,
                ))
                print(f"{elapsed:.1f}s | {in_tok:,} in / {out_tok:,} out | {len(md)} chars")

            except Exception as e:
                print(f"ERROR: {e}")
                results.append(PageResult(
                    pdf_name=pdf_rel,
                    page_num=page_num,
                    description=description,
                    model_name=model_name,
                    provider=model["provider"],
                    markdown="",
                    input_tokens=0,
                    output_tokens=0,
                    elapsed_seconds=0,
                    cost_usd=None,
                    error=str(e),
                ))

    # Write JSON results
    json_path.write_text(json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False))
    print(f"\nJSON results: {json_path}")

    # Write HTML report
    html_path = reports_dir / "ocr_comparison.html"
    generate_html_report(results, html_path)
    print(f"HTML report:  {html_path}")


if __name__ == "__main__":
    main()
