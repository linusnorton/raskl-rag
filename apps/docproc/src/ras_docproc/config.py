"""Pipeline configuration via Pydantic Settings."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class PipelineConfig(BaseSettings):
    """Configuration for the document processing pipeline."""

    model_config = {"env_prefix": "DOCPROC_"}

    # Input/output
    pdf_path: Path = Path(".")
    out_dir: Path = Path("data")
    max_pages: int | None = None
    page_range: str | None = None  # e.g. "1-10" or "5,10,15"
    force: bool = False

    # Boilerplate detection
    boilerplate_threshold: float = Field(default=0.3, description="Fraction of pages for boilerplate")

    # Zone thresholds (as fraction of page height)
    footnote_zone_top: float = Field(default=0.72, description="Top of footnote zone (page fraction)")
    footnote_zone_bottom: float = Field(default=1.0, description="Bottom boundary of footnote zone")
    header_zone_bottom: float = Field(default=0.10, description="Bottom boundary of header zone")
    footer_zone_top: float = Field(default=0.80, description="Top boundary of footer zone")

    # Language detection
    min_lang_chars: int = Field(default=30, description="Minimum characters to attempt language detection")
    lang_set: list[str] = Field(default=["en", "ms", "zh", "ar"], description="Languages to detect")

    # Qwen3 VL (Bedrock)
    bedrock_region: str = Field(default="eu-west-2", description="AWS region for Bedrock API calls")
    bedrock_ocr_model_id: str = Field(default="qwen.qwen3-vl-235b-a22b", description="Bedrock model ID for OCR")
    qwen3vl_dpi: int = Field(default=300, description="DPI for page rendering when using Qwen3 VL backend")
    qwen3vl_max_tokens: int = Field(default=4096, description="Max tokens per page for Qwen3 VL OCR")
    qwen3vl_max_workers: int = Field(default=20, description="Max parallel Bedrock calls for Qwen3 VL")
    qwen3vl_system_prompt: str = Field(
        default="""\
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
- Do NOT describe images or figures — skip them entirely.
- Do NOT add any commentary, explanation, or preamble. Output ONLY the Markdown text.""",
        description="System prompt for Qwen3 VL OCR",
    )

    # Versioning
    extraction_version: str = "0.1.0"

    def parse_page_range(self) -> list[int] | None:
        """Parse page_range string into list of 0-indexed page numbers."""
        if not self.page_range:
            return None
        pages: list[int] = []
        for part in self.page_range.split(","):
            part = part.strip()
            if "-" in part:
                start, end = part.split("-", 1)
                pages.extend(range(int(start) - 1, int(end)))
            else:
                pages.append(int(part) - 1)
        return sorted(set(pages))
