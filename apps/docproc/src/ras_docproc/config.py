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
    boilerplate_threshold: float = Field(default=0.3, description="Fraction of pages a line must appear on to be boilerplate")

    # Zone thresholds (as fraction of page height)
    footnote_zone_top: float = Field(default=0.72, description="Top boundary of footnote zone (fraction of page height)")
    footnote_zone_bottom: float = Field(default=1.0, description="Bottom boundary of footnote zone")
    header_zone_bottom: float = Field(default=0.10, description="Bottom boundary of header zone")
    footer_zone_top: float = Field(default=0.80, description="Top boundary of footer zone")

    # Language detection
    min_lang_chars: int = Field(default=30, description="Minimum characters to attempt language detection")
    lang_set: list[str] = Field(default=["en", "ms", "zh", "ar"], description="Languages to detect")

    # Extraction backend
    extraction_backend: str = Field(default="docling", description="Extraction backend: 'docling', 'deepseek', or 'qwen3vl'")
    vllm_base_url: str = Field(default="http://localhost:8000", description="Base URL for vLLM server")
    vllm_model: str = Field(default="deepseek-ai/DeepSeek-OCR", description="Model name for DeepSeek-OCR")
    deepseek_dpi: int = Field(default=200, description="DPI for page rendering when using DeepSeek backend")
    deepseek_max_tokens: int = Field(default=4000, description="Max tokens per page for DeepSeek OCR")

    # Qwen3 VL (Bedrock)
    bedrock_region: str = Field(default="eu-west-2", description="AWS region for Bedrock API calls")
    bedrock_ocr_model_id: str = Field(default="qwen.qwen3-vl-235b-a22b", description="Bedrock model ID for Qwen3 VL OCR")
    qwen3vl_dpi: int = Field(default=300, description="DPI for page rendering when using Qwen3 VL backend")
    qwen3vl_max_tokens: int = Field(default=4096, description="Max tokens per page for Qwen3 VL OCR")
    qwen3vl_max_workers: int = Field(default=10, description="Max parallel Bedrock calls for Qwen3 VL")

    # OCR (not enabled by default)
    ocr_enabled: bool = False
    ocr_min_chars: int = Field(default=200, description="Minimum chars on page before OCR kicks in")

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
