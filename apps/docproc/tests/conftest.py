"""Test fixtures for docproc tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(REPO_ROOT / ".env")
DOCS_DIR = REPO_ROOT / "docs"
DATA_OUT_DIR = REPO_ROOT / "data" / "out"
SWETTENHAM_PDF = DOCS_DIR / "messy" / "Swettenham Journal 1874-1876.pdf"
ABDULLAH_PDF = DOCS_DIR / "clean" / "Abdullah (2011) JMBRAS 84(1), 1-22.pdf"
AZNAN_PDF = DOCS_DIR / "clean" / "Aznan (2020) JMBRAS 93(1), 119-131.pdf"
ABDULLAH_DOC_ID = "abdullah-2011-jmbras-84-1-1-22-5df957674dec"


@pytest.fixture
def swettenham_pdf() -> Path:
    """Path to the Swettenham Journal PDF."""
    if not SWETTENHAM_PDF.exists():
        pytest.skip(f"PDF not found: {SWETTENHAM_PDF}")
    return SWETTENHAM_PDF


@pytest.fixture
def abdullah_pdf() -> Path:
    """Path to the Abdullah PDF."""
    if not ABDULLAH_PDF.exists():
        pytest.skip(f"PDF not found: {ABDULLAH_PDF}")
    return ABDULLAH_PDF


@pytest.fixture
def aznan_pdf() -> Path:
    """Path to the Aznan PDF."""
    if not AZNAN_PDF.exists():
        pytest.skip(f"PDF not found: {AZNAN_PDF}")
    return AZNAN_PDF


@pytest.fixture
def abdullah_data_dir() -> Path:
    """Path to the pre-processed Abdullah output directory."""
    d = DATA_OUT_DIR / ABDULLAH_DOC_ID
    if not d.exists():
        pytest.skip(f"Processed data not found: {d}")
    return d


@pytest.fixture
def tmp_out_dir(tmp_path: Path) -> Path:
    """Temporary output directory for test runs."""
    out = tmp_path / "data"
    out.mkdir()
    return out
