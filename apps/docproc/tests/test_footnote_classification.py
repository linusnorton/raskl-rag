"""Tests for footnote type classification."""

from __future__ import annotations

import pytest

from ras_docproc.pipeline.detect_footnotes import classify_footnote_type


class TestCitationFootnotes:
    @pytest.mark.parametrize(
        "text",
        [
            "35 Weld to Kimberley, 23 November 1880, CO 273/105.",
            "21 Ibid.",
            "3 Ibid.: 45-6.",
            "19 Gullick (1992: 246-8).",
            "6 Na (1896: 59, 63).",
            "2 Heng, Kwa and Tan (2009); Hack, Margolin and Delaye (2010).",
            "15 Malaya Tribune , 16 December 1918.",
            "1 http://en.wikipedia.org/wiki/Demographics_of_Singapore, accessed 14 March 2013.",
        ],
    )
    def test_citation(self, text: str):
        assert classify_footnote_type(text) == "citation"


class TestExplanatoryFootnotes:
    @pytest.mark.parametrize(
        "text",
        [
            "30 Head of sub-district.",
            "3 'Syed' and 'Sharif' are Arabic honorific titles used to denote descendants of the Prophet Muhammad.",
            "2 Perak is a state in modern Malaysia. The sultanate was founded in 1528.",
        ],
    )
    def test_explanatory(self, text: str):
        assert classify_footnote_type(text) == "explanatory"


class TestMixedFootnotes:
    @pytest.mark.parametrize(
        "text",
        [
            "25 Queen Victoria's journal, 12 July 1878. Bertie's medal was merely the informal phrase used by the Queen to describe the award.",
            "17 Ibid. This dual proposal reflected to some degree the class divisions within the Chinese community.",
        ],
    )
    def test_mixed(self, text: str):
        assert classify_footnote_type(text) == "mixed"
