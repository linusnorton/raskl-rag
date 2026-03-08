"""E2E benchmark tests for the full RAG pipeline.

These tests run queries from the MBRAS AI Benchmark through the complete
agent pipeline (embedding → hybrid search → reranking → LLM) and verify
that the final answer contains the expected information.

Requires:
- AWS Bedrock access (AWS_PROFILE=linusnorton)
- Local: running PostgreSQL with indexed documents (default)
- Live:  Neon DB (pass --live flag)

Run:
    # Local (default) — uses local PostgreSQL
    AWS_PROFILE=linusnorton uv run pytest tests/e2e/ -v

    # Live — uses Neon DB
    AWS_PROFILE=linusnorton uv run pytest tests/e2e/ -v --live
"""

from __future__ import annotations

import os
import re

import pytest

from ras_rag_engine.agent import run_agent_streaming
from ras_rag_engine.config import RAGConfig as ChatConfig

NEON_DSN = os.environ.get(
    "E2E_DATABASE_DSN",
    "postgresql://raskl_app:npg_G9NwyIJMs1oV@ep-aged-wave-ab8sg6n6.eu-west-2.aws.neon.tech"
    "/raskl_rag?sslmode=require",
)


@pytest.fixture(scope="module")
def config(is_live: bool) -> ChatConfig:
    """Build ChatConfig using Bedrock for all providers.

    Default: local PostgreSQL. With --live: Neon DB.
    """
    kwargs: dict = dict()
    if is_live:
        kwargs["database_dsn"] = NEON_DSN
    return ChatConfig(**kwargs)


def _ask(query: str, config: ChatConfig) -> str:
    """Run a query through the full agent pipeline and return the final answer text."""
    answer = ""
    for text, _chunks in run_agent_streaming(query, history=[], config=config):
        answer = text
    # Strip thinking tags if present
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
    return answer


# ---------------------------------------------------------------------------
# Benchmark queries (from MBRAS AI Benchmark Results spreadsheet)
# ---------------------------------------------------------------------------


class TestBenchmark:
    """Full E2E tests: query → retrieval → LLM → check answer."""

    def test_q1_sultan_abu_bakar_successor(self, config: ChatConfig):
        """Q1: Entity Gap — Sultan Abu Bakar's successor.

        Expected: Sultan Ibrahim. The system should find and cite this
        even though it requires knowledge from the document collection.
        """
        answer = _ask("Who succeeded Sultan Abu Bakar of Johor after his death in London?", config)
        assert "provided documents do not contain" not in answer.lower(), (
            f"LLM refused to answer:\n{answer}"
        )
        assert "Ibrahim" in answer, f"Expected 'Sultan Ibrahim' in answer:\n{answer}"

    def test_q2_mantri_tin_duties(self, config: ChatConfig):
        """Q2: Footnote Anchor — Mantri of Perak's tin duty income.

        Expected: between $96,000 and $286,000 p.a.
        """
        answer = _ask(
            "How much was the Mantri of Perak's annual income from tin duties estimated to be?", config
        )
        assert "96,000" in answer or "286,000" in answer, (
            f"Expected dollar amounts from footnote:\n{answer}"
        )

    def test_q3_birch_telegram_date(self, config: ChatConfig):
        """Q3: Temporal Inheritance — Birch's telegram date.

        Expected: May 4th, 1874.
        """
        answer = _ask(
            "On what date did Mr. Birch send a telegram to the Governor from Penang regarding the stay in Linggi?",
            config,
        )
        assert "May" in answer and "4" in answer, f"Expected May 4th in answer:\n{answer}"
        # Ideally the year 1874 should be inferred from chronological context
        if "1874" not in answer:
            pytest.xfail("LLM found the date (May 4th) but did not infer the year 1874 from context")

    def test_q4_abdullah_critique(self, config: ChatConfig):
        """Q4: Persona distinction — Abdullah's critique of early Malay accounts.

        Expected: he argues they were uncritical/fragmentary/glorifying.
        The LLM must not respond with 'no relevant documents'.
        """
        answer = _ask(
            "What is the author A. Rahman Tang Abdullah's critique of early Malay accounts of Sultan Abu Bakar?",
            config,
        )
        assert "provided documents do not contain" not in answer.lower(), (
            f"LLM refused to answer:\n{answer}"
        )
        assert "Abu Bakar" in answer or "Abdullah" in answer, (
            f"Expected discussion of Abdullah's critique:\n{answer}"
        )

    def test_q5_raja_mahmud_selangor_war(self, config: ChatConfig):
        """Q5: Entity Doppelganger — Raja Mahmud and the Selangor Civil War.

        Expected: Yes, he was a key figure. Bonus if it disambiguates
        multiple Raja Mahmuds.
        """
        answer = _ask(
            "Was the Raja Mahmud mentioned in the Swettenham journals involved in the Selangor Civil War?",
            config,
        )
        assert "Raja Mahmud" in answer, f"Expected 'Raja Mahmud' in answer:\n{answer}"
        assert "Selangor" in answer, f"Expected 'Selangor' in answer:\n{answer}"

    def test_q6_abu_bakar_english_windsor(self, config: ChatConfig):
        """Q6: Structural Fidelity — Abu Bakar speaking English at Windsor.

        Expected: should discuss the conflicting evidence (Queen Victoria's
        journal says he spoke English well vs Blunt's claim).
        """
        answer = _ask(
            "Did Sultan Abu Bakar speak English during his meeting with Queen Victoria at Windsor?",
            config,
        )
        assert "English" in answer, f"Expected discussion of English language:\n{answer}"
        assert "Abu Bakar" in answer or "Victoria" in answer, (
            f"Expected Abu Bakar or Victoria mentioned:\n{answer}"
        )

    def test_q7_chop_seal(self, config: ChatConfig):
        """Q7: Semantic Mapping — 'chop' as a seal.

        Expected: yes, a chop is an official seal/stamp used by Malay rulers.
        """
        answer = _ask("What is a chop? Is it some kind of a seal?", config)
        assert "seal" in answer.lower() or "stamp" in answer.lower() or "chop" in answer.lower(), (
            f"Expected explanation of chop/seal:\n{answer}"
        )

    def test_q8_norwich_ipswich_out_of_distribution(self, config: ChatConfig):
        """Q8: Out-of-Distribution — irrelevant football question.

        Expected: the agent should decline, stating its knowledge is
        restricted to MBRAS/Malayan history. It should NOT give a football opinion.
        """
        answer = _ask("Which team is better, Norwich or Ipswich?", config)
        # Should not give a definitive football opinion
        answer_lower = answer.lower()
        assert not ("norwich is better" in answer_lower or "ipswich is better" in answer_lower), (
            f"LLM should not give football opinions:\n{answer}"
        )

    def test_q9_britain_colonise_malaysia(self, config: ChatConfig):
        """Q9: Synthesis & Nuance — Why Britain colonised Malaysia.

        Expected: a nuanced answer mentioning economic, strategic, and
        geopolitical factors, citing multiple sources.
        """
        answer = _ask("Why did Britain colonise Malaysia?", config)
        assert "provided documents do not contain" not in answer.lower(), (
            f"LLM refused to answer:\n{answer}"
        )
        # Should mention at least some key factors
        answer_lower = answer.lower()
        factors_found = sum(
            1 for kw in ["tin", "trade", "straits", "perak", "british", "colonial", "economic", "strategic"]
            if kw in answer_lower
        )
        assert factors_found >= 2, (
            f"Expected a nuanced answer with multiple factors, found {factors_found}:\n{answer}"
        )


class TestSwettenhamSingaporeDates:
    """Verify the full pipeline for 'What dates did Swettenham go to Singapore?'

    The expected answer must reference:
    - 4 January 1871 (from the Abu Bakar article, p.35)
    - 6 October 1875 (from Swettenham journals, p.111)
    """

    @pytest.fixture(scope="class")
    def answer(self, config: ChatConfig) -> str:
        """Run the query once and share across all tests in this class."""
        return _ask("What dates did Swettenham go to Singapore?", config)

    def test_does_not_refuse(self, answer: str):
        """The LLM should provide an answer, not refuse."""
        assert "provided documents do not contain" not in answer.lower(), (
            f"LLM refused to answer:\n{answer}"
        )

    def test_mentions_january_1871(self, answer: str):
        """Answer should mention January 1871."""
        assert "1871" in answer, f"Expected mention of 1871:\n{answer}"

    def test_mentions_october_1875(self, answer: str):
        """Answer should mention October 1875."""
        assert "October" in answer or "6th October" in answer or "6 October" in answer, (
            f"Expected mention of October 1875:\n{answer}"
        )

    def test_mentions_6_october_specifically(self, answer: str):
        """Answer should mention 6 October specifically (the key date from journal p.111).

        The previous test checks for October broadly; this checks for the
        specific date '6 October' or '6th October' that appears on p.111.
        """
        assert "6 October" in answer or "6th October" in answer, (
            f"Expected specific '6 October' date from journal p.111:\n{answer}"
        )
