import os
import re
import time
import json
import pytest
from datetime import datetime
from dataclasses import dataclass, asdict
from ras_rag_engine.agent import run_agent_streaming, SYSTEM_PROMPT  # Import SYSTEM_PROMPT
from ras_rag_engine.config import RAGConfig as ChatConfig

NEON_DSN = os.environ.get(
    "E2E_DATABASE_DSN",
    "postgresql://raskl_app:npg_G9NwyIJMs1oV@ep-aged-wave-ab8sg6n6.eu-west-2.aws.neon.tech/raskl_rag?sslmode=require",
)

LOG_FILE = "benchmark_results.jsonl"

@dataclass
class BenchmarkMetrics:
    timestamp: str
    test_name: str
    query: str
    response: str                 # Log the full response
    status: str
    latency_sec: float
    input_tokens_est: int
    output_tokens_est: int
    chunks_retrieved: int
    # Config Tweaks
    retrieval_top_k: int
    diversity_max_per_doc: int
    llm_thinking_budget: int
    rerank_candidates: int
    llm_max_tokens: int
    llm_context_window: int
    llm_temperature: float
    system_prompt: str            # Log the active system prompt
    error: str = None

def _log_metric(metrics: BenchmarkMetrics):
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(asdict(metrics)) + "\n")

@pytest.fixture(scope="module")
def config(is_live: bool) -> ChatConfig:
    kwargs: dict = {}
    if is_live:
        kwargs["database_dsn"] = NEON_DSN
    return ChatConfig(**kwargs)

def _ask(test_name: str, query: str, config: ChatConfig) -> str:
    start_time = time.perf_counter()
    full_answer_with_thinking = ""
    all_chunks = []
    error_msg = None
    
    try:
        # run_agent_streaming yields (partial_text, chunks)
        for text, chunks in run_agent_streaming(query, history=[], config=config):
            full_answer_with_thinking = text
            all_chunks = chunks
    except Exception as e:
        error_msg = str(e)
        status = "error"
    else:
        status = "success" if "provided documents do not contain" not in full_answer_with_thinking.lower() else "refusal"

    end_time = time.perf_counter()
    latency = end_time - start_time
    
    # Token estimation logic
    context_chars = sum(len(c.text) for c in all_chunks)
    input_tokens = int((len(query) + context_chars) / 3.5) + 100 
    output_tokens = int(len(full_answer_with_thinking) / 3.5)

    _log_metric(BenchmarkMetrics(
        timestamp=datetime.utcnow().isoformat(),
        test_name=test_name,
        query=query,
        response=full_answer_with_thinking,
        status=status,
        latency_sec=round(latency, 2),
        input_tokens_est=input_tokens,
        output_tokens_est=output_tokens,
        chunks_retrieved=len(all_chunks),
        # Extracting current config values
        retrieval_top_k=config.retrieval_top_k,
        diversity_max_per_doc=config.diversity_max_per_doc,
        llm_thinking_budget=config.llm_thinking_budget,
        rerank_candidates=config.rerank_candidates,
        llm_max_tokens=config.llm_max_tokens,
        llm_context_window=config.llm_context_window,
        llm_temperature=config.llm_temperature,
        system_prompt=SYSTEM_PROMPT,
        error=error_msg
    ))

    if error_msg:
        raise Exception(f"Query failed: {error_msg}")

    # Strip tags for the assertion check, but they remain in the log
    return re.sub(r"<think>.*?</think>", "", full_answer_with_thinking, flags=re.DOTALL).strip()

class TestBenchmark:
    """Historical and Corpus-wide Benchmark Queries."""

    def test_q1_mantri_income(self, config):
        answer = _ask("test_q1", "How much was the Mantri of Perak’s annual income from tin duties estimated to be?", config)
        assert any(x in answer for x in ["96,000", "286,000"])

    def test_q2_list_all_articles(self, config):
        answer = _ask("test_q2", "List me all of the articles in JMBRAS", config)
        assert len(answer) > 100

    def test_q3_first_british_visitor(self, config):
        answer = _ask("test_q3", "Who was the first British person to visit Malaysia?", config)
        assert "British" in answer

    def test_q4_isabella_bird_meeting(self, config):
        answer = _ask("test_q4", "Who did Isabella Bird meet in Penang?", config)
        assert "Penang" in answer

    def test_q5_ch_sound_rules(self, config):
        query = "The MBRAS Index mentions a 1904 report on 'Romanized Malay Spelling.' Can you summarize their specific rules for the 'ch' sound?"
        answer = _ask("test_q5", query, config)
        assert "ch" in answer.lower()

    def test_q6_annandale_articles(self, config):
        answer = _ask("test_q6", "List the titles of all articles authored by Annandale, N. in the JMBRAS collection.", config)
        assert "Annandale" in answer

    def test_q7_ampang_gombak_age(self, config):
        answer = _ask("test_q7", "Which is older, Ampang or Gombak?", config)
        assert "Ampang" in answer or "Gombak" in answer