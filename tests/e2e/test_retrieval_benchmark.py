import os
import re
import time
import json
import uuid
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
import pytest

from ras_rag_engine.agent import run_agent_streaming, SYSTEM_PROMPT
from ras_rag_engine.config import RAGConfig as ChatConfig

RESULTS_DIR = Path("tests/results")
# Unique ID for the entire batch execution
BATCH_ID = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
BATCH_FILE = RESULTS_DIR / f"{BATCH_ID}.jsonl"

@dataclass
class BenchmarkMetrics:
    test_name: str
    query: str
    expected_answer: str          # Factual baseline
    eval_goal: str               # Behavioral baseline
    response: str
    status: str
    score: float
    latency_sec: float
    input_tokens_est: int
    output_tokens_est: int
    config: dict
    timestamp: str = datetime.utcnow().isoformat()

def _calculate_score(data: dict) -> float:
    """
    Scoring Rubric (Max 100):
    - Status: 60 pts (Success vs Refusal/Error)
    - Latency: 20 pts (Penalty if > 20s)
    - Efficiency: 20 pts (Penalty if > 15k tokens)
    """
    if data["status"] != "success":
        return 0.0
    
    score = 60.0
    # Latency: -2 pts for every 5s over 20s
    latency_penalty = max(0, (data["latency"] - 20) // 5) * 2
    # Tokens: -2 pts for every 5k tokens over 15k
    token_penalty = max(0, (data["tokens"] - 15000) // 5000) * 2
    
    return max(0, score + 20 - latency_penalty + 20 - token_penalty)

def _log_to_batch(metrics: BenchmarkMetrics):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(BATCH_FILE, "a") as f:
        f.write(json.dumps(asdict(metrics)) + "\n")

def _ask(test_name: str, query: str, expected: str, goal: str, config: ChatConfig) -> str:
    start_time = time.perf_counter()
    answer, chunks = "", []
    
    try:
        for text, retrieved_chunks in run_agent_streaming(query, history=[], config=config):
            answer, chunks = text, retrieved_chunks
        status = "success" if "provided documents do not contain" not in answer.lower() else "refusal"
    except Exception as e:
        answer, status = str(e), "error"

    latency = time.perf_counter() - start_time
    in_tokens = int((len(query) + sum(len(c.text) for c in chunks)) / 3.5) + 100

    metrics = BenchmarkMetrics(
        test_name=test_name,
        query=query,
        expected_answer=expected,
        eval_goal=goal,
        response=answer,
        status=status,
        score=_calculate_score({"status": status, "latency": latency, "tokens": in_tokens}),
        latency_sec=round(latency, 2),
        input_tokens_est=in_tokens,
        output_tokens_est=int(len(answer) / 3.5),
        config=config.model_dump(include={'retrieval_top_k', 'llm_temperature', 'rerank_candidates'})
    )
    _log_to_batch(metrics)
    return answer

class TestBenchmark:
    @pytest.fixture(scope="class")
    def cfg(self, is_live: bool):
        dsn = os.environ.get("E2E_DATABASE_DSN")
        if is_live and dsn:
            return ChatConfig(database_dsn=dsn)
        # Falling back to the defaults defined in RAGConfig
        return ChatConfig()

    def test_q1_basic_retrieval(self, cfg):
        query = "How much was the Mantri of Perak’s annual income from tin duties estimated to be?"
        goal = "Test ability to find a needle in the haystick and cite it correctly."
        expected = "Between $96,000 and $286,000"
        _ask("test_q1", query, expected, goal, cfg)

    def test_q2_catalog_hallucination(self, cfg):
        query = "List me all of the articles in JMBRAS"
        goal = "Agent must use browse_corpus and avoid hallucinating a full list if truncated."
        expected = "A summary or list of available early volumes/articles via tool use."
        _ask("test_q2", query, expected, goal, cfg)

    def test_q3_chronological_superlative(self, cfg):
        query = "Who was the first British person to visit Malaysia?"
        goal = "The agent must prioritize the earliest dated record (e.g., 16th-century explorers) rather than the most semantically relevant 19th-century accounts often favored by the current vector search."
        expected = "Ralph Fitch was the first British visitor to Malaysia."
        _ask("test_q3", query, expected, goal, cfg)

    def test_q4_primary_source(self, cfg):
        query = "Who did Isabella Bird meet in Penang?"
        goal = "Test whether the relevant primary source is retrieved, rather than purely relying on the inaccurate (incomplete) MBRAS sources."
        expected = "Isabella Bird met Bloomfield Douglas, Mr. Low, Mr. Maxwell, and Governor Sir W. Robinson."
        _ask("test_q4", query, expected, goal, cfg)

    def test_q5_false_premise(self, cfg):
        query = "The MBRAS Index mentions a 1904 report on 'Romanized Malay Spelling.' Can you summarize their specific rules for the 'ch' sound?"
        goal = "Test whether the agent checks the corpus and successfully determines that there is no 1904 report on 'Romanized Malay Spellings"
        expected = "Clarify that there is no such report, and find relevant documents related to the question."
        _ask("test_q5", query, expected, goal, cfg)

    def test_q6_author_lookup(self, cfg):
        query = "List the titles of all articles authored by Annandale, N. in the JMBRAS collection."
        goal = "Check that it can retrieve articles written by a specific author"
        expected = "Barnacles from Deep-Sea Telegraph Cables in the Malay, published in 1916 in Volume 74, Part 74, pages 281–302."
        _ask("test_q6", query, expected, goal, cfg)

    def test_q7_comparative_answer(self, cfg):
        query = "Which is older, Ampang or Gombak?"
        goal = "Compare the dataset for two different knowledge fields and synthesize an accurate answer."
        expected = "The MBRAS corpus suggests that Ampang was first settled by Chinese miners in 1857, while Gombak was settled earlier."
        _ask("test_q7", query, expected, goal, cfg)

    def test_q8_complex_query(self, cfg):
        query = "What MBRAS articles tell the story of the Sultan Abdul Samad building, or Selangor Secretariat, and can you display it to me in a clear, easy to understand timeline?"
        goal = "Measure the performance of a complex question, which requires many chunks to answers comprehensively."
        expected = "A timeline of key events from 1880-1897 relevant to the construction of the building."
        _ask("test_q8", query, expected, goal, cfg)