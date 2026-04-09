import os
import re
import time
import json
import uuid
import logging
from unittest.mock import patch
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
    tool_rounds: int
    tool_calls: list
    retrieved_chunks: list
    config: dict
    timestamp: str = datetime.utcnow().isoformat()

def _calculate_score(data: dict) -> float:
    """
    Conservative Scoring Rubric (Max 100):
    - Status: 40 pts (Success vs Refusal/Error)
    - Latency: 30 pts (Target < 8s. Harsh decay until 20s)
    - Efficiency: 20 pts (Target < 4k tokens. Harsh decay until 12k)
    - Rounds: 10 pts (Target: 1 round. Penalty for agent 'loops')
    """
    if data["status"] != "success":
        return 0.0
    
    # 1. Latency (30 pts)
    # Full points if < 8s. 0 points if > 20s.
    latency = data.get("latency", 0)
    latency_score = 30.0
    if latency > 8:
        # Subtract 2.5 pts for every second over 8s
        latency_score -= (latency - 8) * 2.5
    
    # 2. Token Efficiency (20 pts)
    # Full points if < 4,000 tokens. 0 points if > 12,000.
    tokens = data.get("tokens", 0)
    token_score = 20.0
    if tokens > 4000:
        # Subtract 2.5 pts for every 1,000 tokens over 4k
        token_score -= ((tokens - 4000) / 1000) * 2.5
        
    # 3. Agent Rounds (10 pts)
    # Full points for 1 round (retrieval + answer). 
    # -2.5 pts for every extra tool round.
    rounds = data.get("rounds", 1)
    round_score = max(0, 10.0 - (rounds - 1) * 2.5)
    
    total = 40.0 + max(0, latency_score) + max(0, token_score) + round_score
    return round(total, 2)

def _log_to_batch(metrics: BenchmarkMetrics):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(BATCH_FILE, "a") as f:
        f.write(json.dumps(asdict(metrics)) + "\n")

import time
import logging
from unittest.mock import patch

# Configure logging to show up in pytest output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _ask(query_id: str, query: str, expected: str, goal: str, config: ChatConfig) -> str:
    """Run a query through the full agent pipeline and log metrics + retrieved documents."""
    metrics_tracker = {
        "start_time": time.perf_counter(),
        "tool_calls": [],
        "llm_steps": 0,
        "status": "success",
        "input_tokens": 0,    # Initialize token counters
        "output_tokens": 0
    }
    all_retrieved_chunks = []

    # 1. Wrapper to track tool execution (existing logic)
    from ras_rag_engine.agent import execute_tool_call as real_execute
    def tracked_execute(name, args, cfg, start_index=1):
        t0 = time.perf_counter()
        try:
            return real_execute(name, args, cfg, start_index=start_index)
        finally:
            duration = time.perf_counter() - t0
            metrics_tracker["tool_calls"].append({
            "name": name, 
            "args": args, 
            "duration": duration
        })

    # 2. Wrapper to track agent rounds and capture tokens
    from ras_rag_engine.agent import get_llm_provider as real_get_llm
    def tracked_get_llm(cfg):
        llm = real_get_llm(cfg)
        original_chat = llm.chat_completion
        original_stream = llm.chat_completion_stream

        # Wrapper for tool-calling rounds (non-streaming)
        def tracked_chat(messages, **kwargs):
            metrics_tracker["llm_steps"] += 1
            # Capture input tokens for the tool round prompt
            metrics_tracker["input_tokens"] += llm.count_tokens(messages, tools=kwargs.get("tools"))
            
            result = original_chat(messages, **kwargs)
            
            # Capture output tokens for the tool call/reasoning response
            content = result.get("content") or ""
            metrics_tracker["output_tokens"] += llm.count_tokens([{"role": "assistant", "content": content}])
            return result

        # Wrapper for the final answer (streaming)
        def tracked_stream(messages, **kwargs):
            # Capture input tokens for the final prompt
            metrics_tracker["input_tokens"] += llm.count_tokens(messages, tools=kwargs.get("tools"))
            yield from original_stream(messages, **kwargs)

        llm.chat_completion = tracked_chat
        llm.chat_completion_stream = tracked_stream
        return llm

    answer = ""
    try:
        with patch("ras_rag_engine.agent.execute_tool_call", side_effect=tracked_execute), \
             patch("ras_rag_engine.agent.get_llm_provider", side_effect=tracked_get_llm):
            
            for text, chunks in run_agent_streaming(query, history=[], config=config):
                answer = text # 'text' is cumulative in your streaming implementation
                all_retrieved_chunks = chunks
        
        # Capture final output tokens from the streamed response
        # We use a fresh provider instance for the calculation
        from ras_rag_engine.providers import get_llm_provider as get_llm
        metrics_tracker["output_tokens"] += get_llm(config).count_tokens([{"role": "assistant", "content": answer}])

    except Exception as e:
        metrics_tracker["status"] = f"error: {str(e)}"
        answer = f"Error: {e}"

    total_latency = time.perf_counter() - metrics_tracker["start_time"]
    clean_answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

    # Log to Batch File
    bench_metrics = BenchmarkMetrics(
        test_name=query_id,
        query=query,
        expected_answer=expected,
        eval_goal=goal,
        response=clean_answer,
        status=metrics_tracker["status"],
        score=0.0,
        latency_sec=total_latency,
        tool_rounds=metrics_tracker["llm_steps"],
        tool_calls=metrics_tracker["tool_calls"],
        retrieved_chunks=[{"doc": c.source_filename, "score": c.score, "page": c.start_page} for c in all_retrieved_chunks],
        input_tokens_est=metrics_tracker["input_tokens"],  # Pass the tracked values
        output_tokens_est=metrics_tracker["output_tokens"],
        config={
            "max_tokens": config.llm_max_tokens,
            "context_window": config.llm_context_window,
            "temperature": config.llm_temperature,
            "rerank": config.rerank_enabled,
            "rerank_candidates": config.rerank_candidates, 
            "thinking_budget": config.llm_thinking_budget, 
            "top_k": config.retrieval_top_k,
            "diversity_max_per_doc": config.diversity_max_per_doc,
            "prompt_caching": config.llm_prompt_caching
        }
    )

    # Update score calculation to use the real token sum
    bench_metrics.score = _calculate_score({
        "status": bench_metrics.status,
        "latency": bench_metrics.latency_sec,
        "tokens": bench_metrics.input_tokens_est + bench_metrics.output_tokens_est
    })
    
    _log_to_batch(bench_metrics)
    return clean_answer

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