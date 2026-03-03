"""Local reranking providers: Qwen3-Reranker (causal LM) and CrossEncoder."""

from __future__ import annotations

import logging

from .base import RerankProvider

log = logging.getLogger(__name__)

_qwen3_tokenizer = None
_qwen3_model = None
_yes_token_id: int | None = None
_no_token_id: int | None = None
_cross_encoder = None


class Qwen3RerankProvider(RerankProvider):
    def __init__(self, model_path: str, device: str, instruction: str):
        self.model_path = model_path
        self.device = device
        self.instruction = instruction

    def _get_model(self):
        global _qwen3_tokenizer, _qwen3_model, _yes_token_id, _no_token_id
        if _qwen3_model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            log.info("Loading Qwen3 reranker %s on %s (BF16)...", self.model_path, self.device)
            _qwen3_tokenizer = AutoTokenizer.from_pretrained(self.model_path, padding_side="left")
            _qwen3_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
            )
            _qwen3_model.eval()
            _yes_token_id = _qwen3_tokenizer.convert_tokens_to_ids("yes")
            _no_token_id = _qwen3_tokenizer.convert_tokens_to_ids("no")
            log.info("Qwen3 reranker loaded (yes=%d, no=%d).", _yes_token_id, _no_token_id)
        return _qwen3_tokenizer, _qwen3_model

    def rerank(self, query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]:
        import torch

        tokenizer, model = self._get_model()

        prompts = [
            f"<Instruct>{self.instruction}</Instruct>\n"
            f"<Query>{query}</Query>\n"
            f"<Document>{doc}</Document>\n"
            "<think>\n\n</think>\n"
            for doc in documents
        ]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)

        token_ids = torch.tensor([_no_token_id, _yes_token_id])
        last_logits = outputs.logits[:, -1, :]
        pair_logits = last_logits[:, token_ids]
        probs = torch.softmax(pair_logits, dim=-1)
        scores = probs[:, 1].tolist()

        indexed_scores = [(i, s) for i, s in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        return indexed_scores[:top_k]


class CrossEncoderRerankProvider(RerankProvider):
    def __init__(self, model_path: str, device: str):
        self.model_path = model_path
        self.device = device

    def _get_model(self):
        global _cross_encoder
        if _cross_encoder is None:
            from sentence_transformers import CrossEncoder

            log.info("Loading CrossEncoder reranker %s on %s...", self.model_path, self.device)
            _cross_encoder = CrossEncoder(self.model_path, device=self.device)
            log.info("CrossEncoder reranker loaded.")
        return _cross_encoder

    def rerank(self, query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]:
        model = self._get_model()
        pairs = [(query, doc) for doc in documents]
        scores = model.predict(pairs).tolist()

        indexed_scores = [(i, s) for i, s in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        return indexed_scores[:top_k]
