"""Cross-encoder reranker.

Scores (task, lesson) pairs jointly, so it sees the words of both at once
instead of comparing two fixed vectors. The raw logit orders candidates and
also decides abstention: on the frozen eval set the stock checkpoint scores
every no-answer task below -10 and every relevant row above -1.
"""
from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

STOCK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class CrossEncoderRanker:
    def __init__(self, model: str | Path = STOCK_MODEL, device: str | None = None, max_length: int = 384):
        self.model_name = str(model)
        self.device = device or pick_device()
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).eval().to(self.device)

    @torch.no_grad()
    def score(self, task: str, lessons: list[str]) -> list[float]:
        if not lessons:
            return []
        enc = self.tokenizer([task] * len(lessons), lessons, padding=True, truncation=True,
                             max_length=self.max_length, return_tensors="pt").to(self.device)
        return self.model(**enc).logits.squeeze(-1).float().cpu().tolist()

    def rank(self, task: str, candidates: list[dict], *, k: int, min_score: float) -> list[dict]:
        scores = self.score(task, [c["lesson"] for c in candidates])
        kept = [dict(c, score=s) for c, s in zip(candidates, scores) if s >= min_score]
        kept.sort(key=lambda c: -c["score"])
        return kept[:k]
