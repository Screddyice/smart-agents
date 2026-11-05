"""Cross-encoder rerank stage for /advise, with a fallback that costs nothing.

The stock cross-encoder/ms-marco-MiniLM-L-6-v2 scores (task, lesson) pairs
jointly. On the frozen eval set it lifts hit@3 from 0.900 to 1.000 and its
raw logit separates tasks the corpus has never covered from ones it has
(BASELINE.md, Phase 2). It also brings near-duplicate rows back, because
twins score alike, so a hard cosine dedupe runs after it.

Everything here is optional at runtime: disabled by env, missing torch, a
crashed model, or a slow call all return None, and /advise falls back to the
Phase 1 ordering. Defaults (threshold -8, dedupe at cosine 0.90, pool 12) come
from the v1+v2 recipe sweep in BASELINE.md: dedupe at 0.85 removed labeled
paraphrases and cost hit@3 on v2; 0.90 still drops verbatim twins. The model loads lazily on the first call and runs on
RERANK_THREADS cores so Ollama and the corpus services keep the rest.

Runs flat out of /opt/corpus/app, so it imports siblings only; the deploy
script rewrites the corpus_api import below the way it does for main.py.
"""
from __future__ import annotations

import concurrent.futures
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass

from corpus_api.rank import _cosine

log = logging.getLogger("corpus-api.rerank")

Scorer = Callable[[str, list[str]], list[float]]


@dataclass(frozen=True)
class Settings:
    enabled: bool
    model: str
    min_score: float
    pool: int
    budget_s: float
    threads: int
    dedupe_cos: float


def settings_from_env() -> Settings:
    return Settings(
        enabled=os.environ.get("RERANK_ENABLED", "1") == "1",
        model=os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        min_score=float(os.environ.get("RERANK_MIN_SCORE", "-8.0")),
        pool=int(os.environ.get("RERANK_POOL", "12")),
        budget_s=float(os.environ.get("RERANK_BUDGET_S", "1.5")),
        threads=int(os.environ.get("RERANK_THREADS", "2")),
        dedupe_cos=float(os.environ.get("RERANK_DEDUPE_COS", "0.90")),
    )


def dedupe(rows: list[dict], *, k: int, max_cos: float) -> list[dict]:
    """Keep rows in order, skipping any above max_cos to a row already kept."""
    kept: list[dict] = []
    for r in rows:
        if all(_cosine(r["embedding"], s["embedding"]) <= max_cos for s in kept):
            kept.append(r)
        if len(kept) == k:
            break
    return kept


def _load_cross_encoder(model: str, threads: int) -> Scorer:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(model)
    net = AutoModelForSequenceClassification.from_pretrained(model).eval()

    @torch.no_grad()
    def score(task: str, lessons: list[str]) -> list[float]:
        enc = tok([task] * len(lessons), lessons, padding=True, truncation=True, max_length=384, return_tensors="pt")
        return net(**enc).logits.squeeze(-1).float().tolist()

    from smart_agents.rerank.sweep_threshold import sha256_of_dir
    from pathlib import Path
    score.model_sha256 = sha256_of_dir(Path(model))
    return score


class Reranker:
    def __init__(self, settings: Settings, scorer: Scorer | None = None):
        self.settings = settings
        self._scorer = scorer
        # One worker so the budget can be enforced without cancelling torch.
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def _score(self, task: str, lessons: list[str]) -> list[float]:
        if self._scorer is None:
            self._scorer = _load_cross_encoder(self.settings.model, self.settings.threads)
        return self._scorer(task, lessons)

    def warm(self) -> bool:
        """Load the model now so the first request does not pay for it.

        The first live deploy fell back on every early call: the lazy load ran
        inside the 1.5 s budget and lost. Called from the app's startup hook;
        a failure here is logged and the stage stays in fallback.
        """
        if not self.settings.enabled or self._scorer is not None:
            return self._scorer is not None
        try:
            self._scorer = _load_cross_encoder(self.settings.model, self.settings.threads)
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("rerank model failed to load, stage stays off: %s", exc)
            return False

    def rerank(self, task: str, candidates: list[dict], *, k: int) -> list[dict] | None:
        s = self.settings
        if not s.enabled or not candidates:
            return None
        rows = candidates[: s.pool]
        try:
            fut = self._pool.submit(self._score, task, [r["lesson"] for r in rows])
            scores = fut.result(timeout=s.budget_s)
        except concurrent.futures.TimeoutError:
            log.warning("rerank exceeded %.2fs budget, falling back", s.budget_s)
            return None
        except Exception as exc:  # noqa: BLE001
            log.warning("rerank failed, falling back: %s", exc)
            return None
        scored = [dict(r, score=float(sc)) for r, sc in zip(rows, scores) if sc >= s.min_score]
        scored.sort(key=lambda r: -r["score"])
        return dedupe(scored, k=k, max_cos=s.dedupe_cos)
