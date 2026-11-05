"""Score any ranker against the frozen eval pools.

A ranker maps (task, lessons) to one float per lesson; higher is better. The
same three numbers Phase 1 used gate every ranker here: precision@k, hit@k
(at least one relevant row in the top k), and no-answer accuracy (a case the
corpus has never covered returns nothing). Abstention is a score threshold:
rows under `min_score` are dropped before the top-k cut.
"""
from __future__ import annotations

import json
from pathlib import Path
import os
from typing import Protocol

DEFAULT_POOLS = Path(os.environ.get("SMART_AGENTS_DATA_DIR", Path.home() / ".local/share/smart-agents")) / "learning-corpus" / "eval" / "eval_pools.json"
# v2 (2026-09-03): ~100 multi-label cases plus ~30 no-answer, built by smart_agents.eval.build_v2.
# v1 is saturated (hit@3 1.000, precision@3 ceiling 0.411), so v2 is the gate's primary signal
# whenever its pools file exists; v1 stays as a no-regression check.
V2_POOLS = DEFAULT_POOLS.with_name("eval_pools_v2.json")


def pool_sets() -> dict[str, Path]:
    """The pools files that exist, keyed v1/v2. v2 comes first when present."""
    sets = {}
    if V2_POOLS.exists():
        sets["v2"] = V2_POOLS
    sets["v1"] = DEFAULT_POOLS
    return sets


class Ranker(Protocol):
    def score(self, task: str, lessons: list[str]) -> list[float]: ...


def load_pools(path: Path = DEFAULT_POOLS) -> list[dict]:
    return json.loads(Path(path).read_text())


def rank_case(case: dict, ranker: Ranker, *, k: int, min_score: float | None) -> list[int]:
    """Ids of the top k pool rows for one case, after the threshold."""
    pool = case["pool"]
    if not pool:
        return []
    scores = ranker.score(case["task"], [r["lesson"] for r in pool])
    kept = [(s, r["id"]) for s, r in zip(scores, pool) if min_score is None or s >= min_score]
    kept.sort(key=lambda t: -t[0])
    return [rid for _, rid in kept[:k]]


def evaluate(cases: list[dict], ranker: Ranker, *, k: int = 3, min_score: float | None = None) -> dict:
    answerable = [c for c in cases if c["kind"] == "answerable"]
    no_answer = [c for c in cases if c["kind"] == "no_answer"]

    precisions, hits = [], 0
    for c in answerable:
        top = rank_case(c, ranker, k=k, min_score=min_score)
        h = sum(1 for rid in top if rid in set(c["relevant_ids"]))
        precisions.append(h / k)
        hits += h > 0

    quiet = sum(1 for c in no_answer if not rank_case(c, ranker, k=k, min_score=min_score))

    return {
        "precision_at_k": sum(precisions) / len(precisions) if precisions else 0.0,
        "hit_at_k": hits / len(answerable) if answerable else 0.0,
        "no_answer_accuracy": quiet / len(no_answer) if no_answer else 0.0,
        "k": k,
        "n_answerable": len(answerable),
        "n_no_answer": len(no_answer),
    }
