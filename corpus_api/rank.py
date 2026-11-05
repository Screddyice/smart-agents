"""Result diversification for /advise.

Plain cosine ordering returns near-identical lessons, because the corpus holds
many rows expressing one idea. Maximal Marginal Relevance picks each next result
for a blend of relevance to the query and dissimilarity from what is already
picked, so a fixed result budget carries distinct ideas.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

# Cosine similarity below which two lessons count as unrelated. nomic-embed-text
# places any two lessons from this corpus at roughly 0.6-0.85 (measured
# 2026-09-03 over the eval set's candidate pools: p10 0.63, median 0.72, p90
# 0.85). Raw cosine as the redundancy term therefore penalises every candidate
# about equally and MMR spends precision without buying variety. Rescaling so
# that only similarity above this floor counts leaves genuinely distinct rows
# unpenalised and reserves the diversity penalty for near-duplicates.
SIMILARITY_FLOOR = 0.8


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _redundancy(similarity: float, floor: float) -> float:
    """Map raw cosine onto [0, 1] so that anything at or below the floor is 0."""
    if floor >= 1.0:
        return 0.0
    return max(0.0, (similarity - floor) / (1.0 - floor))


def mmr(
    candidates: list[dict],
    *,
    lambda_: float = 0.7,
    k: int = 5,
    floor: float = SIMILARITY_FLOOR,
) -> list[dict]:
    """Select k candidates balancing query relevance against mutual similarity.

    candidates carry `dist` (cosine distance to the query, lower is better) and
    `embedding`. lambda_=1 is pure relevance; lambda_=0 is pure diversity.
    floor is the cosine similarity below which candidates are treated as
    unrelated; see SIMILARITY_FLOOR.
    """
    if not candidates:
        return []
    pool = list(candidates)
    selected: list[dict] = []

    while pool and len(selected) < k:
        best = None
        best_score = -math.inf
        for cand in pool:
            relevance = 1.0 - float(cand["dist"])
            if selected:
                similarity = max(
                    _cosine(cand["embedding"], s["embedding"]) for s in selected
                )
                redundancy = _redundancy(similarity, floor)
            else:
                redundancy = 0.0
            score = lambda_ * relevance - (1.0 - lambda_) * redundancy
            if score > best_score:
                best_score = score
                best = cand
        selected.append(best)
        pool.remove(best)

    return selected


from datetime import datetime, timezone  # noqa: E402

# Multipliers on cosine distance, so below 1.0 means "treat as closer".
# Traps are what the agent most needs and are 24% of the corpus, so they get a
# boost. The values are gentle on purpose: relevance must still dominate, or a
# distantly-related mistake outranks an exactly-matching fact. The plan started
# at 0.85/0.87/0.90/0.95; on the eval set that strength, and half of it, each
# pushed one labeled answer out of the top 3 (a correction at raw rank 1 lost
# a near-tie to a mistake and then paid the MMR floor penalty against it).
# 0.4x of the plan's boost is the largest tested that loses nothing; see
# learning-corpus/eval/BASELINE.md.
_CATEGORY_WEIGHT = {
    "mistake": 0.94,
    "correction": 0.948,
    "inefficiency": 0.96,
    "strategy_pivot": 0.98,
    "new_info": 1.00,
}

# A year old costs 5% of distance. Old lessons stay reachable; fresh ones win ties.
_AGE_PENALTY_PER_YEAR = 0.05
_MAX_AGE_PENALTY = 0.15


def apply_weights(candidates: list[dict], *, now: datetime | None = None) -> list[dict]:
    """Adjust each candidate's `dist` by category and age. Mutates and returns."""
    now = now or datetime.now(timezone.utc)
    for c in candidates:
        weight = _CATEGORY_WEIGHT.get(c.get("category"), 1.0)
        occurred = c.get("occurred_at")
        penalty = 0.0
        if occurred is not None:
            if occurred.tzinfo is None:
                occurred = occurred.replace(tzinfo=timezone.utc)
            years = (now - occurred).days / 365.25
            penalty = min(max(years, 0.0) * _AGE_PENALTY_PER_YEAR, _MAX_AGE_PENALTY)
        c["dist"] = max(0.0, float(c["dist"]) * weight + penalty)
    return candidates
