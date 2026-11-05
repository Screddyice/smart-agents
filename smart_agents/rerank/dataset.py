"""Turn labeled pairs into a training set the model can learn from.

Each task carries ~20 candidates and ~2 positives, so raw pairs are 90%
negative and the model learns to say no. Cap negatives per positive so the
loss sees both sides; keep a few negatives for tasks with no positive at all,
since "nothing here applies" is a behaviour the ranker must keep.
"""
from __future__ import annotations

import random
from collections import defaultdict


def balance(pairs: list[dict], *, max_neg_per_pos: int = 4, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    by_task: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_task[p["task"]].append(p)
    out: list[dict] = []
    for task_pairs in by_task.values():
        pos = [p for p in task_pairs if p["label"] == 1]
        neg = [p for p in task_pairs if p["label"] == 0]
        rng.shuffle(neg)
        keep = max_neg_per_pos * len(pos) if pos else max_neg_per_pos
        out.extend(pos)
        out.extend(neg[:keep])
    rng.shuffle(out)
    return out
