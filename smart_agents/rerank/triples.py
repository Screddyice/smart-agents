"""Build rerank training triples from the corpus.

A triple is (task, candidates[20], relevant_ids). Tasks are real
`what_happened` situations from the corpus; candidates are the 20 nearest rows
by vector with the source row itself removed (it would be a free positive);
labels come from codex. Rows that appear anywhere in the frozen eval set, or
in its candidate pools, are never used as tasks or candidates, so training
cannot see the test set's neighbourhood.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import os

import psycopg
from psycopg.rows import dict_row

from smart_agents.eval.embed import embed
from smart_agents.rerank.labeler import label

ROOT = Path(os.environ.get("SMART_AGENTS_DATA_DIR", Path.home() / ".local/share/smart-agents"))
EVAL = ROOT / "learning-corpus" / "eval" / "retrieval_eval.jsonl"
POOLS = ROOT / "learning-corpus" / "eval" / "eval_pools.json"
TRAP = ("mistake", "correction", "inefficiency", "strategy_pivot")
MAX_NEW_INFO_SHARE = 0.55


EVAL_V2 = EVAL.with_name("retrieval_eval_v2.jsonl")
POOLS_V2 = POOLS.with_name("eval_pools_v2.json")


def excluded_ids(eval_path: Path = EVAL, pools_path: Path = POOLS,
                 eval_v2: Path = EVAL_V2, pools_v2: Path = POOLS_V2) -> set[int]:
    """Every row the eval sets label or pool, v1 and (when present) v2."""
    ids: set[int] = set()
    for ev, pools in ((eval_path, pools_path), (eval_v2, pools_v2)):
        if ev.exists():
            for line in ev.read_text().splitlines():
                if line.strip():
                    case = json.loads(line)
                    ids.update(case.get("relevant_ids", []))
                    if case.get("source_id") is not None:
                        ids.add(case["source_id"])
        if pools.exists():
            for case in json.loads(pools.read_text()):
                ids.update(r["id"] for r in case["pool"])
    return ids


def sample_tasks(conn, *, n: int, seed: int, exclude_ids: set[int]) -> list[dict]:
    rows = conn.execute(
        "select id, category, source, what_happened, tags, source_session_id from learnings "
        "where superseded_by is null and length(trim(what_happened)) >= 40 order by id"
    ).fetchall()
    rows = [r for r in rows if r["id"] not in exclude_ids]
    rng = random.Random(seed)
    nebby_corrections = [
        r for r in rows
        if r.get("source") == "observed" and "nebby-correction" in (r.get("tags") or [])
    ]
    rng.shuffle(nebby_corrections)
    priority_ids = {r["id"] for r in nebby_corrections}
    remaining = [r for r in rows if r["id"] not in priority_ids]
    rng.shuffle(remaining)
    new_info = [r for r in remaining if r["category"] == "new_info"]
    traps = [r for r in remaining if r["category"] in TRAP]
    picked = nebby_corrections[:n]
    slots = n - len(picked)
    n_new = min(len(new_info), int(n * MAX_NEW_INFO_SHARE), slots)
    picked += new_info[:n_new] + traps[: slots - n_new]
    if len(picked) < n:
        picked += new_info[n_new : n_new + (n - len(picked))]
    rng.shuffle(picked)
    return picked[:n]


def candidates_for(conn, vector: list[float], *, exclude_id: int, k: int = 20) -> list[dict]:
    rows = conn.execute(
        """
        select id, lesson, source_session_id, embedding::text as emb, embedding <=> %s::vector as dist
        from learnings
        where embedding is not null and superseded_by is null and id <> %s
        order by embedding <=> %s::vector
        limit %s
        """,
        (str(vector), exclude_id, str(vector), k),
    ).fetchall()
    return [{"id": r["id"], "lesson": r["lesson"], "dist": float(r["dist"]), "embedding":[float(v) for v in r["emb"].strip("[]").split(",")], "source_session_id":r["source_session_id"]} for r in rows]


def split_pairs(triples: list[dict], *, val_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Expand triples into (task, lesson, label) pairs, split by task."""
    task_ids = sorted({t["task_id"] for t in triples})
    rng = random.Random(seed)
    rng.shuffle(task_ids)
    n_val = max(1, int(len(task_ids) * val_fraction)) if len(task_ids) > 1 else 0
    val_tasks = set(task_ids[:n_val])
    train, val = [], []
    for t in triples:
        rel = set(t["relevant_ids"])
        for c in t["candidates"]:
            pair = {"task": t["task"], "lesson": c["lesson"], "label": int(c["id"] in rel)}
            (val if t["task_id"] in val_tasks else train).append(pair)
    return train, val


def label_with_retry(task, candidates, *, timeout=30):
    """Compatibility name: make one call only; uncertain/billed calls are not retried."""
    try:return label(task,candidates,timeout=timeout)
    except Exception as exc:
        print(f"labeler needs review: {type(exc).__name__}",file=sys.stderr)
        return None


async def build(out: Path, *, n: int, seed: int, dsn: str, ollama: str) -> int:
    done: set[int] = set()
    if out.exists():
        done = {json.loads(l)["task_id"] for l in out.read_text().splitlines() if l.strip()}
    excluded = excluded_ids() | done
    written = 0
    spent = 0.0
    budget = float(os.environ.get("SMART_AGENTS_LABEL_BUDGET_USD", "0.25"))
    if budget <= 0:raise ValueError("A positive labeling budget is required")
    with psycopg.connect(dsn, row_factory=dict_row, prepare_threshold=None) as conn:
        tasks = sample_tasks(conn, n=n, seed=seed, exclude_ids=excluded)
        with out.open("a") as fh:
            for row in tasks:
                vector = await embed(row["what_happened"], ollama)
                cands = [c for c in candidates_for(conn, vector, exclude_id=row["id"]) if c["id"] not in excluded]
                if len(cands) < 5:
                    continue
                if spent >= budget:raise RuntimeError("Labeling budget exhausted; no further requests made")
                result = label_with_retry(row["what_happened"], cands)
                if result is None:
                    print(f"skipping task #{row['id']}: labeler failed twice", file=sys.stderr)
                    continue
                spent += sum(float(r.get("usage",{}).get("cost",0)) for r in result.receipts)
                if spent > budget:raise RuntimeError("Labeling budget reached; retain provider receipts and review before continuing")
                cand_ids = {c["id"] for c in cands}
                fh.write(json.dumps({
                    "task_id": row["id"], "task": row["what_happened"], "category": row["category"],
                    "candidates": cands, "relevant_ids": [i for i in result.relevant_ids if i in cand_ids],
                    "source_session_id": row["source_session_id"], "rare": "rare" in (row.get("tags") or []), "labeler": f"configured:{result.model}", "reason": result.reason,
                    "receipts": result.receipts, "labeled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }) + "\n")
                fh.flush()
                written += 1
                print(f"[{written}/{len(tasks)}] task #{row['id']}: {len(result.relevant_ids)} relevant", file=sys.stderr)
    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", type=Path, default=ROOT / "learning-corpus" / "rerank" / "triples.jsonl")
    p.add_argument("--dsn-env", default="LEARNING_CORPUS_RDS_URL")
    p.add_argument("--ollama", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    args = p.parse_args(argv)
    dsn = os.environ.get(args.dsn_env)
    if not dsn:
        print(f"{args.dsn_env} is not set", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = asyncio.run(build(args.out, n=args.n, seed=args.seed, dsn=dsn, ollama=args.ollama))
    print(f"wrote {n} triples to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
