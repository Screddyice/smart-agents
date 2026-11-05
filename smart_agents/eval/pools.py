"""Dump the candidate pool behind every eval case to JSON.

Runs where Ollama (nomic-embed-text) and the corpus are both reachable. On
the Mac that is local Ollama plus Supabase. The pool is the 18 nearest rows
by cosine, the same over-fetch /advise uses for k=3, plus what the live
pipeline returns today so a ranker can be compared against it without
calling the endpoint.

    RDS_URL=placeholder uv run python -m smart_agents.eval.pools --out learning-corpus/eval/eval_pools.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

from smart_agents.eval.embed import embed  # noqa: F401  (re-exported for callers)

EVAL = Path(__file__).resolve().parents[2] / "learning-corpus" / "eval" / "retrieval_eval.jsonl"
POOL_SIZE = 18


def pool_for(conn: psycopg.Connection, vector: list[float]) -> list[dict]:
    rows = conn.execute(
        """
        select id, category, lesson, occurred_at::date::text as occurred_at,
               embedding::text as emb, embedding <=> %s::vector as dist
        from learnings
        where embedding is not null and superseded_by is null
        order by embedding <=> %s::vector
        limit %s
        """,
        (str(vector), str(vector), POOL_SIZE),
    ).fetchall()
    return [
        {
            "id": r["id"], "category": r["category"], "lesson": r["lesson"],
            "occurred_at": r["occurred_at"], "dist": float(r["dist"]),
            # Five decimals keep cosine to 1e-5 and halve the fixture's size.
            "embedding": [round(float(x), 5) for x in r["emb"].strip("[]").split(",")],
        }
        for r in rows
    ]


def live_top3(pool: list[dict]) -> list[int]:
    # corpus_api.main reads RDS_URL at import; keep that out of module import.
    from corpus_api.main import MAX_DISTANCE
    from corpus_api.rank import apply_weights, mmr

    cands = [dict(r, raw_dist=r["dist"], occurred_at=None) for r in pool if r["dist"] < MAX_DISTANCE]
    return [c["id"] for c in mmr(apply_weights(cands), lambda_=0.7, k=3, floor=0.8)]


async def build(dsn: str, ollama: str, eval_path: Path = EVAL) -> list[dict]:
    cases = [json.loads(l) for l in eval_path.read_text().splitlines() if l.strip()]
    out = []
    with psycopg.connect(dsn, row_factory=dict_row, prepare_threshold=None) as conn:
        for c in cases:
            pool = pool_for(conn, await embed(c["task"], ollama))
            out.append({"task": c["task"], "kind": c["kind"], "relevant_ids": c.get("relevant_ids", []),
                        "live_top3": live_top3(pool), "pool": pool})
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--eval", type=Path, default=EVAL, help="eval JSONL to dump pools for (v1 by default)")
    p.add_argument("--dsn-env", default="LEARNING_CORPUS_RDS_URL")
    p.add_argument("--ollama", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    args = p.parse_args(argv)
    dsn = os.environ.get(args.dsn_env)
    if not dsn:
        print(f"{args.dsn_env} is not set", file=sys.stderr)
        return 2
    cases = asyncio.run(build(dsn, args.ollama, args.eval))
    args.out.write_text(json.dumps(cases, separators=(",", ":")))
    print(f"wrote {len(cases)} cases to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
