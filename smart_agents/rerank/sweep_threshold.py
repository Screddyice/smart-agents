"""Pick the abstention threshold for a ranker on the frozen pools, and record it.

    uv run python -m smart_agents.rerank.sweep_threshold                 # table only
    uv run python -m smart_agents.rerank.sweep_threshold --record        # also writes champion.json
    uv run python -m smart_agents.rerank.sweep_threshold --model <dir>   # a fine-tuned run
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
import os

from smart_agents.eval.rerank_score import evaluate, load_pools, pool_sets
from smart_agents.rerank.cross_encoder import STOCK_MODEL, CrossEncoderRanker

CHAMPION = Path(os.environ.get("SMART_AGENTS_DATA_DIR", Path.home() / ".local/share/smart-agents")) / "learning-corpus" / "rerank" / "champion.json"
THRESHOLDS = [None, -10.0, -8.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0]


def sha256_of_dir(path: Path) -> str | None:
    if not path.is_dir():
        return None
    h = hashlib.sha256()
    for f in sorted(path.rglob("*")):
        if f.is_file():
            h.update(str(f.relative_to(path)).encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def sweep(ranker: CrossEncoderRanker, cases: list[dict]) -> list[tuple[float | None, dict]]:
    return [(t, evaluate(cases, ranker, k=3, min_score=t)) for t in THRESHOLDS]


def choose(rows: list[tuple[float | None, dict]]) -> tuple[float, dict]:
    """Middle of the band of thresholds that keep no-answer accuracy at 1.0 at the best hit@3.

    The loosest passing threshold sits right on top of the highest no-answer
    score (for the stock model, -10.0 against a no-answer max of -10.61) and
    the tightest sits on top of the lowest relevant score. The middle of the
    band keeps margin on both sides, so a slightly different query is not a
    coin flip between answering and staying quiet.
    """
    best_hit = max(r["hit_at_k"] for _, r in rows)
    ok = sorted(
        ((t, r) for t, r in rows if t is not None and r["no_answer_accuracy"] == 1.0 and r["hit_at_k"] == best_hit),
        key=lambda x: x[0],
    )
    if not ok:
        return max(((t, r) for t, r in rows if t is not None),
                   key=lambda x: (x[1]["no_answer_accuracy"], x[1]["hit_at_k"]))
    return ok[(len(ok) - 1) // 2]


def sweep_sets(ranker: CrossEncoderRanker) -> dict[str, tuple[float, dict]]:
    """chosen (min_score, scores) per pools set, v2 first when it exists."""
    return {name: choose(sweep(ranker, load_pools(path))) for name, path in pool_sets().items()}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=STOCK_MODEL)
    p.add_argument("--record", action="store_true")
    p.add_argument("--pools", type=Path, default=None, help="one pools file to sweep (default: every known set)")
    args = p.parse_args(argv)

    ranker = CrossEncoderRanker(args.model)
    sets = {"custom": args.pools} if args.pools else pool_sets()
    chosen: dict[str, tuple[float, dict]] = {}
    for name, path in sets.items():
        rows = sweep(ranker, load_pools(path))
        print(f"== {name}: {path.name} ==")
        print("min_score  p@3    hit@3  no_answer")
        for t, r in rows:
            print(f"{str(t):<10} {r['precision_at_k']:.3f}  {r['hit_at_k']:.3f}  {r['no_answer_accuracy']:.3f}")
        t, r = choose(rows)
        chosen[name] = (t, r)
        print(f"chosen min_score={t}: p@3 {r['precision_at_k']:.3f} hit@3 {r['hit_at_k']:.3f} no_answer {r['no_answer_accuracy']:.3f}\n")

    if args.record:
        primary = "v2" if "v2" in chosen else next(iter(chosen))
        t, r = chosen[primary]
        CHAMPION.parent.mkdir(parents=True, exist_ok=True)
        CHAMPION.write_text(json.dumps({
            "model": args.model, "sha256": sha256_of_dir(Path(args.model)), "min_score": t,
            "scores": {k: r[k] for k in ("precision_at_k", "hit_at_k", "no_answer_accuracy")},
            "sets": {name: {"min_score": tt, **{k: rr[k] for k in ("precision_at_k", "hit_at_k", "no_answer_accuracy")}}
                     for name, (tt, rr) in chosen.items()},
            "primary": primary,
            "recorded_at": date.today().isoformat(),
        }, indent=2) + "\n")
        print(f"recorded champion in {CHAMPION} (primary set: {primary})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
