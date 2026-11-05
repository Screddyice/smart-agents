"""Collapse near-duplicate lessons.

The corpus accumulates many rows expressing one idea, because each session
distils independently. MMR hides that at query time; this removes it at the
source so result slots and training rows both carry distinct ideas.

The survivor of a group is the LOWEST id, which is the oldest row. Ids already
cited in transcripts and specs stay valid that way. Losers are marked with
`superseded_by`, never deleted, and only when `--apply` is passed; the default
run reports and changes nothing.

Run it where the database is reachable. The live store answers only on its own
host, so on `src`:

    PYTHONPATH=. python -m smart_agents.consolidation.learnings --dsn-env RDS_URL
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

from corpus_api.rank import _cosine


def find_duplicate_groups(rows: list[dict], *, threshold: float = 0.97) -> list[list[int]]:
    """Group rows whose embeddings exceed the cosine similarity threshold.

    Greedy in id order: the lowest unclaimed id opens a group and claims every
    later unclaimed row above the threshold. With numpy present each row's
    similarities come from one matrix product; without it the same walk runs
    in pure Python, which is fine for tests and far too slow for the corpus
    (5,800 rows is 17 million pairs of 768 floats).
    """
    remaining = sorted(rows, key=lambda r: r["id"])
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - exercised only where numpy is absent
        np = None

    if np is not None and remaining:
        m = np.asarray([r["embedding"] for r in remaining], dtype=np.float64)
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        m = m / norms
        sims_of = lambda i: m[i] @ m.T  # noqa: E731
    else:
        sims_of = lambda i: [  # noqa: E731
            _cosine(remaining[i]["embedding"], other["embedding"]) for other in remaining
        ]

    groups: list[list[int]] = []
    claimed: set[int] = set()
    for i, row in enumerate(remaining):
        if row["id"] in claimed:
            continue
        group = [row["id"]]
        claimed.add(row["id"])
        sims = sims_of(i)
        if np is not None:
            # Only the rows above threshold are walked, not all n of them.
            hits = (np.nonzero(sims[i + 1:] >= threshold)[0] + i + 1).tolist()
        else:
            hits = [j for j in range(i + 1, len(remaining)) if sims[j] >= threshold]
        for j in hits:
            other = remaining[j]
            if other["id"] in claimed:
                continue
            group.append(other["id"])
            claimed.add(other["id"])
        groups.append(group)

    return groups


def _load_rows(conn: psycopg.Connection) -> list[dict]:
    # pgvector's adapter hands back numpy arrays straight from the binary
    # protocol. Casting 5,800 embeddings to text and parsing 4.4 million floats
    # in Python was the slow half of a run that did not finish in ten minutes.
    from pgvector.psycopg import register_vector

    register_vector(conn)
    t0 = time.monotonic()
    rows = conn.execute(
        """
        select id, lesson, embedding
        from learnings
        where embedding is not null and superseded_by is null
        order by id
        """
    ).fetchall()
    print(f"loaded {len(rows)} rows in {time.monotonic() - t0:.1f}s", file=sys.stderr)
    # pgvector >= 0.4 hands back a Vector wrapper; older versions a numpy array.
    return [
        {
            "id": r["id"],
            "lesson": r["lesson"],
            "embedding": r["embedding"].to_numpy() if hasattr(r["embedding"], "to_numpy") else r["embedding"],
        }
        for r in rows
    ]


def _report(groups: Iterable[list[int]], by_id: dict[int, dict], *, show: int) -> None:
    dupes = [g for g in groups if len(g) > 1]
    losers = sum(len(g) - 1 for g in dupes)
    print(f"groups with duplicates: {len(dupes)}")
    print(f"rows that would be superseded: {losers}")
    for g in dupes[:show]:
        print(f"\n#{g[0]} keeps; supersedes {', '.join(f'#{x}' for x in g[1:])}")
        for rid in g:
            print(f"   #{rid}: {by_id[rid]['lesson'][:110]!r}")


REVIEW_PROMPT = (
    "These lessons from an agent's learning corpus were flagged as near-duplicates by embedding "
    "similarity. Decide whether they state the SAME lesson (same advice, same situation) such that "
    "keeping one and hiding the rest loses nothing an engineer would need. Paraphrases and "
    "re-statements are the same lesson; different advice, different tools, or a materially different "
    "condition are not. Return JSON only.\n\nLESSONS:\n{lessons}\n"
)
REVIEW_SCHEMA = {"type": "object",
                 "properties": {"same": {"type": "boolean"}, "reason": {"type": "string"}},
                 "required": ["same", "reason"], "additionalProperties": False}

Reviewer = Callable[[list[dict]], tuple[bool, str]]


def codex_review(rows: list[dict], *, timeout: int = 180) -> tuple[bool, str]:
    """Ask codex whether a group is one lesson. (same, reason); a failure counts as not same."""
    from smart_agents.judgments import decide,confident_choice
    try:
        result=decide({'text':json.dumps([{'id':r['id'],'lesson':r['lesson']} for r in rows]),
                       'context':'Compare operational lessons for exact advice and applicability. Similar topics alone are not the same lesson.'},
                      {'same_lesson':{'type':'choice','instructions':'Can these records be consolidated without losing any distinct advice or condition?',
                         'criteria':{'same_record':'Same advice and same conditions','different_record':'Distinct advice or applicability','needs_review':'Insufficient evidence'}}})
        choice=confident_choice(result['answers']['same_lesson'])
        return choice=='same_record',choice
    except Exception as exc:return False,'needs_review:'+type(exc).__name__


def prior_decisions(log_path: Path | None) -> dict[tuple[int, ...], bool]:
    """Decisions already in the review log, so a rerun does not ask codex twice."""
    if not log_path or not log_path.exists():
        return {}
    out: dict[tuple[int, ...], bool] = {}
    for line in log_path.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            if not str(d.get("reason", "")).startswith(("needs_review", "review failed")):
                out[tuple(d["group"])] = bool(d["same"])
    return out


def review_groups(groups: Iterable[list[int]], by_id: dict[int, dict], *, reviewer: Reviewer,
                  log=None, prior: dict[tuple[int, ...], bool] | None = None) -> list[list[int]]:
    """Keep only the duplicate groups the reviewer confirms. Logs one JSON line per new decision."""
    confirmed: list[list[int]] = []
    prior = prior or {}
    for g in groups:
        if len(g) < 2:
            continue
        if tuple(g) in prior:
            if prior[tuple(g)]:
                confirmed.append(g)
            continue
        same, reason = reviewer([by_id[i] for i in g])
        if log is not None:
            log.write(json.dumps({"group": g, "same": same, "reason": reason}) + "\n"); log.flush()
        if same:
            confirmed.append(g)
    return confirmed


EVAL_FILES = (
    Path(os.environ.get("SMART_AGENTS_DATA_DIR", Path.home()/".local/share/smart-agents")) / "learning-corpus" / "eval" / "retrieval_eval.jsonl",
    Path(os.environ.get("SMART_AGENTS_DATA_DIR", Path.home()/".local/share/smart-agents")) / "learning-corpus" / "eval" / "retrieval_eval_v2.jsonl",
)


def eval_label_ids(files: Iterable[Path] = EVAL_FILES) -> set[int]:
    ids: set[int] = set()
    for f in files:
        if Path(f).exists():
            for line in Path(f).read_text().splitlines():
                if line.strip():
                    ids.update(json.loads(line).get("relevant_ids", []))
    return ids


def protect_labels(groups: Iterable[list[int]], labels: set[int]) -> tuple[list[list[int]], list[list[int]]]:
    """Split groups into (safe, skipped): a group is skipped when a row it would
    supersede is an eval label, because a superseded row can never be returned
    and the eval would count that as a miss forever. The survivor being labeled
    is fine."""
    safe, skipped = [], []
    for g in groups:
        (skipped if any(i in labels for i in g[1:]) else safe).append(g)
    return safe, skipped


def _apply(conn: psycopg.Connection, groups: Iterable[list[int]]) -> int:
    n = 0
    for g in groups:
        if len(g) < 2:
            continue
        survivor, losers = g[0], g[1:]
        conn.execute(
            "update learnings set superseded_by = %s where id = any(%s) and superseded_by is null",
            (survivor, losers),
        )
        n += len(losers)
    conn.commit()
    return n


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--threshold", type=float, default=0.97)
    p.add_argument("--dsn-env", default="SUPABASE_DB_URL_POOLER_FULL",
                   help="environment variable holding the connection string")
    p.add_argument("--show", type=int, default=10, help="how many groups to print in the report")
    p.add_argument("--apply", action="store_true",
                   help="write superseded_by; without this the run only reports")
    p.add_argument("--review", action="store_true",
                   help="ask typed JEV whether each group is one lesson; only confirmed groups count")
    p.add_argument("--review-log", type=Path, default=None,
                   help="JSONL of every review decision (default: learning-corpus/rerank/consolidation-review.jsonl)")
    args = p.parse_args(argv)

    dsn = os.environ.get(args.dsn_env)
    if not dsn:
        print(f"{args.dsn_env} is not set", file=sys.stderr)
        return 2

    # prepare_threshold=None: Supabase's pooler rejects auto-prepared statements.
    with psycopg.connect(dsn, row_factory=dict_row, prepare_threshold=None) as conn:
        rows = _load_rows(conn)
        t0 = time.monotonic()
        groups = find_duplicate_groups(rows, threshold=args.threshold)
        print(f"grouped in {time.monotonic() - t0:.1f}s", file=sys.stderr)
        by_id = {r["id"]: r for r in rows}
        if args.review:
            log_path = args.review_log or Path(os.environ.get("SMART_AGENTS_DATA_DIR", Path.home()/".local/share/smart-agents")) / "learning-corpus" / "rerank" / "consolidation-review.jsonl"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            dupes = [g for g in groups if len(g) > 1]
            print(f"reviewing {len(dupes)} groups with typed JEV", file=sys.stderr)
            prior = prior_decisions(log_path)
            print(f"{len(prior)} decisions already on file", file=sys.stderr)
            with log_path.open("a") as log:
                groups = review_groups(dupes, by_id, reviewer=codex_review, log=log, prior=prior)
            print(f"confirmed {len(groups)} of {len(dupes)} groups", file=sys.stderr)
        labels = eval_label_ids()
        groups, skipped = protect_labels(groups, labels)
        if skipped:
            print(f"protecting eval labels: {len(skipped)} group(s) skipped "
                  f"({', '.join('#' + str(i) for g in skipped for i in g[1:] if i in labels)})", file=sys.stderr)
        _report(groups, by_id, show=args.show)
        if args.apply:
            print(f"\nsuperseded {_apply(conn, groups)} rows")
        else:
            print("\ndry run; pass --apply to write superseded_by")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
