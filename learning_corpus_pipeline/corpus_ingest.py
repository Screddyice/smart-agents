"""claude-mem hub → corpus `learnings` ingest. Backfill in one shot, or poll forever.

This replaced the HyperSwarm filesystem watcher. The old daemon was
event-driven (inotify/FSEvents on ~/HyperSwarm/entries); the hub has no such
local surface, so this polls its timeline API instead. Polling is fine at this
volume — roughly 50 learnings a day — and it removes the whole class of bug the
old watcher kept hitting, where rsync's write-then-rename arrived as a move
rather than a create and entries went silently unseen for weeks.

Two modes, one code path:

    corpus-ingest --once              # backfill everything, exit
    corpus-ingest --poll 900          # same work every 15 min, forever

Idempotency comes from the writer's unique constraint
(instance, agent_id, source_entry_hash, category) plus the local cursor, so
running the backfill twice inserts nothing the second time. That matters here:
the first run has ~1500 items of accumulated backlog to work through, and it
needs to be safe to interrupt and restart.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import signal
import sys
import pathlib
import threading
import time
from datetime import datetime, timedelta, timezone

from learning_classifier import classify

from .config import Config
from .cursor import Cursor
from .cmem_source import CmemClient, fetch_learnings
from .learning_items import DEFAULT_CHUNK_SIZE, group_into_chunks
from .embedding import EmbeddingBackend

log = logging.getLogger(__name__)

class SeenIds:
    """Hub item ids already folded into a written chunk.

    Separate from the Cursor (which keys on chunk body_hash) because it answers
    a different question: the cursor says "don't redo this work", this says
    "don't even download it". Ids are added only after a successful write, so an
    interrupted backfill resumes rather than skipping what it never wrote.
    """

    def __init__(self, path: pathlib.Path):
        self.path = path
        self.ids: set[str] = set()
        if path.exists():
            try:
                data = json.loads(path.read_text())
                self.ids = set(data.get("ingested_ids", []))
            except (json.JSONDecodeError, OSError, TypeError):
                log.warning("corpus-ingest: unreadable seen-ids at %s; starting fresh", path)

    def add(self, ids) -> None:
        self.ids.update(ids)

    def save(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"ingested_ids": sorted(self.ids)}))
        os.replace(tmp, self.path)


_REL_RE = re.compile(r"^(\d+)([dhw])$")


def _parse_since(s: str) -> datetime:
    """Accept '14d' / '6h' / '2w' / an ISO date, or 'all' for no lower bound."""
    if s.lower() == "all":
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
    m = _REL_RE.match(s.strip())
    if m:
        n, unit = int(m.group(1)), m.group(2)
        delta = {"d": timedelta(days=n), "h": timedelta(hours=n), "w": timedelta(weeks=n)}[unit]
        return datetime.now(timezone.utc) - delta
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"could not parse --since={s!r}: {e}")


def _build_client(cfg: Config) -> CmemClient:
    """The hub is the only source. CMEM_PRO_TOKEN comes from the unit's
    environment (~/.learning-corpus/corpus.env on src); never logged."""
    token = os.environ.get("CMEM_PRO_TOKEN", "")
    if not token:
        raise RuntimeError(
            "corpus-ingest needs CMEM_PRO_TOKEN. "
            "Source ~/.learning-corpus/corpus.env into the unit's environment."
        )
    return CmemClient(token)


def run_once(cfg: Config, *, since: datetime, limit: int | None, chunk_size: int,
             dry_run: bool, cursor: Cursor | None, seen: "SeenIds | None" = None) -> dict:
    """One full pass. Returns a stats dict."""
    stats = {"learnings_fetched": 0, "sessions": 0, "seen": 0, "skipped_cursor": 0,
             "classified": 0, "no_learnings": 0, "inserted": 0, "errors": 0,
             "backend_errors": 0}

    client = _build_client(cfg)
    items = fetch_learnings(client, since=since, limit=limit,
                            skip_ids=(seen.ids if seen else None))
    stats["learnings_fetched"] = len(items)
    if not items:
        log.info("corpus-ingest: nothing to do")
        return stats

    embedder = None
    rds_writer = None
    if not dry_run:
        embedder = EmbeddingBackend(cfg.ollama_url, cfg.embedding_model)
        from .writer_rds import RDSWriter
        rds_writer = RDSWriter(cfg, embedder)

    try:
        chunks = list(group_into_chunks(items, chunk_size=chunk_size))
        stats["sessions"] = len({c.entry.session_id for c in chunks})
        log.info("corpus-ingest: %d learnings → %d chunks across %d sessions",
                 len(items), len(chunks), stats["sessions"])

        for chunk in chunks:
            entry = chunk.entry
            stats["seen"] += 1
            if cursor is not None and cursor.already_processed(entry.body_hash):
                stats["skipped_cursor"] += 1
                continue
            try:
                if dry_run:
                    log.info("[dry-run] %s → %d chars, agent=%s",
                             entry.path.name, len(entry.body), entry.agent_id)
                    continue

                # Hub items are already distilled; the prefilter is a
                # transcript heuristic and rejects all of them.
                result = classify(entry.body, skip_prefilter=True)

                def done() -> None:
                    """Record this chunk so it is never fetched or judged again.

                    Called for every settled outcome, not only a written one. A
                    chunk the classifier judged empty is finished business, but
                    only a successful write used to record anything, so every
                    rejected chunk was re-downloaded and re-classified on every
                    pass, forever. Measured on src: consecutive passes fetched
                    52, then 100, then 118 chunks while the seen-file held 77,
                    97 of that last 118 were re-judgements of rows already
                    rejected, and one pass took 892 seconds.
                    """
                    if cursor is not None:
                        cursor.mark(entry.body_hash, entry.path.name)
                    if seen is not None:
                        seen.add(chunk.source_ids)
                        seen.save()

                if result.skipped_reason:
                    # A backend failure arrives as a skip rather than an
                    # exception, so it must not settle the chunk: a classifier
                    # outage would otherwise drop every row it touched,
                    # permanently and without a word.
                    if str(result.skipped_reason).startswith("backend_error"):
                        stats["backend_errors"] += 1
                        continue
                    stats["no_learnings"] += 1
                    done()
                    continue
                stats["classified"] += 1
                if not result.learnings:
                    stats["no_learnings"] += 1
                    done()
                    continue

                if rds_writer is not None:
                    stats["inserted"] += rds_writer.write(entry, result.learnings)

                # Only now are these items safely in the corpus.
                done()
            except Exception:
                stats["errors"] += 1
                log.exception("corpus-ingest: error on %s", entry.path.name)
    finally:
        if rds_writer is not None:
            try:
                rds_writer.close()
            except Exception:
                log.warning("corpus-ingest: writer close failed", exc_info=True)

    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--since", default="all", type=_parse_since,
                        help="lower bound on the learning's own date (14d / ISO / all)")
    parser.add_argument("--limit", type=int, default=None, help="cap learnings fetched")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help="learnings per classifier call")
    parser.add_argument("--poll", type=int, default=0, metavar="SECONDS",
                        help="poll forever at this interval; 0 = run once and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="fetch and group only; no LLM calls, no DB writes")
    parser.add_argument("--no-cursor", action="store_true",
                        help="ignore the cursor (force reprocess; DB still dedupes)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    cfg = Config.from_env(need_rds=not args.dry_run)
    cursor = None
    seen = None
    if not args.no_cursor and not args.dry_run:
        cursor = Cursor(cfg.state_dir / "corpus_ingest.cursor.json")
        seen = SeenIds(cfg.state_dir / "corpus_ingest.seen.json")

    log.info("corpus-ingest: host=%s since=%s poll=%s dry_run=%s",
             cfg.host_role, args.since.date().isoformat(), args.poll or "off", args.dry_run)

    if not args.poll:
        stats = run_once(cfg, since=args.since, limit=args.limit,
                         chunk_size=args.chunk_size, dry_run=args.dry_run,
                         cursor=cursor, seen=seen)
        log.info("corpus-ingest: done stats=%s", stats)
        return 0 if stats["errors"] == 0 else 1

    stop = threading.Event()

    def _sigterm(signum, frame):
        log.info("corpus-ingest: signal %s, stopping…", signum)
        stop.set()

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)

    while not stop.is_set():
        t0 = time.monotonic()
        try:
            stats = run_once(cfg, since=args.since, limit=args.limit,
                             chunk_size=args.chunk_size, dry_run=args.dry_run,
                             cursor=cursor, seen=seen)
            log.info("corpus-ingest: pass done in %.1fs stats=%s",
                     time.monotonic() - t0, stats)
        except Exception:
            # A poll loop must outlive a transient hub outage.
            log.exception("corpus-ingest: pass failed; retrying next interval")
        stop.wait(args.poll)

    log.info("corpus-ingest: stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
