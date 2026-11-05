"""One distilled learning from the claude-mem hub, and how it becomes a corpus row.

A hub row is ONE already-distilled learning, so the classifier is handed one
item at a time and does what it is good at: mapping narrative onto the
`Learning` schema (category / what_happened / lesson / tags). Items are still
grouped by the session they came from so `session_id` / `agent_id` survive.

Exposes:
    LearningItem                        one item as fetched from the hub
    session_key_to_agent_id(key)        'claude_<id>' -> 'shawn-cc', etc.
    group_into_chunks(items, ...)       -> Iterator[Chunk]
    group_into_sessions(items, ...)     -> Iterator[SessionEntry]
"""
from __future__ import annotations

import hashlib
import logging
import pathlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Iterator

from .hyperswarm_source import SessionEntry

log = logging.getLogger(__name__)

# Items per classifier call. **Keep this at 1.**
#
# Batching looks like an easy win and silently destroys most of the data. The
# writer dedupes on (instance, agent_id, source_entry_hash, category), and every
# learning in a chunk shares that chunk's body_hash, so a chunk collapses to at
# most ONE row per category, five in total, no matter how many learnings it
# produced. Measured on the first backfill: 1527 items in, 1266 learnings
# classified, 241 rows written. Roughly 84% discarded, with `errors: 0` and a
# clean log. At 1, the hash identifies the actual hub item, which is what
# `source_entry_hash` means, and nothing collides.
DEFAULT_CHUNK_SIZE = 1


@dataclass(frozen=True)
class LearningItem:
    """One distilled learning as fetched from the hub."""

    data_id: str
    session_key: str          # 'claude_<id>' | 'codex_<id>' | 'hermes_<id>'
    occurred_at: datetime
    text: str


def session_key_to_agent_id(session_key: str) -> str:
    """Map a session key onto a canonical corpus agent_id.

    Mirrors hyperswarm_source._runtime_to_agent_id so rows written through this
    path are indistinguishable downstream from the ones HyperSwarm produced.
    """
    prefix = session_key.split("_", 1)[0].lower()
    if prefix == "claude":
        return "shawn-cc"
    if prefix == "codex":
        return "codex"
    if prefix == "hermes":
        return "screddy"
    return prefix or "unknown"


def _chunk(seq: list[LearningItem], size: int) -> Iterator[list[LearningItem]]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


@dataclass(frozen=True)
class Chunk:
    """One classifier-sized unit of work, plus the hub items that fed it.

    `source_ids` exists so the caller can mark those items ingested only after
    the chunk is actually written. Marking them at fetch time would drop
    learnings on the floor whenever a classify or insert failed, the failure
    mode that is hardest to notice, because the corpus simply grows a little
    less than it should.
    """

    entry: SessionEntry
    source_ids: tuple[str, ...]


def group_into_chunks(
    learnings: Iterable[LearningItem],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Iterator[Chunk]:
    """Group items by session and yield one Chunk per bounded batch.

    The synthesized `body` is what the classifier sees. `path` is synthetic:
    writers never read it (they use agent_id / project / occurred_at /
    session_id / body_hash); it exists because SessionEntry is shared with the
    filesystem path, and the daemon logs `path.name`, so it is built to be
    readable in a log line rather than to look like a real path.
    """
    if chunk_size > 1:
        log.warning(
            "chunk_size=%d will silently discard learnings. Every learning in a "
            "chunk shares one source_entry_hash, and the writer dedupes on "
            "(instance, agent_id, source_entry_hash, category), so each chunk keeps "
            "at most one row per category (5 max). Use chunk_size=1.", chunk_size)

    by_session: dict[str, list[LearningItem]] = defaultdict(list)
    for item in learnings:
        by_session[item.session_key].append(item)

    for session_key in sorted(by_session):
        items = sorted(by_session[session_key], key=lambda x: (x.occurred_at, x.data_id))
        agent_id = session_key_to_agent_id(session_key)

        for idx, chunk in enumerate(_chunk(items, chunk_size)):
            body = "\n\n".join(f"- {c.text}" for c in chunk)
            # Hash the content, not the chunk index: if chunk_size changes the
            # boundaries move, and an index-based hash would re-insert
            # everything as "new" on the next run.
            body_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()

            entry = SessionEntry(
                path=pathlib.Path(f"cmem_{session_key}_{idx:04d}"),
                runtime="cmem",
                scope="cross",
                project=None,
                session_id=session_key,
                occurred_at=chunk[-1].occurred_at,
                summary=f"{len(chunk)} distilled learnings from {session_key}"[:500],
                body=body,
                body_hash=body_hash,
                agent_id=agent_id,
            )
            yield Chunk(entry=entry, source_ids=tuple(c.data_id for c in chunk))


def group_into_sessions(
    learnings: Iterable[LearningItem],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Iterator[SessionEntry]:
    """Entries only, for callers that don't need to track source items."""
    for chunk in group_into_chunks(learnings, chunk_size=chunk_size):
        yield chunk.entry
