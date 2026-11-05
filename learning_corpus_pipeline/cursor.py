"""Per-pipeline cursor state — small JSON file in ~/.learning-corpus/.

Tracks what we've already processed so a daemon restart doesn't redo work
or skip new entries. Idempotency at the DB level (unique constraint) is
the backstop; the cursor is the optimization that keeps us from re-doing
classifier work.

Stored format:
    {
      "last_processed_hash": "abcd…",        // SHA-256 of last entry's body
      "last_processed_path": "<filename>",    // human-readable for logs
      "last_processed_at":   "2026-05-15T…",  // ISO timestamp
      "processed_count":     12345           // running total since install
    }
"""
from __future__ import annotations

import json
import os
import pathlib
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional


@dataclass
class CursorState:
    last_processed_hash: Optional[str] = None
    last_processed_path: Optional[str] = None
    last_processed_at:   Optional[str] = None
    processed_count: int = 0


class Cursor:
    """Atomic-write JSON cursor."""

    def __init__(self, path: pathlib.Path):
        self.path = path
        self.state = self._load()

    def _load(self) -> CursorState:
        if not self.path.exists():
            return CursorState()
        try:
            return CursorState(**json.loads(self.path.read_text()))
        except (json.JSONDecodeError, TypeError) as e:
            # Corrupted file — keep it as a backup, start fresh.
            backup = self.path.with_suffix(".json.corrupt")
            self.path.rename(backup)
            return CursorState()

    def mark(self, body_hash: str, path_name: str) -> None:
        """Update cursor after successfully processing one entry."""
        self.state.last_processed_hash = body_hash
        self.state.last_processed_path = path_name
        self.state.last_processed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.state.processed_count += 1
        self._save()

    def already_processed(self, body_hash: str) -> bool:
        """True if this body_hash matches the last-processed one.

        Not a full history check — the DB unique constraint catches anything
        the cursor misses. The cursor's job is just to keep startup catch-up
        fast.
        """
        return body_hash == self.state.last_processed_hash

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Atomic write: temp file in same dir + rename.
        fd, tmp_name = tempfile.mkstemp(dir=self.path.parent, prefix=".cursor-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(asdict(self.state), f, indent=2)
            os.chmod(tmp_name, 0o600)
            os.replace(tmp_name, self.path)
        except Exception:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
            raise
