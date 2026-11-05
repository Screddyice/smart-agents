"""Parse HyperSwarm markdown entries into structured session records.

HyperSwarm captures look like:

    ---
    runtime: claude-code | codex | openclaw-runs-* | claude_harness | ...
    scope: NEB | Cliqk | TRC | personal | cross
    project: <repo-name-or-empty>
    cwd: <path>
    session_id: <uuid>
    timestamp: <ISO>
    summary: <short text>
    ---

    <body — usually transcript or run summary>

This module exposes:
  parse(path) → SessionEntry | None
"""
from __future__ import annotations

import hashlib
import logging
import pathlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class SessionEntry:
    """A normalized session record extracted from one HyperSwarm entry file."""

    path: pathlib.Path
    runtime: str          # 'claude-code' | 'codex' | 'openclaw-runs-*' | 'claude_harness' | ...
    scope: str            # 'NEB' | 'Cliqk' | 'TRC' | 'personal' | 'cross'
    project: str | None   # repo name or None
    session_id: str | None
    occurred_at: datetime
    summary: str          # may be empty
    body: str             # full transcript / report
    body_hash: str        # sha256 of body, stable across reads
    agent_id: str         # derived: 'shawn-cc' | 'codex' | 'screddy' | 'clawdbot' | 'main' | …


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)
_KV_RE = re.compile(r"^([A-Za-z_]+):\s*(.*)$")


def _runtime_to_agent_id(runtime: str, body: str) -> str:
    """Map the HyperSwarm `runtime` tag to one of our canonical agent_ids.

    The `openclaw-runs-<host>` source is a special case — those entries
    contain a sessionKey like `agent:screddy:cron:...` in the body. We dig
    that out so screddy runs get the right agent_id (load-bearing for
    Pipeline B's filter).
    """
    if runtime == "claude-code":
        return "shawn-cc"
    if runtime == "codex":
        return "codex"
    if runtime == "claude_harness":
        return "claude-harness"
    if runtime == "hermes":
        return "screddy"
    if runtime.startswith("openclaw-runs"):
        m = re.search(r"agent:([a-z][a-z0-9_-]*)", body)
        if m:
            return m.group(1)
        # Fall back to "openclaw" — Pipeline A still ingests these, just
        # with a generic agent label.
        return "openclaw"
    if runtime == "openclaw":
        return "openclaw"
    return runtime  # unknown sources keep their raw runtime tag


def parse(path: pathlib.Path) -> Optional[SessionEntry]:
    """Read one HyperSwarm entry. Returns None if the file is malformed."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        log.warning("could not read %s: %s", path, e)
        return None

    m = _FRONTMATTER_RE.match(text)
    if not m:
        log.debug("no frontmatter in %s", path)
        return None

    fm_block, body = m.group(1), m.group(2)
    fm: dict[str, str] = {}
    for line in fm_block.splitlines():
        km = _KV_RE.match(line.strip())
        if km:
            fm[km.group(1).lower()] = km.group(2).strip()

    runtime = fm.get("runtime", "unknown")
    scope = fm.get("scope", "cross")
    project = fm.get("project") or None

    ts_str = fm.get("timestamp")
    try:
        occurred_at = datetime.fromisoformat(ts_str.replace("Z", "+00:00")) if ts_str else \
                      datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except (ValueError, OSError):
        occurred_at = datetime.now(timezone.utc)

    body_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
    agent_id = _runtime_to_agent_id(runtime, body)

    return SessionEntry(
        path=path,
        runtime=runtime,
        scope=scope,
        project=project,
        session_id=fm.get("session_id"),
        occurred_at=occurred_at,
        summary=fm.get("summary", "")[:500],
        body=body,
        body_hash=body_hash,
        agent_id=agent_id,
    )
