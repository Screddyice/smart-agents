"""claude-mem (cmem.ai) → corpus source.

Pages the hub-authoritative `timeline` MCP tool newest-first with its `nextCursor`,
stopping once a whole page is already in the cursor. Every device that syncs
(Mac, src, r2h, TMN) is covered.
Rows are already-compressed observations and summaries, so the classifier runs
with skip_prefilter=True: there are no dialogue turns to prefilter on.

Auth: CMEM_PRO_TOKEN (Bearer) from the environment; the unit gets it from
~/src-neb/credentials.env. Never logged.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator, Optional

from .learning_items import LearningItem

log = logging.getLogger(__name__)
MCP_URL = os.environ.get("CMEM_MCP_URL", "https://cmem.ai/api/mcp")
_TIMEOUT = 60.0
_PAGE = 100


class CmemClient:
    def __init__(self, token: str, url: str = MCP_URL):
        if not token:
            raise RuntimeError("cmem ingest needs CMEM_PRO_TOKEN in the unit environment")
        self._token, self._url, self._id = token, url, 0

    def _call(self, tool: str, args: dict) -> dict:
        self._id += 1
        body = {"jsonrpc": "2.0", "id": self._id, "method": "tools/call", "params": {"name": tool, "arguments": args}}
        req = urllib.request.Request(self._url, data=json.dumps(body).encode(), method="POST", headers={
            "Authorization": f"Bearer {self._token}", "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
            raw = r.read().decode()
        for line in raw.splitlines():
            line = line[5:].strip() if line.startswith("data:") else line.strip()
            if line.startswith("{"):
                d = json.loads(line)
                if d.get("error"):
                    raise RuntimeError(f"cmem {tool}: {d['error']}")
                res = d.get("result", {})
                if res.get("isError"):
                    raise RuntimeError(f"cmem {tool}: {res}")
                text = next((c.get("text", "") for c in res.get("content", []) if c.get("type") == "text"), "{}")
                return json.loads(text) if text.strip().startswith("{") else {"rows": [], "text": text}
        return {"rows": []}

    def get_summaries(self, ids: list[str]) -> dict:
        """Full summary rows (learned/investigated/...) for a batch of ids; timeline only carries snippets."""
        out: dict = {}
        for i in range(0, len(ids), 20):
            res = self._call("get_summaries", {"ids": ids[i:i + 20]})
            for r in (res.get("rows") or res.get("summaries") or []):
                rid = str(r.get("id") or "")
                if rid:
                    out[rid] = r
        return out

    def timeline(self, before: Optional[str] = None, limit: int = _PAGE) -> dict:
        args: dict = {"limit": limit}
        if before:
            args["before"] = before
        return self._call("timeline", args)


def _clean(v) -> str:
    # the hub renders missing titles as the literal string "None"
    return "" if v is None or str(v).strip() in ("", "None", "null") else str(v).strip()


def _row_text(row: dict) -> str:
    """Timeline rows carry `title` + `snippet`; enriched summary rows add the full
    `learned` / `investigated` / `completed` / `next_steps` fields."""
    parts = [_clean(row.get("title"))]
    for k in ("learned", "investigated", "completed", "next_steps", "narrative", "text", "body", "snippet"):
        v = row.get(k)
        if isinstance(v, list):
            v = "\n".join(str(x) for x in v)
        parts.append(_clean(v))
    seen, out = set(), []
    for p in parts:
        if p and p not in seen:
            seen.add(p); out.append(p)
    return "\n".join(out).strip()


def _row_time(row: dict) -> datetime:
    for k in ("created_at", "createdAt", "created_at_epoch", "timestamp"):
        v = row.get(k)
        if isinstance(v, (int, float)):
            return datetime.fromtimestamp(v / (1000 if v > 1e12 else 1), tz=timezone.utc)
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v.replace("Z", "+00:00"))
            except ValueError:
                pass
    return datetime.now(tz=timezone.utc)


# Only distilled rows are learnings. The timeline also carries every synced user
# prompt (`kind == "prompt"`, ~900 of ~920 rows), which is Shawn's raw input,
# not something learned. Until snippets were read those rows had no text and
# fell out by accident; the filter makes it deliberate.
LEARNING_KINDS = ("observation", "summary")


def parse_row(row: dict) -> Optional[LearningItem]:
    kind = row.get("kind")
    if kind and kind not in LEARNING_KINDS:
        return None
    text = _row_text(row)
    rid = str(row.get("id") or row.get("observation_id") or "")
    if not text or not rid:
        return None
    agent = str(row.get("agent_type") or row.get("platformSource") or row.get("source") or "cmem")
    session = str(row.get("memory_session_id") or row.get("session_id") or rid)
    return LearningItem(data_id=f"cmem:{rid}", session_key=f"{agent}_{session}", occurred_at=_row_time(row), text=text)


def fetch_learnings(client: CmemClient, *, since: Optional[datetime] = None, limit: Optional[int] = None,
                    skip_ids: Optional[set[str]] = None) -> list[LearningItem]:
    """Newest-first pages until a full page is already known or `since` is passed."""
    skip_ids = skip_ids or set()
    out: list[LearningItem] = []
    cursor: Optional[str] = None
    visited: set[str] = set()
    while True:
        page = client.timeline(before=cursor)
        rows = page.get("rows") or []
        if not rows:
            break
        fresh = 0
        sum_ids = [str(r.get("id")) for r in rows if r.get("kind") == "summary" and r.get("id")]
        full = client.get_summaries(sum_ids) if sum_ids else {}
        for row in rows:
            if row.get("kind") == "summary" and str(row.get("id")) in full:
                row = {**row, **full[str(row.get("id"))]}
            item = parse_row(row)
            if item is None:
                continue
            if since and item.occurred_at < since:
                rows = []  # signal stop after this page
                break
            if item.data_id in skip_ids:
                continue
            fresh += 1
            out.append(item)
            if limit and len(out) >= limit:
                rows = []
                break
        cursor = page.get("nextCursor")
        if not rows or not cursor:
            break
        if cursor in visited:
            raise RuntimeError("Source pagination repeated a cursor")
        visited.add(cursor)
    out.sort(key=lambda i: i.occurred_at)
    log.info("cmem: %d new learning rows fetched", len(out))
    return out
