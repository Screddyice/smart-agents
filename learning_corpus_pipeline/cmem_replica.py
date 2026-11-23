"""Read scoped, distilled observations from an operator-paired CMEM replica."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path, PurePosixPath

# Local guardrails for routine credentials and high-risk subjects. Workspace roots
# must already be authorized for the configured JEV provider and recall audience.
_RESTRICTED = re.compile(r"\b(?:confidential|hipaa|medical|patient|payroll|private)\b", re.I)
_CREDENTIAL = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r"|\b(?:password|passwd|api[_ -]?key|access[_ -]?token|secret)\s*[:=]"
    r"|\b(?:sk-(?:proj-)?[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9_]{12,})"
    r"|\bBearer\s+[A-Za-z0-9._~+/-]{12,}"
    r"|://[^/\s@:]+:[^/\s@]+@",
    re.I,
)
_LABEL = re.compile(r"[\w][\w .-]{0,119}")


def readonly_uri(path: Path) -> str:
    return path.expanduser().resolve().as_uri() + "?mode=ro"


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def read_observations(db: sqlite3.Connection, cursor: int, limit: int) -> list[sqlite3.Row]:
    columns = {row[1] for row in db.execute("PRAGMA table_info(observations)")}
    session_columns = {row[1] for row in db.execute("PRAGMA table_info(sdk_sessions)")}
    optional = ("subtitle", "metadata", "origin_device_id", "origin_local_id")
    extras = ", ".join(name if name in columns else f"NULL AS {name}" for name in optional)
    cwd = "NULL"
    if "memory_session_id" in columns and {"id", "cwd", "memory_session_id"} <= session_columns:
        cwd = ("(SELECT s.cwd FROM sdk_sessions s WHERE s.memory_session_id = o.memory_session_id "
               "AND s.cwd IS NOT NULL ORDER BY s.id DESC LIMIT 1)")
    return db.execute(
        f"""SELECT id, project, type, title, narrative, created_at_epoch, files_read,
                   files_modified, {extras}, {cwd} AS cwd
            FROM observations o WHERE id > ? ORDER BY id LIMIT ?""",
        (cursor, limit),
    ).fetchall()


def format_excerpt(title: str | None, narrative: str | None) -> str:
    title = " ".join((title or "").split())
    narrative = " ".join((narrative or "").split())
    if len(narrative) > 600:
        prefix = narrative[:600]
        sentence_end = max(prefix.rfind(separator) for separator in (". ", "! ", "? "))
        narrative = prefix[:sentence_end + 1] if sentence_end >= 300 else prefix.rpartition(" ")[0]
    return f"{title}\n{narrative}" if narrative else title


def _source_paths(row: sqlite3.Row) -> list[str]:
    paths = []
    if row["cwd"]:
        paths.append(row["cwd"])
    if row["subtitle"] == "Manual memory" and row["metadata"]:
        metadata = json.loads(row["metadata"])
        if not isinstance(metadata, dict):
            raise ValueError("Expected an observation metadata object")
        if metadata.get("cwd") is not None:
            paths.append(metadata["cwd"])
    if any(not isinstance(path, str) or not PurePosixPath(path).is_absolute() for path in paths):
        raise ValueError("Expected absolute working directories")
    for field in ("files_read", "files_modified"):
        files = json.loads(row[field] if row[field] is not None else "[]")
        if not isinstance(files, list) or any(not isinstance(path, str) for path in files):
            raise ValueError("Expected a list of source paths")
        paths.extend(files)
    return paths


def eligible_excerpt(
    row: sqlite3.Row, roots: tuple[PurePosixPath, ...], excluded_terms: tuple[str, ...]
) -> str | None:
    project, kind = row["project"], row["type"]
    if not all(isinstance(value, str) and _LABEL.fullmatch(value) for value in (project, kind)):
        return None
    title, narrative = row["title"], row["narrative"]
    if not isinstance(title, str) or not 18 <= len(" ".join(title.split())) <= 300:
        return None
    if narrative is not None and not isinstance(narrative, str):
        return None
    try:
        paths = _source_paths(row)
    except (TypeError, ValueError):
        return None

    absolute = []
    for value in paths:
        path = PurePosixPath(value)
        if ".." in path.parts or "\x00" in value:
            return None
        if path.is_absolute():
            absolute.append(path)
    if not absolute or not all(any(path.is_relative_to(root) for root in roots) for path in absolute):
        return None

    text = "\n".join((project, kind, title, narrative or "", *paths))
    if _RESTRICTED.search(text) or _CREDENTIAL.search(text):
        return None
    if any(term.casefold() in text.casefold() for term in excluded_terms):
        return None
    return format_excerpt(title, narrative)
