"""Read-only MCP lookup of published, JEV-approved CMEM excerpts."""
from __future__ import annotations

import argparse
import heapq
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from learning_corpus_pipeline.cmem_replica import digest, format_excerpt, readonly_uri

_WORD = re.compile(r"[0-9]+(?:\.[0-9]+)+|[^\W_]+")
_STOP = frozenset("about a an and are can could did do for from has have how i in is it me of on "
                  "or please should tell that the this to was were what when where which who with you".split())


@dataclass(frozen=True)
class Memory:
    source_id: int
    corpus_id: int
    project: str
    excerpt: str


def _terms(text: str) -> set[str]:
    return {word for word in _WORD.findall(text.casefold()) if len(word) > 1 and word not in _STOP}


def _matching_memories(db, terms):
    rows = db.execute(
        """SELECT a.source_id, a.corpus_id, a.project, a.excerpt, a.excerpt_sha256,
                  o.title, o.narrative
           FROM main.approved a
           JOIN replica.observations o
             ON o.id = a.source_id AND o.project = a.project AND o.type = a.kind
           WHERE a.receipt != ''"""
    )
    for row in rows:
        excerpt = row["excerpt"]
        score = len(terms & _terms(f"{row['project']} {excerpt}"))
        if not score or not all(value is None or isinstance(value, str)
                                for value in (row["title"], row["narrative"])):
            continue
        if excerpt != format_excerpt(row["title"], row["narrative"]) or digest(excerpt) != row["excerpt_sha256"]:
            continue
        yield score, Memory(row["source_id"], row["corpus_id"], row["project"], excerpt)


def recall(source: Path, publication: Path, query: str, limit: int = 5) -> list[Memory]:
    if not isinstance(query, str) or not query.strip() or len(query) > 1000:
        raise ValueError("Query must contain 1 to 1000 characters")
    if type(limit) is not int or not 1 <= limit <= 10:
        raise ValueError("Limit must be an integer from 1 to 10")
    terms = _terms(query)
    if not terms:
        return []
    with closing(sqlite3.connect(readonly_uri(publication), uri=True)) as db:
        db.row_factory = sqlite3.Row
        db.execute("ATTACH DATABASE ? AS replica", (readonly_uri(source),))
        db.execute("PRAGMA query_only = ON")
        ranked = heapq.nlargest(limit, _matching_memories(db, terms),
                               key=lambda hit: (hit[0], hit[1].source_id))
    return [memory for _, memory in ranked]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--publication-db", type=Path, required=True)
    args = parser.parse_args(argv)
    server = FastMCP("smart-agents-cmem")

    @server.tool(annotations={"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False})
    def recall_cmem_memory(query: str, limit: int = 5) -> str:
        """Recall approved operational facts from past work, based on a project, version, or topic."""
        try:
            rows = recall(args.source_db, args.publication_db, query, limit)
        except sqlite3.Error:
            raise RuntimeError("CMEM recall storage is unavailable") from None
        return "\n\n".join(
            f"CMEM observation #{row.source_id} · Corpus lesson #{row.corpus_id} · {row.project}\n{row.excerpt}"
            for row in rows
        ) or "No matching approved CMEM memories."

    server.run(transport="stdio")


if __name__ == "__main__":
    main()
