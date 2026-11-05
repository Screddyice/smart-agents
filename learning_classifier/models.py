"""Pydantic models for the classifier output.

`Learning` is what `classify()` emits per detected event. Field shapes mirror
the `learnings` table on RDS so a writer can map row-for-row.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


Category = Literal[
    "correction",      # explicit "no, do X" / "you're wrong"
    "new_info",        # info the agent didn't have before
    "strategy_pivot",  # direction / priorities changed
    "inefficiency",    # slow / wasteful path; better way found
    "mistake",         # factual error, broken tool call, hallucination
]


# Where the corrective signal came from. Broader than just "Shawn corrected" —
# we want to capture the agent's own mid-task realizations and patterns
# detectable from session metadata, not just user feedback.
Source = Literal[
    "shawn",     # Shawn explicitly told the agent (legacy default)
    "self",      # the agent recognized its own error / inefficiency mid-task
    "observed",  # pattern detectable from session metadata (retries, slow convergence)
]


class Learning(BaseModel):
    """One distilled learning extracted from a session.

    Stored row-for-row into RDS.learnings. Fields populated by the caller (writer)
    are intentionally NOT on this model:

      - id, occurred_at, ingested_at, org, instance, agent_id, project,
        session_id, source_session_id, source_entry_hash, embedding

    `shawn_feedback` is optional because self/observed learnings have no
    user quote to attribute. The DB schema column is NULLABLE (see
    sql/002_self_noticed_source.sql).
    """

    category: Category
    source: Source = "shawn"  # default preserves back-compat with v1 prompt outputs
    what_happened: str = Field(..., min_length=1, max_length=2000)
    shawn_feedback: Optional[str] = Field(default=None, max_length=2000)
    lesson: str = Field(..., min_length=1, max_length=2000)
    tags: list[str] = Field(default_factory=list, max_length=20)


class ClassifierResult(BaseModel):
    """Wraps the list[Learning] with diagnostic metadata about the run.

    Useful for telemetry and debugging the eval set. Writers only care about
    `.learnings`; eval tooling looks at the whole envelope.
    """

    learnings: list[Learning] = Field(default_factory=list)
    skipped_reason: Optional[str] = None   # set when prefilter or backend skips
    backend_model: Optional[str] = None
    elapsed_ms: Optional[int] = None
