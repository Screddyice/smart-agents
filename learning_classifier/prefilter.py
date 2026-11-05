"""Rule-based pre-filter that skips sessions with no chance of containing
a learning moment, before we spend an LLM call on them.

Goal: cut classifier workload by ~80% so daily-batch jobs fit in their
window without losing real signal. Conservative rules only — when in doubt
we let it through to the LLM, which is the more accurate (but expensive)
filter.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Markers that strongly indicate a Shawn-authored turn in a captured
# transcript. We accept multiple capture formats:
#   - HyperSwarm openclaw_runs entries (no Shawn turns at all — drop)
#   - Claude Code session transcripts (use `User:` / `Human:` prefixes)
#   - codex CLI dumps (similar)
#   - openclaw .jsonl sessions (json messages with role=user)
_SHAWN_MARKERS = [
    # Speaker-prefix style: "User:", "Human:", "Shawn:"
    re.compile(r"^\s*(?:User|Human|Shawn)\s*:", re.MULTILINE | re.IGNORECASE),
    # Markdown-header style (HyperSwarm + Claude Code formats):
    # "## User", "## Human", "## Shawn"
    re.compile(r"^#{1,6}\s+(?:User|Human|Shawn)\b", re.MULTILINE | re.IGNORECASE),
    # JSON role markers
    re.compile(r'"role"\s*:\s*"user"'),
    re.compile(r"<user>", re.IGNORECASE),
]

# The live capture path is the LLM-DISTILLED claude-mem-session source, which
# emits `## Trigger / ## Lesson / ## Session` with `request:` / `learned:`
# summary fields instead of raw turns — so it carries NONE of the _SHAWN_MARKERS
# above. Before this, every distilled session was skipped 'no_shawn_turns' and
# the learnings table went flat (the corpus capture-format mismatch, 2026-06-27).
# Treat a distilled session overview as a valid learning signal: its `request:`
# field IS Shawn's intent (paraphrased), and `learned:` is candidate lesson
# material — let the conservative LLM classifier decide from there.
_DISTILLED_SECTION = re.compile(
    r"^#{1,6}\s+(?:Session|Lesson|Trigger)\b", re.MULTILINE | re.IGNORECASE
)
_DISTILLED_FIELD = re.compile(
    r"^\s*(?:request|learned)\s*:\s*\S", re.MULTILINE | re.IGNORECASE
)

# Substrings that mark this as a boilerplate cron output with no user input.
_BOILERPLATE_MARKERS = [
    "Daily Digest",
    "no user activity",
    "no Shawn input",
    "cron job summary",
]


@dataclass
class PrefilterDecision:
    """Result of running the pre-filter."""

    skip: bool
    reason: str   # one of: "ok" | "no_shawn_turns" | "too_short" | "boilerplate"


def prefilter(transcript: str) -> PrefilterDecision:
    """Decide whether to run the LLM classifier on this transcript.

    Returns `skip=False, reason="ok"` if the transcript should be classified.
    Anything else means we skip the LLM call entirely.
    """
    if not transcript or len(transcript) < 200:
        return PrefilterDecision(skip=True, reason="too_short")

    # Boilerplate detection: openclaw cron digests typically declare themselves.
    head_window = transcript[:2000]
    if any(b.lower() in head_window.lower() for b in _BOILERPLATE_MARKERS):
        # Boilerplate cron digests rarely contain Shawn-correction moments.
        # We still let them through IF a Shawn marker is present (in case Shawn
        # later commented on the digest in the same session).
        if not _has_learning_signal(transcript):
            return PrefilterDecision(skip=True, reason="boilerplate")

    if not _has_learning_signal(transcript):
        return PrefilterDecision(skip=True, reason="no_shawn_turns")

    return PrefilterDecision(skip=False, reason="ok")


def _has_shawn_turn(transcript: str) -> bool:
    """True if any marker for a Shawn-authored turn appears in the transcript."""
    return any(rx.search(transcript) for rx in _SHAWN_MARKERS)


def _has_distilled_session(transcript: str) -> bool:
    """True for the LLM-distilled claude-mem-session format (## Session/Lesson
    plus a non-empty request:/learned: field)."""
    return bool(_DISTILLED_SECTION.search(transcript) and _DISTILLED_FIELD.search(transcript))


def _has_learning_signal(transcript: str) -> bool:
    """Accept if the transcript carries either a raw Shawn turn or a distilled
    claude-mem-session overview worth sending to the LLM classifier."""
    return _has_shawn_turn(transcript) or _has_distilled_session(transcript)
