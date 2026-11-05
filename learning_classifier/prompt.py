"""Classifier prompts.

Kept in a dedicated module so we can iterate on prompt text without touching
the rest of the package. Versioned via PROMPT_VERSION so eval runs can be
compared apples-to-apples.

v2 broadens the signal beyond Shawn-corrective feedback. The agent often
recognizes its own mistakes / inefficiencies mid-task (self), and some
patterns are detectable from session metadata without any explicit
correction (observed: e.g., 5 retries on the same tool call, slow
convergence to a result). All three sources are now valid learning moments.
"""
from __future__ import annotations

PROMPT_VERSION = "v2.2026-05-20"


SYSTEM_PROMPT = """You analyze a captured agent session and extract "learning moments" — anything the agent (or fleet) should remember for next time. Conservative output: most sessions have nothing to learn; return `[]`.

A learning moment has a CATEGORY and a SOURCE.

CATEGORIES:
- "correction"      — the previous approach was wrong / off-target
- "new_info"        — info the agent didn't have before (about a person, system, preference, constraint)
- "strategy_pivot"  — direction or priorities changed mid-task
- "inefficiency"    — a slow / wasteful path that has a clearly better alternative
- "mistake"         — factual error, broken tool call, hallucination, wrong assumption

SOURCES (where the corrective signal came from):
- "shawn"    — Shawn explicitly told the agent ("no", "actually do X", "you misunderstood")
- "self"     — the agent recognized its own error / inefficiency mid-task and pivoted, with no
                Shawn input. ("I was looping on a 401 — turns out the bearer header was malformed.
                Set it right and continued.") This is *the most valuable signal*: the agent caught
                its own pattern. Look hard for these.
- "observed" — pattern detectable from session metadata without any explicit correction:
                retries on the same tool, long convergence, deleting+rewriting the same edit, etc.
                What got LEARNED isn't always said out loud.

OUTPUT — strict JSON array. For each moment:
{
  "category":       "correction" | "new_info" | "strategy_pivot" | "inefficiency" | "mistake",
  "source":         "shawn" | "self" | "observed",
  "what_happened":  "1-2 sentences describing what the agent did",
  "shawn_feedback": "Shawn's verbatim or near-verbatim quote that drove the lesson — REQUIRED when source=\"shawn\", OMIT when source is self or observed",
  "lesson":         "next time, do/avoid X — concrete, actionable, transferable to similar situations",
  "tags":           ["short", "string", "tags"]   // optional, max 5
}

CRITICAL RULES:
1. Output JSON ONLY. No preamble, no trailing explanation. Start with `[` end with `]`.
2. Conservative: when in doubt, omit. Empty `[]` is the right answer for most sessions.
3. Routine continuations ("ok", "thanks", "continue", "yes") are NOT learning moments — even if Shawn said them.
4. Tool-call interludes, planning blocks, debug output: NOT a learning moment unless they reveal a clear self-correction or inefficiency.
5. For source=shawn: `shawn_feedback` must be grounded in the transcript. Don't invent.
6. For source=self: cite what the agent said about its own pivot in `what_happened`. The lesson should be what the agent now knows.
7. For source=observed: `what_happened` describes the observable pattern (e.g., "made 6 separate tool calls for what should have been one batched query"); the lesson generalizes it.
8. `lesson` is forward-looking and actionable: what to do (or avoid) next time. Generalize beyond the specific situation when possible.

You output the JSON array and nothing else."""


USER_PROMPT_TEMPLATE = """Session transcript:

---
{transcript}
---

Output the JSON array of learning moments (or `[]` if none)."""


def build_messages(transcript: str) -> list[dict]:
    """Return the chat-completions-style messages list for the backend.

    Truncate very long transcripts to keep context predictable. Tail-bias the
    truncation: the end of a session is where feedback typically lands.
    """
    max_chars = 24000  # ~8k tokens at 3 chars/token, comfortable for Qwen3 8B
    if len(transcript) > max_chars:
        # Keep first 4k for setup + last 20k for feedback-rich tail
        head = transcript[:4000]
        tail = transcript[-(max_chars - 4000):]
        transcript = head + "\n\n…[truncated]…\n\n" + tail

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(transcript=transcript)},
    ]
