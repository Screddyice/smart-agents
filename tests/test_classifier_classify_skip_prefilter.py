"""The prefilter bypass for already-distilled input.

The claude-mem hub became the only source feeding the corpus. Its items carry no dialogue
turns, so the transcript heuristic rejects every one of them as
`no_shawn_turns` — which looks exactly like "nothing worth learning happened"
rather than "the corpus is not being written to at all".
"""
import json

from learning_classifier import classify
from learning_classifier.backend import BackendResult

# No Shawn turn, no claude-mem markers: the prefilter rejects this shape.
DISTILLED = (
    "- When exposing REST handlers through an MCP transport, call them "
    "in-process rather than looping back through fetch, so IP-keyed rate "
    "limiting keys on the real caller's IP.\n\n"
    "- Auth tokens should be read from the incoming request's Bearer header "
    "rather than from an environment variable, because the env var is shared "
    "across all callers and cannot be attributed.\n"
) * 3


class StubBackend:
    def __init__(self, raw="[]"):
        self.calls = 0
        self._raw = raw

    def chat(self, messages):
        self.calls += 1
        return BackendResult(raw=self._raw, model="stub", elapsed_ms=1)


def test_distilled_input_is_rejected_by_default():
    """Guards the premise: without the bypass this input never reaches the LLM."""
    backend = StubBackend()
    result = classify(DISTILLED, backend=backend)
    assert result.skipped_reason == "no_shawn_turns"
    assert backend.calls == 0


def test_skip_prefilter_lets_distilled_input_through():
    backend = StubBackend()
    result = classify(DISTILLED, backend=backend, skip_prefilter=True)
    assert backend.calls == 1
    assert result.skipped_reason != "no_shawn_turns"


def test_skip_prefilter_still_returns_parsed_learnings():
    payload = json.dumps([{
        "category": "new_info",
        "source": "self",
        "what_happened": "Called REST handlers in-process from the MCP transport.",
        "lesson": "Loop-back fetch re-keys rate limiting onto the server's egress IP.",
        "tags": ["mcp"],
    }])
    result = classify(DISTILLED, backend=StubBackend(payload), skip_prefilter=True)
    assert len(result.learnings) == 1
    assert result.learnings[0].category == "new_info"


def test_bypass_defaults_off_so_transcript_callers_are_unchanged():
    """The HyperSwarm path shares this function; the prefilter is a real cost
    saver there and must stay on unless a caller opts out."""
    backend = StubBackend()
    classify("short", backend=backend)
    assert backend.calls == 0
