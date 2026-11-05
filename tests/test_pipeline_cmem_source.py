"""claude-mem hub rows -> learning items.

The silent failure to guard here is ingesting the wrong kind of row. The hub's
timeline carries every synced user prompt alongside the distilled observations
and summaries; prompts are Shawn's raw input and must never become learnings.
"""
from learning_corpus_pipeline.cmem_source import LEARNING_KINDS, _row_text, parse_row


def _row(kind, **extra):
    base = {"id": f"{kind}:abc123", "kind": kind, "title": "Webhook secrets", "snippet": "HMAC secret is required.",
            "project": "nebos-v2", "created_at_epoch": 1788000000}
    base.update(extra)
    return base


def test_observation_and_summary_rows_become_items():
    for kind in LEARNING_KINDS:
        item = parse_row(_row(kind))
        assert item is not None
        assert item.data_id == f"cmem:{kind}:abc123"


def test_prompt_rows_are_dropped_even_with_text():
    """906 of ~920 timeline rows are prompts; reading snippets gave them text."""
    assert parse_row(_row("prompt")) is None


def test_rows_without_a_kind_are_still_parsed():
    r = _row("observation"); del r["kind"]
    assert parse_row(r) is not None


def test_literal_none_title_is_not_text():
    assert _row_text({"title": "None", "snippet": "None"}) == ""
    assert _row_text({"title": "None", "snippet": "real body"}) == "real body"


def test_enriched_summary_fields_are_read_once():
    text = _row_text({"title": "T", "learned": "L", "snippet": "L", "next_steps": ["a", "b"]})
    assert text == "T\nL\na\nb"
