"""Pre-filter tests. Cover the four reasons + a happy path."""
from learning_classifier.prefilter import prefilter


def test_too_short_skipped():
    d = prefilter("Hello world.")
    assert d.skip
    assert d.reason == "too_short"


def test_empty_string_skipped():
    d = prefilter("")
    assert d.skip
    assert d.reason == "too_short"


def test_no_shawn_turns_skipped():
    # Pure agent output, no User/Human/Shawn marker, > 200 chars.
    transcript = (
        "Agent: I will now compose the daily report. Today is May 14, 2026. "
        "Calendar: no meetings until 3pm. Email: 7 unread, 2 urgent. "
        "Tasks: 4 in progress, 1 blocked on legal review. Health metrics: "
        "HRV stable at 38ms, sleep 6h12m. Sending the report now."
    )
    d = prefilter(transcript)
    assert d.skip
    assert d.reason == "no_shawn_turns"


def test_boilerplate_with_no_shawn_skipped():
    transcript = (
        "Daily Digest — Thursday May 14\n\n"
        "Today's edge: get to the gym. Three meetings on calendar. "
        "Inbox at 7 unread. No Shawn input today, autonomous run only. " * 4
    )
    d = prefilter(transcript)
    assert d.skip
    assert d.reason == "boilerplate"


def test_boilerplate_but_shawn_replied_is_kept():
    transcript = (
        "Daily Digest — Thursday May 14\n\n"
        "Today's edge: get to the gym. Three meetings on calendar.\n\n"
        "Shawn: actually skip the gym today, prep for the Victor call at 4pm "
        "instead. That meeting matters more than recovery.\n\n"
        "Agent: noted, repriori1tizing." * 3
    )
    d = prefilter(transcript)
    assert not d.skip
    assert d.reason == "ok"


def test_human_marker_detected():
    transcript = (
        "Human: hey can you finish the linkedin draft I started yesterday\n\n"
        "Agent: pulling it up now…\n\n"
        "Human: actually nevermind, scrap it. Different angle.\n\n"
        "Agent: ok, scrapping." + " padding " * 30
    )
    d = prefilter(transcript)
    assert not d.skip


def test_role_user_json_marker_detected():
    transcript = (
        '{"role":"user","content":"do not ping Victor before 10am, '
        "he is eastern time and gets grouchy"
        + '"}' + " padding " * 40
    )
    d = prefilter(transcript)
    assert not d.skip


# ── Distilled claude-mem-session format (2026-06-27 capture-mismatch fix) ──────

_DISTILLED = (
    "## Trigger\n\nlesson\n\n"
    "## Lesson\n\nWhen the user asks about X, verify Y before answering.\n\n"
    "## Session\n\nrequest: Fix the hyperscale PR conflict cleanly\n"
    "learned: Resolving the conflict would have been wasted effort because "
    "the branch was already superseded; check mergeability first.\n"
    "next_steps: None\n"
)


def test_distilled_session_accepted():
    # No raw Shawn turn marker, but it IS a distilled claude-mem-session with
    # request:/learned: — must reach the LLM classifier, not be skipped.
    d = prefilter(_DISTILLED)
    assert not d.skip
    assert d.reason == "ok"


def test_distilled_with_verbatim_user_header_accepted():
    # Part B (HyperSwarmAgents) adds a verbatim `## User` section; that path is
    # also accepted (via the existing Shawn-turn marker).
    body = "## User\n\nNo this is the second agent for the second X account\n\n" + _DISTILLED
    d = prefilter(body)
    assert not d.skip
    assert d.reason == "ok"


def test_distilled_section_without_fields_still_skipped():
    # A bare section header with no request:/learned: content is not a signal.
    body = (
        "## Session\n\nThis is just narrative prose with no structured fields, "
        "long enough to clear the 200-char minimum so we exercise the marker "
        "logic rather than the length gate. Nothing actionable here at all, "
        "just filler text to pad the length well beyond two hundred characters."
    )
    d = prefilter(body)
    assert d.skip
    assert d.reason == "no_shawn_turns"
