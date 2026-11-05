"""Parser tests. Cover the strict / isolate / repair / fallback ladder
and the always-safe contract (never raises, returns list[Learning])."""
from learning_classifier.parser import parse_learnings


def test_empty_input_returns_empty():
    assert parse_learnings("") == []


def test_strict_json_array_parsed():
    raw = """[
      {
        "category": "correction",
        "what_happened": "Agent emailed Victor at 7am ET",
        "shawn_feedback": "don't ping Victor before 10am",
        "lesson": "send to Victor only after 10am ET",
        "tags": ["timezone","relationships"]
      }
    ]"""
    learnings = parse_learnings(raw)
    assert len(learnings) == 1
    assert learnings[0].category == "correction"
    assert "Victor" in learnings[0].shawn_feedback


def test_empty_array_returned_unchanged():
    assert parse_learnings("[]") == []


def test_prose_around_json_extracted():
    raw = """Here is the JSON:

    [
      {
        "category": "mistake",
        "what_happened": "Agent claimed Q2 revenue was $1.2M when it was $1.5M",
        "shawn_feedback": "no, Q2 was 1.5M not 1.2M",
        "lesson": "double-check revenue figures against the source-of-truth dashboard before quoting",
        "tags": []
      }
    ]

    That should cover it.
    """
    learnings = parse_learnings(raw)
    assert len(learnings) == 1
    assert learnings[0].category == "mistake"


def test_fenced_code_block_extracted():
    raw = """```json
    [{"category":"new_info","what_happened":"didn't know meeting moved","shawn_feedback":"the Tuesday standup moved to 9am","lesson":"check calendar for Tuesday standup time","tags":[]}]
    ```"""
    learnings = parse_learnings(raw)
    assert len(learnings) == 1
    assert learnings[0].category == "new_info"


def test_dict_response_promoted_to_list():
    # Some models return a single dict instead of an array.
    raw = """{
      "category": "strategy_pivot",
      "what_happened": "focused on LinkedIn outbound this week",
      "shawn_feedback": "let's pivot to cold email instead this week",
      "lesson": "shift outbound channel from LinkedIn to email for this sprint",
      "tags": ["gtm"]
    }"""
    learnings = parse_learnings(raw)
    assert len(learnings) == 1
    assert learnings[0].category == "strategy_pivot"


def test_invalid_json_returns_empty_does_not_raise():
    raw = "haha I forgot to output JSON, my bad"
    learnings = parse_learnings(raw)
    assert learnings == []


def test_bad_category_skipped_others_kept():
    raw = """[
      {
        "category": "definitely_not_a_real_category",
        "what_happened": "x",
        "shawn_feedback": "y",
        "lesson": "z"
      },
      {
        "category": "inefficiency",
        "what_happened": "looped through items one at a time",
        "shawn_feedback": "batch this, don't loop",
        "lesson": "prefer batched API calls when working over a list",
        "tags": ["efficiency"]
      }
    ]"""
    learnings = parse_learnings(raw)
    # First entry fails enum validation; second one survives.
    assert len(learnings) == 1
    assert learnings[0].category == "inefficiency"
