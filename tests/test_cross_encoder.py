import os
import pytest
pytestmark = pytest.mark.skipif(os.environ.get("SMART_AGENTS_MODEL_TESTS") != "1", reason="explicit model integration tests; requires a downloaded model")

torch = pytest.importorskip("torch")
from smart_agents.rerank.cross_encoder import CrossEncoderRanker  # noqa: E402


@pytest.fixture(scope="module")
def ranker():
    return CrossEncoderRanker(device="cpu")


def test_relevant_lesson_outscores_unrelated_one(ranker):
    task = "I am about to run a database migration that renames a role"
    lessons = [
        "When resolving merge conflicts involving imports, compare both sides symbol by symbol.",
        "Before applying migrations, first create or validate all referenced database roles.",
    ]
    a, b = ranker.score(task, lessons)
    assert b > a


def test_rank_applies_threshold_and_order(ranker):
    task = "How do I tune a classical guitar to drop D"
    cands = [{"id": 1, "lesson": "Use viewport units for full-height CSS layouts."},
             {"id": 2, "lesson": "Run the full test suite before opening a pull request."}]
    assert ranker.rank(task, cands, k=3, min_score=-5.0) == []
    ranked = ranker.rank(task, cands, k=3, min_score=-100.0)
    assert [c["id"] for c in ranked] and all("score" in c for c in ranked)
