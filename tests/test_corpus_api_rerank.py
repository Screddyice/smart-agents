import time

from corpus_api.rerank import Reranker, Settings, dedupe


def _s(**kw):
    base = dict(enabled=True, model="fake", min_score=-5.0, pool=12, budget_s=1.5, threads=1, dedupe_cos=0.85)
    base.update(kw)
    return Settings(**base)


def _c(cid, lesson, vec):
    return {"id": cid, "lesson": lesson, "embedding": vec}


def test_orders_by_score_and_drops_below_min_score():
    scorer = lambda task, lessons: [float(l.split("=")[1]) for l in lessons]  # noqa: E731
    r = Reranker(_s(), scorer=scorer)
    out = r.rerank("t", [_c(1, "s=1", [1, 0]), _c(2, "s=9", [0, 1]), _c(3, "s=-7", [0, 0, 1])], k=3)
    assert [c["id"] for c in out] == [2, 1]
    assert out[0]["score"] == 9.0


def test_dedupe_skips_near_twins_in_order():
    rows = [_c(1, "a", [1.0, 0.0]), _c(2, "a again", [0.99, 0.14]), _c(3, "b", [0.0, 1.0])]
    assert [c["id"] for c in dedupe(rows, k=2, max_cos=0.85)] == [1, 3]


def test_pool_cap_limits_how_many_rows_are_scored():
    seen = []
    scorer = lambda task, lessons: seen.append(len(lessons)) or [0.0] * len(lessons)  # noqa: E731
    Reranker(_s(pool=2), scorer=scorer).rerank("t", [_c(i, "x", [1, 0]) for i in range(5)], k=3)
    assert seen == [2]


def test_returns_none_when_disabled():
    assert Reranker(_s(enabled=False), scorer=lambda t, l: [1.0]).rerank("t", [_c(1, "x", [1, 0])], k=3) is None


def test_returns_none_when_scorer_raises():
    def boom(task, lessons):
        raise RuntimeError("model missing")
    assert Reranker(_s(), scorer=boom).rerank("t", [_c(1, "x", [1, 0])], k=3) is None


def test_returns_none_when_scorer_is_too_slow():
    def slow(task, lessons):
        time.sleep(0.2)
        return [1.0] * len(lessons)
    assert Reranker(_s(budget_s=0.05), scorer=slow).rerank("t", [_c(1, "x", [1, 0])], k=3) is None


def test_settings_from_env(monkeypatch):
    from corpus_api.rerank import settings_from_env
    monkeypatch.setenv("RERANK_ENABLED", "0")
    monkeypatch.setenv("RERANK_MIN_SCORE", "-3.5")
    s = settings_from_env()
    assert s.enabled is False and s.min_score == -3.5 and s.pool == 12


def test_warm_swallows_a_failed_model_load(monkeypatch):
    import corpus_api.rerank as mod

    def broken(model, threads):
        raise RuntimeError("no torch here")
    monkeypatch.setattr(mod, "_load_cross_encoder", broken)
    r = Reranker(_s())
    assert r.warm() is False
    assert r.rerank("t", [_c(1, "x", [1, 0])], k=3) is None  # still falls back, never raises


def test_warm_is_a_no_op_when_disabled_or_already_loaded():
    assert Reranker(_s(enabled=False)).warm() is False
    assert Reranker(_s(), scorer=lambda t, l: [1.0]).warm() is True
