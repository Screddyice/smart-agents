import sqlite3

import pytest

from smart_agents import media_gate
from smart_agents.media_gate import HOLD, Gate

NOW = 1_800_000_000_000


def make_db():
    db = sqlite3.connect(':memory:')
    db.executescript("""
        CREATE TABLE media_attachments (id TEXT PRIMARY KEY, origin TEXT, state TEXT,
            upload_state TEXT DEFAULT 'pending', upload_error TEXT, upload_retry_at INTEGER DEFAULT 0,
            created_at INTEGER);
        CREATE TABLE observation_media_links (observation_id INTEGER, attachment_id TEXT);
        CREATE TABLE observations (id INTEGER PRIMARY KEY, type TEXT, project TEXT, title TEXT,
            subtitle TEXT, narrative TEXT, facts TEXT);
    """)
    return db


def capture(db, attachment_id, memory_id=None, origin='native', age_ms=0):
    db.execute("INSERT INTO media_attachments (id, origin, state, created_at) VALUES (?, ?, 'ready', ?)",
               (attachment_id, origin, NOW - age_ms))
    if memory_id is not None:
        db.execute('INSERT OR IGNORE INTO observations (id, type, project, title) VALUES (?, ?, ?, ?)',
                   (memory_id, 'bugfix', 'demo', f'memory {memory_id}'))
        db.execute('INSERT INTO observation_media_links VALUES (?, ?)', (memory_id, attachment_id))


def row(db, attachment_id):
    return db.execute('SELECT upload_state, upload_error, upload_retry_at FROM media_attachments WHERE id=?',
                      (attachment_id,)).fetchone()


def fixed(probability, calls=None):
    def decide(state, questions):
        if calls is not None:
            calls.append(state['text'])
        return {'id': 'gen-test', 'answers': {'worth_uploading': {'type': 'noul', 'noul': probability}}}
    return decide


def gate(db, decide):
    return Gate(db, decide=decide, now_ms=lambda: NOW)


def test_trigger_holds_new_native_screenshots_only():
    db = make_db()
    gate(db, fixed(0.0)).install_trigger()
    capture(db, 'native-1')
    capture(db, 'replica-1', origin='replica')
    assert row(db, 'native-1')[2] == HOLD
    assert row(db, 'replica-1')[2] == 0


def test_approved_screenshots_are_released_for_upload():
    db = make_db()
    g = gate(db, fixed(0.7))
    g.install_trigger()
    capture(db, 'a', memory_id=1)
    assert g.run()['uploaded'] == 1
    assert tuple(row(db, 'a')) == ('pending', None, 0)


def test_routine_screenshots_are_skipped_and_kept_locally():
    db = make_db()
    g = gate(db, fixed(0.1))
    g.install_trigger()
    capture(db, 'a', memory_id=1)
    assert g.run()['skipped'] == 1
    assert row(db, 'a')[:2] == ('failed', 'jev_skipped')


def test_one_jev_call_per_memory_covers_all_its_screenshots():
    db = make_db()
    calls = []
    g = gate(db, fixed(0.9, calls))
    g.install_trigger()
    capture(db, 'a', memory_id=1)
    capture(db, 'b', memory_id=1)
    assert g.run()['uploaded'] == 2
    assert len(calls) == 1 and 'Screenshots attached to this memory: 2' in calls[0]


def test_unlinked_screenshot_waits_then_is_skipped():
    db = make_db()
    g = gate(db, fixed(0.9))
    g.install_trigger()
    capture(db, 'fresh')
    capture(db, 'stale', age_ms=media_gate.WAIT_FOR_LINK_MS)
    counts = g.run()
    assert counts == {'uploaded': 0, 'skipped': 1, 'deferred': 1}
    assert row(db, 'fresh')[2] == HOLD
    assert row(db, 'stale')[:2] == ('failed', 'jev_skipped')


def failing(state, questions):
    raise TimeoutError('JEV response timed out; not retried')


@pytest.mark.parametrize('age_ms, expected', [(0, 'deferred'), (media_gate.GIVE_UP_MS, 'skipped')])
def test_jev_failure_defers_then_gives_up_after_a_day(age_ms, expected):
    db = make_db()
    g = gate(db, failing)
    g.install_trigger()
    capture(db, 'a', memory_id=1, age_ms=age_ms)
    assert g.run()[expected] == 1


def test_uninstall_drops_trigger_and_releases_held_screenshots():
    db = make_db()
    g = gate(db, fixed(0.0))
    g.install_trigger()
    capture(db, 'held')
    g.uninstall()
    assert row(db, 'held')[2] == 0
    capture(db, 'after')
    assert row(db, 'after')[2] == 0


def test_memory_text_is_bounded():
    memory = {'type': 'discovery', 'project': 'p', 'title': 't' * 500, 'subtitle': None,
              'narrative': 'n' * 5000, 'facts': ''}
    text = media_gate.memory_text(memory, 1)
    assert len(text) < 2300 and 'Subtitle' not in text
