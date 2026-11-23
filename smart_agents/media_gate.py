"""JEV review gate for claude-mem screenshot uploads.

claude-mem uploads every captured screenshot to its cloud. Most are routine. This gate
adds a SQLite trigger that holds each new native screenshot (``upload_retry_at = HOLD``),
then asks JEV, from the text of the memory the screenshot belongs to, whether it is worth
uploading. Approved screenshots are released (``upload_retry_at = 0``); the rest are marked
``upload_state='failed'``, ``upload_error='jev_skipped'`` and stay on the local machine.

JEV sees text only, never pixels. Run it on a schedule (see deploy/launchd); nothing runs on
installation. The trigger and the schedule must be enabled and disabled together: without
the schedule, held screenshots are never reviewed.
"""
import argparse
import os
import sqlite3
import time
from pathlib import Path

from smart_agents import judgments

HOLD = 4102444800000            # 2100-01-01 in ms, a sentinel meaning "held for JEV review"
DEFAULT_UPLOAD_AT = 0.50        # routine screenshots score ~0.1; a clear visual bug ~0.7
WAIT_FOR_LINK_MS = 30 * 60_000  # claude-mem links a screenshot to its memory shortly after capture
GIVE_UP_MS = 24 * 3_600_000     # if JEV keeps failing for a day, skip rather than hold forever
MAX_MEMORIES_PER_RUN = 25

TRIGGER_NAME = 'jev_gate_hold_new_media'
TRIGGER = f"""CREATE TRIGGER IF NOT EXISTS {TRIGGER_NAME}
AFTER INSERT ON media_attachments WHEN NEW.origin='native'
BEGIN UPDATE media_attachments SET upload_retry_at={HOLD}
      WHERE id=NEW.id AND upload_state='pending'; END"""

QUESTIONS = {'worth_uploading': {
    'type': 'noul',
    'instructions': ('Decide whether the screenshot attached to this coding-agent memory is worth uploading '
        'to long-term cloud memory. Most screenshots are NOT worth it. Answer true only when the image '
        "is clearly the main evidence of something durable that the memory's text cannot capture."),
    'criteria': {
        'true': ('Shows a visual result or problem someone will want to see later: a UI or layout bug, a finished '
                 'design or visual output, a meaningful chart or dashboard state, or an error whose on-screen '
                 'appearance matters.'),
        'false': ('Routine or redundant: navigating, page loads, confirming a click or setting worked, login or auth '
                  'screens, terminal or text output already captured as text, intermediate verification steps, '
                  'repeats of earlier screens, or anything likely to show secrets or personal data.')}}}


def default_db():
    return Path(os.environ.get('SMART_AGENTS_CMEM_DB', Path.home() / '.claude-mem' / 'claude-mem.db'))


def default_log():
    return Path(os.environ.get('SMART_AGENTS_MEDIA_GATE_LOG', Path.home() / '.claude-mem' / 'logs' / 'jev-media-gate.log'))


def memory_text(memory, image_count):
    def cut(value, limit):
        return (value or '').strip()[:limit]
    parts = [f"Memory type: {memory['type']}", f"Project: {memory['project']}", f"Title: {cut(memory['title'], 200)}"]
    if memory['subtitle']:
        parts.append(f"Subtitle: {cut(memory['subtitle'], 300)}")
    if memory['narrative']:
        parts.append(f"Narrative: {cut(memory['narrative'], 1800)}")
    if memory['facts']:
        parts.append(f"Facts: {cut(memory['facts'], 800)}")
    parts.append(f'Screenshots attached to this memory: {image_count}')
    return '\n'.join(parts)


class Gate:
    def __init__(self, db, log_path=None, decide=judgments.decide, upload_at=DEFAULT_UPLOAD_AT, now_ms=None):
        self.db = db
        self.db.row_factory = sqlite3.Row
        self.log_path = log_path
        self.decide = decide
        self.upload_at = upload_at
        self.now_ms = now_ms or (lambda: time.time() * 1000)

    def log(self, message):
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open('a') as handle:
            handle.write(time.strftime('[%Y-%m-%d %H:%M:%S] ') + message + '\n')

    def install_trigger(self):
        self.db.execute(TRIGGER)
        self.db.commit()

    def release(self, ids):
        self.db.executemany("UPDATE media_attachments SET upload_retry_at=0 WHERE id=? AND upload_state='pending'",
                            [(i,) for i in ids])
        self.db.commit()

    def skip(self, ids, why):
        self.db.executemany("UPDATE media_attachments SET upload_state='failed', upload_error='jev_skipped' "
                            "WHERE id=? AND upload_state='pending'", [(i,) for i in ids])
        self.db.commit()
        self.log(f'SKIP {len(ids)} {why}')

    def held(self):
        return self.db.execute("""SELECT a.id, a.created_at, l.observation_id AS memory_id
            FROM media_attachments a LEFT JOIN observation_media_links l ON l.attachment_id=a.id
            WHERE a.origin='native' AND a.state='ready' AND a.upload_state='pending' AND a.upload_retry_at=?
            ORDER BY a.created_at""", (HOLD,)).fetchall()

    def run(self):
        """Review held screenshots once. Returns counts of uploaded, skipped and deferred screenshots."""
        self.install_trigger()  # idempotent; re-adds the trigger if a claude-mem migration drops it
        now = self.now_ms()
        counts = {'uploaded': 0, 'skipped': 0, 'deferred': 0}
        groups = {}
        for row in self.held():
            if row['memory_id'] is None:
                if now - row['created_at'] >= WAIT_FOR_LINK_MS:
                    self.skip([row['id']], 'no linked memory after 30 min')
                    counts['skipped'] += 1
                else:
                    counts['deferred'] += 1
                continue
            groups.setdefault(row['memory_id'], []).append(row)
        for index, (memory_id, rows) in enumerate(groups.items()):
            ids = sorted({row['id'] for row in rows})
            if index >= MAX_MEMORIES_PER_RUN:
                counts['deferred'] += len(ids)
                continue
            memory = self.db.execute('SELECT type, project, title, subtitle, narrative, facts FROM observations WHERE id=?',
                                     (memory_id,)).fetchone()
            if memory is None:
                self.skip(ids, f'memory {memory_id} missing')
                counts['skipped'] += len(ids)
                continue
            try:
                result = self.decide({'text': memory_text(memory, len(ids))}, QUESTIONS)
                probability = float(result['answers']['worth_uploading']['noul'])
            except Exception as error:  # judgments.decide never retries; we retry on the next run
                if now - min(row['created_at'] for row in rows) >= GIVE_UP_MS:
                    self.skip(ids, f'JEV unavailable for 24h: {type(error).__name__}')
                    counts['skipped'] += len(ids)
                else:
                    self.log(f'JEV error for memory {memory_id}, will retry: {type(error).__name__}')
                    counts['deferred'] += len(ids)
                continue
            tag = f"memory={memory_id} p={probability:.2f} jev={result.get('id')} title={(memory['title'] or '')[:80]!r}"
            if probability >= self.upload_at:
                self.release(ids)
                self.log(f'UPLOAD {len(ids)} {tag}')
                counts['uploaded'] += len(ids)
            else:
                self.skip(ids, tag)
                counts['skipped'] += len(ids)
        return counts

    def uninstall(self):
        """Drop the trigger and release anything still held, so claude-mem uploads it as before."""
        self.db.execute(f'DROP TRIGGER IF EXISTS {TRIGGER_NAME}')
        self.db.execute("UPDATE media_attachments SET upload_retry_at=0 WHERE upload_state='pending' AND upload_retry_at=?", (HOLD,))
        self.db.commit()


def main(argv=None):
    parser = argparse.ArgumentParser(description='JEV review gate for claude-mem screenshot uploads')
    parser.add_argument('--db', type=Path, default=default_db())
    parser.add_argument('--upload-at', type=float, default=float(os.environ.get('SMART_AGENTS_MEDIA_UPLOAD_AT', DEFAULT_UPLOAD_AT)))
    parser.add_argument('--uninstall', action='store_true', help='drop the trigger and release held screenshots')
    args = parser.parse_args(argv)
    gate = Gate(sqlite3.connect(args.db, timeout=30), default_log(), upload_at=args.upload_at)
    if args.uninstall:
        gate.uninstall()
        print('JEV media gate removed; held screenshots released')
        return
    counts = gate.run()
    if any(counts.values()):
        gate.log(f"run: uploaded={counts['uploaded']} skipped={counts['skipped']} deferred={counts['deferred']}")


if __name__ == '__main__':
    main()
