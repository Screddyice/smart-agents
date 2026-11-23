# Screenshot upload gate

claude-mem uploads every screenshot an agent captures to its cloud. Most are routine: page loads, login screens, confirmation clicks, terminal output that the memory already records as text. `smart-agents-media-gate` asks JEV which screenshots are worth uploading and keeps the rest on the local machine.

## How it works

1. A SQLite trigger on claude-mem's `media_attachments` table holds each new native screenshot by setting `upload_retry_at` to a far-future sentinel. claude-mem's uploader only sends rows whose retry time has passed, so held rows wait.
2. Each run, the gate groups held screenshots by the memory they belong to and sends JEV one typed yes/no question per memory, built from the memory's type, project, title, subtitle, narrative and facts (bounded to about 3 KB). JEV sees text only, never the image.
3. If JEV's probability is at least the threshold (default 0.50), the screenshots are released (`upload_retry_at = 0`) and claude-mem uploads them. Otherwise they are marked `upload_state='failed'`, `upload_error='jev_skipped'`, the same terminal state claude-mem uses for uploads that can never succeed. The image files stay local.

The prompt leans toward skipping. In a check against real memories, routine screenshots scored 0.10 to 0.14 and a described visual layout bug scored 0.70.

Edge cases:

- A screenshot not linked to a memory within 30 minutes is skipped.
- JEV calls are never retried within a run (`smart_agents.judgments`). A failed call is retried on the next run; after 24 hours the screenshot is skipped instead of held forever.
- At most 25 memories are reviewed per run; the rest wait for the next run.
- Replica screenshots pulled from other devices are never held.
- Each run re-creates the trigger if a claude-mem migration has dropped it.

Every decision is appended to the gate log with the JEV request ID, probability and memory title.

## Install

Nothing runs on package installation. Copy `deploy/launchd/ai.smart-agents.media-gate.plist` to `~/Library/LaunchAgents/`, replace the placeholder paths, then load it with `launchctl bootstrap gui/$(id -u) <plist>`. The first run installs the trigger.

The trigger and the schedule belong together. Without the schedule, held screenshots are never reviewed.

## Remove

```sh
launchctl bootout gui/$(id -u)/ai.smart-agents.media-gate
smart-agents-media-gate --uninstall
```

`--uninstall` drops the trigger and releases anything still held, so claude-mem uploads it as before. Screenshots already skipped stay skipped.

## Configuration

| Variable | Purpose |
|---|---|
| `SMART_AGENTS_CMEM_DB` | claude-mem database, default `~/.claude-mem/claude-mem.db` |
| `SMART_AGENTS_MEDIA_GATE_LOG` | Decision log, default `~/.claude-mem/logs/jev-media-gate.log` |
| `SMART_AGENTS_MEDIA_UPLOAD_AT` | Minimum JEV probability to upload, default `0.50` |
| `SMART_AGENTS_JEV_BIN` | JEV stdio adapter, shared with the classifier |

Each JEV call cost about $0.00001 in testing.
