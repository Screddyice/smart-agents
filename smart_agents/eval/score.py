"""Scoring for the retrieval eval set.

precision@k measures whether the right lessons surface. no-answer accuracy
measures whether the system stays quiet when nothing applies. Both matter:
a ranker that always returns three rows scores well on the first and destroys
the cheap-call property the invocation rule depends on.

The CLI drives a live `/advise` endpoint (see `corpus_api.main:advise`) over
each case in the eval JSONL and prints both numbers. The endpoint is not
reachable directly from this Mac -- it only answers on its own host's
loopback -- so `--ssh-host` shells out to `ssh <host> curl ...` instead of
making a direct HTTP call. See `.superpowers/sdd/2026-09-03-phase-1-retrieval/
task-2-brief.md` Ruling 3 for why.
"""
from __future__ import annotations

import argparse
import atexit
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

import httpx

DEFAULT_ADVISE_URL = "http://127.0.0.1:8210/advise"
DEFAULT_TIMEOUT = 10.0

# One ssh connection per case is what a naive loop does, and v2 has 131 of them.
# The box's sshd starts refusing at that rate -- the same MaxStartups wall the
# deploy script hit at six connections -- and a refusal mid-run aborts the whole
# score. Multiplexing puts every case through a single connection, which removes
# the refusals and most of the runtime with them.
_CONTROL_DIR: str | None = None


def _ssh_base() -> list[str]:
    """ssh argv prefix that reuses one shared connection for the whole run."""
    global _CONTROL_DIR
    if _CONTROL_DIR is None:
        # The control socket is a unix domain socket, so the whole path has to
        # fit in sockaddr_un: 104 bytes on macOS. The default temp directory
        # here is /var/folders/../T/, which spends 50 of them before the socket
        # name starts, and ssh then refuses with "ControlPath too long". Hence a
        # short directory and the shortest useful token ssh offers.
        _CONTROL_DIR = tempfile.mkdtemp(prefix="ce-", dir="/tmp")
        atexit.register(shutil.rmtree, _CONTROL_DIR, ignore_errors=True)
    return [
        "ssh",
        "-o", "ControlMaster=auto",
        "-o", f"ControlPath={_CONTROL_DIR}/%C",
        "-o", "ControlPersist=120",
        "-o", "BatchMode=yes",
    ]


# A refused connection is transient: sshd is shedding load, not rejecting the
# key. Losing a ten-minute score to one of them is the wrong trade, so back off
# and try again before giving up on the run.
SSH_ATTEMPTS = 3
SSH_BACKOFF_S = 2.0


def precision_at_k(returned: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """Fraction of the top k returned ids that are relevant.

    Divides by k rather than len(returned) so a system that returns one correct
    row scores lower than one returning three correct rows.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    relevant_set = set(relevant)
    hits = sum(1 for rid in list(returned)[:k] if rid in relevant_set)
    return hits / k


def no_answer_accuracy(cases: Sequence[dict]) -> float:
    """Fraction of no-answer cases that correctly returned nothing."""
    if not cases:
        return 0.0
    correct = sum(1 for c in cases if not c.get("returned"))
    return correct / len(cases)


def load_eval_cases(eval_path: Path) -> list[dict]:
    """Read the eval JSONL file into a list of case dicts."""
    cases = []
    with eval_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


def _extract_advice_ids(data: dict) -> list[int]:
    """Pull the ranked ids out of a parsed `/advise` response body.

    `corpus_api.main:advise` wraps its whole handler in a blanket
    `except Exception` and returns HTTP 200 with `{"ok": false, "advice": [],
    "error": "..."}` whenever something internal breaks (Ollama down, DB
    unreachable). An HTTP-status check or an ssh exit-code check only catches
    *transport* failures -- neither one sees this. Without this check, an
    application-level failure would come back as an empty advice list,
    indistinguishable from a correct abstention: `no_answer_accuracy` would
    read as improving while the endpoint was actually broken.

    So: if `ok` is present in the body and falsy, raise here, the same way
    the transport layer already raises on its own failures.
    """
    if not data.get("ok", True):
        raise RuntimeError(
            f"advise endpoint reported failure: {data.get('error', '<no error field>')}"
        )
    return [row["id"] for row in data.get("advice", [])]


def _query_advise_http(task: str, advise_url: str, limit: int, timeout: float) -> list[int]:
    """Call `/advise` directly over HTTP and return the returned ids, in rank order."""
    resp = httpx.post(advise_url, json={"task": task, "limit": limit}, timeout=timeout)
    resp.raise_for_status()
    return _extract_advice_ids(resp.json())


def _query_advise_ssh(
    task: str, advise_url: str, limit: int, timeout: float, ssh_host: str
) -> list[int]:
    """Call `/advise` by shelling out to `ssh <host> curl ...`.

    The advise host answers only on its own loopback (verified: a direct
    curl from this Mac to the box's public IP returns no route), so every
    request has to originate on the box itself.
    """
    payload = json.dumps({"task": task, "limit": limit})
    # shlex.quote, not hand-written single quotes: a task containing an
    # apostrophe ("the scaffold's manifest") ended the quoted string early and
    # the remote shell died with "unexpected EOF while looking for matching".
    # v1's tasks happened to contain none; v2's do.
    remote_cmd = (
        f"curl -s -m {int(timeout)} -X POST {shlex.quote(advise_url)} "
        f"-H 'Content-Type: application/json' -d {shlex.quote(payload)}"
    )
    last_error = ""
    for attempt in range(1, SSH_ATTEMPTS + 1):
        result = subprocess.run(
            [*_ssh_base(), ssh_host, remote_cmd],
            capture_output=True,
            text=True,
            timeout=timeout + 15,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return _extract_advice_ids(data)
        last_error = result.stderr.strip()
        if attempt < SSH_ATTEMPTS:
            time.sleep(SSH_BACKOFF_S * attempt)

    raise RuntimeError(
        f"ssh advise call failed after {SSH_ATTEMPTS} attempts "
        f"(exit {result.returncode}): {last_error}"
    )


def query_advise(
    task: str,
    advise_url: str = DEFAULT_ADVISE_URL,
    limit: int = 3,
    timeout: float = DEFAULT_TIMEOUT,
    ssh_host: str | None = None,
) -> list[int]:
    """Query the live advise endpoint for one task, returning ranked ids."""
    if ssh_host:
        return _query_advise_ssh(task, advise_url, limit, timeout, ssh_host)
    return _query_advise_http(task, advise_url, limit, timeout)


def evaluate_dataset(
    eval_path: Path,
    k: int,
    advise_url: str = DEFAULT_ADVISE_URL,
    ssh_host: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict:
    """Score every case in the eval file against a live /advise endpoint."""
    cases = load_eval_cases(eval_path)
    answerable = [c for c in cases if c["kind"] == "answerable"]
    no_answer_cases = [c for c in cases if c["kind"] == "no_answer"]

    precisions = []
    hits = 0
    for case in answerable:
        returned = query_advise(case["task"], advise_url, k, timeout, ssh_host)
        relevant = set(case["relevant_ids"])
        precisions.append(precision_at_k(returned, relevant, k))
        # hit@k: did at least one relevant row make the top k. precision@k on a
        # set where most cases label one id cannot separate "the answer is at
        # rank 1" from "the answer is at rank 4"; this can.
        hits += any(rid in relevant for rid in list(returned)[:k])

    no_answer_results = []
    for case in no_answer_cases:
        returned = query_advise(case["task"], advise_url, k, timeout, ssh_host)
        no_answer_results.append({"returned": returned})

    mean_precision = sum(precisions) / len(precisions) if precisions else 0.0
    return {
        "precision_at_k": mean_precision,
        "hit_at_k": hits / len(answerable) if answerable else 0.0,
        "no_answer_accuracy": no_answer_accuracy(no_answer_results),
        "k": k,
        "n_answerable": len(answerable),
        "n_no_answer": len(no_answer_cases),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score the retrieval eval set against a live /advise endpoint."
    )
    parser.add_argument("--eval", required=True, type=Path, help="Path to the eval JSONL file")
    parser.add_argument("--k", type=int, default=3, help="k for precision@k (default 3)")
    parser.add_argument(
        "--advise-url",
        default=DEFAULT_ADVISE_URL,
        help=(
            "URL of the live /advise endpoint as seen from where the request "
            f"actually runs (default {DEFAULT_ADVISE_URL})"
        ),
    )
    parser.add_argument(
        "--ssh-host",
        default=None,
        help=(
            "If set, issue each /advise call via `ssh <host> curl ...` instead of a "
            "direct HTTP call. Needed when /advise only answers on its own host's "
            "loopback -- e.g. --ssh-host hermes@127.0.0.1"
        ),
    )
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT, help="Per-request timeout in seconds"
    )
    args = parser.parse_args(argv)

    result = evaluate_dataset(args.eval, args.k, args.advise_url, args.ssh_host, args.timeout)
    print(
        f"precision@{args.k}: {result['precision_at_k']:.3f} "
        f"({result['n_answerable']} answerable cases)"
    )
    print(f"hit@{args.k}: {result['hit_at_k']:.3f}")
    print(
        f"no_answer_accuracy: {result['no_answer_accuracy']:.3f} "
        f"({result['n_no_answer']} no-answer cases)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
