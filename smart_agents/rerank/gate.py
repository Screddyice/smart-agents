"""Publish gate: a candidate ranker replaces the champion only by beating it.

Three numbers on the frozen pools, at the candidate's own abstention
threshold: precision@3, hit@3, no-answer accuracy. The candidate must be at
least the champion on all three and better on at least one. The old trainer
promoted whatever finished; this is the fix the spec asks for.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path

METRICS = ("precision_at_k", "hit_at_k", "no_answer_accuracy")
CHAMPION_DIR = Path(os.environ.get("SMART_AGENTS_STATE_DIR", Path.home() / ".local/state/smart-agents")) / "rerank" / "champion"


@dataclass
class Verdict:
    publish: bool
    reason: str
    candidate: dict
    champion: dict


def compare(candidate: dict, champion: dict) -> Verdict:
    if any(m not in row or not isinstance(row[m], (int,float)) or not math.isfinite(row[m]) or not 0 <= row[m] <= 1 for row in (candidate, champion) for m in METRICS):
        return Verdict(False, "invalid or incomplete metrics", candidate, champion)
    worse = [m for m in METRICS if candidate[m] < champion[m] - 1e-9]
    if worse:
        return Verdict(False, f"candidate loses on {', '.join(worse)}", candidate, champion)
    better = [m for m in METRICS if candidate[m] > champion[m] + 1e-9]
    if not better:
        return Verdict(False, "candidate ties the champion on every number", candidate, champion)
    return Verdict(True, f"candidate wins on {', '.join(better)}", candidate, champion)


def decide(candidate: dict[str, dict], champion: dict[str, dict], *, primary: str) -> Verdict:
    """Two-set rule: win on the primary set, lose nowhere on the others.

    v1 is saturated, so a candidate publishes by beating the champion on v2
    (at least as good on all three numbers, better on one) while staying at
    least as good on every other set. With only one set this is `compare`.
    """
    if primary not in candidate or primary not in champion or not set(champion).issubset(candidate):
        return Verdict(False, "missing evaluation sets", candidate, champion)
    verdict = compare(candidate[primary], champion[primary])
    if not verdict.publish:
        return Verdict(False, f"{primary}: {verdict.reason}", candidate, champion)
    for name, champ in champion.items():
        if name == primary or name not in candidate:
            continue
        worse = [m for m in METRICS if candidate[name][m] < champ[m] - 1e-9]
        if worse:
            return Verdict(False, f"wins on {primary} but loses on {name}: {', '.join(worse)}", candidate, champion)
    return Verdict(True, f"{primary}: {verdict.reason}; no regression elsewhere", candidate, champion)


def score_run(run_dir: Path) -> dict[str, tuple[float, dict]]:
    # Imported here so `compare` and `decide` stay usable without torch installed.
    from smart_agents.rerank.cross_encoder import CrossEncoderRanker
    from smart_agents.rerank.sweep_threshold import sweep_sets

    return sweep_sets(CrossEncoderRanker(run_dir / "model"))


def promote(run_dir: Path, *, chosen: dict[str, tuple[float, dict]], primary: str) -> Path:
    from smart_agents.rerank.sweep_threshold import CHAMPION, sha256_of_dir

    # Never delete previous champions. Publish a new immutable version.
    from uuid import uuid4
    version = CHAMPION_DIR.parent / ("release-" + uuid4().hex)
    version.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(run_dir / "model", version, symlinks=False)
    min_score, scores = chosen[primary]
    record = version / "release.json"
    record.write_text(json.dumps({
        "model": str(version), "sha256": sha256_of_dir(version), "min_score": min_score,
        "scores": {m: scores[m] for m in METRICS},
        "sets": {name: {"min_score": t, **{m: r[m] for m in METRICS}} for name, (t, r) in chosen.items()},
        "primary": primary, "recorded_at": date.today().isoformat(), "source_run": str(run_dir),
    }, indent=2) + "\n")
    return version


def main(argv: list[str] | None = None) -> int:
    from smart_agents.rerank.sweep_threshold import CHAMPION

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--promote", action="store_true")
    args = p.parse_args(argv)

    champion = json.loads(CHAMPION.read_text())
    chosen = score_run(args.run)
    primary = "v2" if "v2" in chosen else "v1"
    champ_sets = champion.get("sets") or {"v1": {**champion["scores"], "min_score": champion["min_score"]}}
    if primary not in champ_sets:
        # The champion predates v2: treat its v1 numbers as the bar on every set it lacks.
        champ_sets = {name: champ_sets.get(name, champ_sets["v1"]) for name in chosen}
    cand_sets = {name: r for name, (_, r) in chosen.items()}
    verdict = decide(cand_sets, {k: v for k, v in champ_sets.items() if k in cand_sets}, primary=primary)
    for name, (t, r) in chosen.items():
        print(f"candidate {name} at min_score={t}: " + ", ".join(f"{m}={r[m]:.3f}" for m in METRICS))
        c = champ_sets[name]
        print(f"champion  {name}: " + ", ".join(f"{m}={c[m]:.3f}" for m in METRICS))
    print(f"verdict: {'PUBLISH' if verdict.publish else 'KEEP CHAMPION'} ({verdict.reason})")
    if verdict.publish and args.promote:
        print(f"promoted to {promote(args.run, chosen=chosen, primary=primary)}")
    return 0 if verdict.publish else 1


if __name__ == "__main__":
    raise SystemExit(main())
