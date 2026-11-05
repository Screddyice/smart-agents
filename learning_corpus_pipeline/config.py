"""Environment-driven config for the pipeline daemons.

All hosts read from `~/.learning-corpus/rds.env` (Phase 1 distributed this).
There is one corpus store: the `learnings` table behind LEARNING_CORPUS_RDS_URL.
The private `screddy_learnings` lane (Pipeline B, HOST_ROLE=neb) was retired on
2026-09-04; every agent's rows land in the same table.

The HOST_ROLE env var (or auto-detected from hostname) determines which
agents this host hosts and which writers are allowed.
"""
from __future__ import annotations

import os
import socket
import pathlib
from dataclasses import dataclass
from typing import Literal


HostRole = Literal["mac", "neb", "cliqk", "trc"]


def _detect_host_role() -> HostRole:
    """Map the current hostname to a host_role tag.

    The pipeline daemons run on 4 named hosts. Anything else means we're
    in a dev or test environment — fail loud rather than guess.
    """
    explicit = os.environ.get("HOST_ROLE")
    if explicit:
        return explicit  # type: ignore[return-value]

    hostname = socket.gethostname()
    if hostname.endswith(".local") or "mac" in hostname.lower() or hostname == "shawn-mac":
        return "mac"
    # screddy-consult (the SRC box) took over the brain-server role on 2026-08-20 when the
    # corpus stack moved off the TMN Hostinger box. The ROLE keeps its legacy "neb" label
    # (and therefore org=NEB on writes) on purpose: the org-column taxonomy is collapsed
    # NEB->TMN by the 2026-08 corpus migration, and
    # renaming the writer's vocabulary before that lands would fork the taxonomy mid-stream.
    # The VPS hostname became "src" on 2026-08-27; "screddy-consult" is kept so
    # a host that has not been renamed yet still resolves. Matching on the exact
    # string rather than a substring is deliberate: "src" appears inside plenty
    # of unrelated hostnames, and a false match would route another box's
    # sessions into this one's org.
    if hostname in ("src", "screddy-consult"):
        return "neb"
    if "neb" in hostname.lower() or "ip-172-31-11" in hostname:
        return "neb"
    if "cliqk" in hostname.lower() or "ip-172-31-13" in hostname:
        return "cliqk"
    if "trc" in hostname.lower() or "ip-172-31-6" in hostname:
        return "trc"
    return "mac"


# Org per host. Used for the entries.org column.
ORG_BY_HOST: dict[HostRole, str] = {
    "mac":   "cross",   # Mac sees work across orgs
    "neb":   "NEB",
    "cliqk": "Cliqk",
    "trc":   "TRC",
}

# Instance label per host. Used for the entries.instance column.
INSTANCE_BY_HOST: dict[HostRole, str] = {
    "mac":   "shawn-mac",
    "neb":   "neb-server",
    "cliqk": "cliqk-server",
    "trc":   "trc-server",
}



@dataclass(frozen=True)
class Config:
    """Resolved daemon configuration."""

    host_role: HostRole
    org: str
    instance: str

    # Where HyperSwarm entries land — same on every host.
    hyperswarm_dir: pathlib.Path

    # State / cursor dir.
    state_dir: pathlib.Path

    # Corpus connection string (populated when need_rds=True).
    rds_url: str | None

    # Ollama for embedding.
    ollama_url: str
    embedding_model: str


    @classmethod
    def from_env(cls, *, need_rds: bool = False) -> "Config":
        host_role = _detect_host_role()
        home = pathlib.Path.home()
        state_dir = pathlib.Path(os.environ.get("SMART_AGENTS_STATE_DIR", home / ".local/state/smart-agents"))
        state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

        rds_url = None
        if need_rds:
            # Sourced from ~/.learning-corpus/rds.env in the systemd/launchd unit.
            rds_url = os.environ.get("LEARNING_CORPUS_RDS_URL") or os.environ.get("RDS_URL")
            if not rds_url:
                raise RuntimeError(
                    "LEARNING_CORPUS_RDS_URL not set. "
                    "Ensure ~/.learning-corpus/rds.env is sourced into the daemon's env."
                )

        return cls(
            host_role=host_role,
            org=os.environ.get("SMART_AGENTS_ORG", ORG_BY_HOST.get(host_role, "cross")),
            instance=os.environ.get("SMART_AGENTS_INSTANCE", socket.gethostname()),
            hyperswarm_dir=home / "HyperSwarm" / "entries",
            state_dir=state_dir,
            rds_url=rds_url,
            ollama_url=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
            embedding_model=os.environ.get("LEARNING_CORPUS_EMBED_MODEL", "nomic-embed-text"),
        )
