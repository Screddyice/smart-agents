"""RDS writer for Pipeline A.

Inserts classifier-produced learnings into the public.learnings table on
AWS RDS. Idempotent via the unique constraint
(instance, agent_id, source_entry_hash, category).
"""
from __future__ import annotations

import logging
import hashlib
from typing import Optional

import psycopg
from learning_classifier.models import Learning

from .config import Config
from .embedding import EmbeddingBackend
from .hyperswarm_source import SessionEntry

# Width of this table's pgvector column. `learnings` was migrated to vector(768) with the move to nomic-embed-text.
EMBED_DIM = 768

log = logging.getLogger(__name__)


def _vector_literal(vec: Optional[list[float]]) -> Optional[str]:
    """pgvector accepts the literal '[a,b,c,...]' format."""
    if vec is None:
        return None
    return "[" + ",".join(f"{x:.6g}" for x in vec) + "]"


class RDSWriter:
    """Writer for Pipeline A.

    One long-lived psycopg connection. Reconnects on transient failures.
    """

    def __init__(self, cfg: Config, embedder: EmbeddingBackend):
        assert cfg.rds_url, "RDSWriter requires a populated rds_url"
        self.cfg = cfg
        self.embedder = embedder
        self._conn: Optional[psycopg.Connection] = None

    def _get_conn(self) -> psycopg.Connection:
        if self._conn is None or self._conn.closed:
            # prepare_threshold=None: rds_url points at Supabase's
            # transaction-mode pooler, where a server-side prepared statement
            # made on one pooled backend is invisible to the next
            # (InvalidSqlStatementName '_pg3_N'). psycopg3 auto-prepares any
            # statement executed >5 times, so repeated INSERTs on this
            # long-lived connection start failing mid-batch without this.
            self._conn = psycopg.connect(
                self.cfg.rds_url, autocommit=True, prepare_threshold=None
            )
        return self._conn

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()

    def write(self, session: SessionEntry, learnings: list[Learning]) -> int:
        """Write one session's learnings. Returns count of new rows inserted."""
        if not learnings:
            return 0

        rows = []
        for learning in learnings:
            embed_text = f"{learning.what_happened}\n{learning.lesson}"
            embedding = self.embedder.embed(embed_text, target_dim=EMBED_DIM)
            if embedding is None:
                raise RuntimeError("Embedding unavailable; source must remain retryable")
            rows.append((
                self.cfg.org,
                self.cfg.instance,
                session.agent_id,
                session.project,
                session.occurred_at,
                learning.category,
                getattr(learning, "source", "shawn"),
                learning.what_happened,
                learning.shawn_feedback,
                learning.lesson,
                learning.tags or None,
                _vector_literal(embedding),
                session.session_id,
                hashlib.sha256((session.body_hash + "\0" + learning.category + "\0" + learning.what_happened.strip() + "\0" + learning.lesson.strip()).encode()).hexdigest(),
            ))

        sql = """
            insert into learnings (
                org, instance, agent_id, project, occurred_at,
                category, source, what_happened, shawn_feedback, lesson, tags, embedding,
                source_session_id, source_entry_hash
            )
            values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            on conflict (instance, agent_id, source_entry_hash, category) do nothing
            returning id
        """

        inserted = 0
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                for row in rows:
                    cur.execute(sql, row)
                    if cur.fetchone():
                        inserted += 1
        except psycopg.OperationalError as e:
            # Transient — drop conn and let next call reconnect.
            log.warning("RDS write transient error, closing conn: %s", e)
            self.close()
            raise

        return inserted
