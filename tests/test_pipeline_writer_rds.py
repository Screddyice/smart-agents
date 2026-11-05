"""Unit tests for RDSWriter connection setup.

Covers the 2026-07 catch-up outage: rds_url points at Supabase's
transaction-mode pooler, where psycopg3's auto-prepared statements
(prepare_threshold default 5) fail with InvalidSqlStatementName once the
pooler hands the session a different backend. 24 of 72 classified entries
died at the write step before prepare_threshold=None was forced.
"""
from unittest import mock

import pytest

import learning_corpus_pipeline.writer_rds as wr
from learning_corpus_pipeline.writer_rds import RDSWriter


class _Cfg:
    rds_url = "postgresql://user:pw@pooler.supabase.example:6543/postgres"
    org = "personal"
    instance = "test"


class _Embedder:
    def embed(self, text):
        return [0.0, 1.0]


def _writer():
    return RDSWriter(_Cfg(), _Embedder())


def test_get_conn_disables_prepared_statements(monkeypatch):
    """The pooler cannot serve auto-prepared statements — must be off."""
    connect = mock.Mock(return_value=mock.Mock(closed=False))
    monkeypatch.setattr(wr.psycopg, "connect", connect)

    writer = _writer()
    conn = writer._get_conn()

    connect.assert_called_once_with(
        _Cfg.rds_url, autocommit=True, prepare_threshold=None
    )
    assert conn is connect.return_value


def test_get_conn_reuses_open_connection(monkeypatch):
    connect = mock.Mock(return_value=mock.Mock(closed=False))
    monkeypatch.setattr(wr.psycopg, "connect", connect)

    writer = _writer()
    first = writer._get_conn()
    second = writer._get_conn()

    assert first is second
    assert connect.call_count == 1


def test_get_conn_reconnects_after_close(monkeypatch):
    closed_conn = mock.Mock(closed=True)
    fresh_conn = mock.Mock(closed=False)
    connect = mock.Mock(side_effect=[closed_conn, fresh_conn])
    monkeypatch.setattr(wr.psycopg, "connect", connect)

    writer = _writer()
    writer._get_conn()          # returns closed_conn
    conn = writer._get_conn()   # closed → reconnect

    assert conn is fresh_conn
    assert connect.call_count == 2
    for call in connect.call_args_list:
        assert call.kwargs.get("prepare_threshold", "MISSING") is None
