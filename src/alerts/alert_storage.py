"""
alerts/alert_storage.py
SQLite persistence layer for IDS alerts.

Schema
------
    alerts (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp           TEXT NOT NULL,
        src_ip              TEXT NOT NULL,
        dst_ip              TEXT NOT NULL,
        src_port            INTEGER NOT NULL,
        dst_port            INTEGER NOT NULL,
        protocol            INTEGER NOT NULL,
        attack_type         TEXT NOT NULL,
        attack_probability  REAL NOT NULL,
        rf_probability      REAL NOT NULL DEFAULT 0.0,
        xgb_probability     REAL NOT NULL DEFAULT 0.0,
        meta_probability    REAL NOT NULL DEFAULT 0.0,
        attack_confidence   REAL NOT NULL,
        iso_score           REAL NOT NULL,
        severity            TEXT NOT NULL,
        latency_ms          REAL NOT NULL
    )

Grafana connects via the SQLite plugin (frser-sqlite-datasource).
Future migration to PostgreSQL requires only changing the connection string.
"""

import logging
import os
import sqlite3
from contextlib import contextmanager
from typing import List, Optional

from alerts.alert import Alert

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS alerts (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp           TEXT    NOT NULL,
    src_ip              TEXT    NOT NULL,
    dst_ip              TEXT    NOT NULL,
    src_port            INTEGER NOT NULL,
    dst_port            INTEGER NOT NULL,
    protocol            INTEGER NOT NULL,
    attack_type         TEXT    NOT NULL,
    attack_probability  REAL    NOT NULL,
    rf_probability      REAL    NOT NULL DEFAULT 0.0,
    xgb_probability     REAL    NOT NULL DEFAULT 0.0,
    meta_probability    REAL    NOT NULL DEFAULT 0.0,
    attack_confidence   REAL    NOT NULL,
    iso_score           REAL    NOT NULL,
    severity            TEXT    NOT NULL,
    latency_ms          REAL    NOT NULL
);
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_timestamp   ON alerts (timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_severity     ON alerts (severity);",
    "CREATE INDEX IF NOT EXISTS idx_attack_type  ON alerts (attack_type);",
    "CREATE INDEX IF NOT EXISTS idx_src_ip       ON alerts (src_ip);",
    "CREATE INDEX IF NOT EXISTS idx_dst_ip       ON alerts (dst_ip);",
]

# Migration queries for databases created before the stacking ensemble refactor.
# ALTER TABLE ADD COLUMN is safe — it preserves all existing rows.
_MIGRATION_COLUMNS = [
    ("rf_probability",   "ALTER TABLE alerts ADD COLUMN rf_probability   REAL NOT NULL DEFAULT 0.0;"),
    ("xgb_probability",  "ALTER TABLE alerts ADD COLUMN xgb_probability  REAL NOT NULL DEFAULT 0.0;"),
    ("meta_probability", "ALTER TABLE alerts ADD COLUMN meta_probability REAL NOT NULL DEFAULT 0.0;"),
]

_INSERT = """
INSERT INTO alerts (
    timestamp, src_ip, dst_ip, src_port, dst_port, protocol,
    attack_type, attack_probability, rf_probability, xgb_probability,
    meta_probability, attack_confidence, iso_score, severity, latency_ms
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


class AlertStorage:
    """
    Thread-safe SQLite alert store.

    Parameters
    ----------
    db_path : str
        Path to the SQLite file. Created automatically if it does not exist.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._init_db()
        logger.info("AlertStorage ready: %s", db_path)

    def insert(self, alert: Alert) -> int:
        with self._conn() as conn:
            cur = conn.execute(_INSERT, (
                alert.timestamp,
                alert.src_ip,    alert.dst_ip,
                alert.src_port,  alert.dst_port,
                alert.protocol,  alert.attack_type,
                alert.attack_probability,
                alert.rf_probability,
                alert.xgb_probability,
                alert.meta_probability,
                alert.attack_confidence,
                alert.iso_score, alert.severity, alert.latency_ms,
            ))
            row_id = cur.lastrowid
            alert.id = row_id
            logger.debug("Alert stored | id=%d | type=%s | severity=%s",
                         row_id, alert.attack_type, alert.severity)
            return row_id

    def recent(self, limit: int = 100) -> List[Alert]:
        """Return the most recent `limit` alerts, newest first."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_row_to_alert(r) for r in rows]

    def count_by_type(self) -> dict:
        """Return {attack_type: count} for all stored alerts."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT attack_type, COUNT(*) FROM alerts GROUP BY attack_type"
            ).fetchall()
        return {r[0]: r[1] for r in rows}

    def count_by_severity(self) -> dict:
        """Return {severity: count} for all stored alerts."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT severity, COUNT(*) FROM alerts GROUP BY severity"
            ).fetchall()
        return {r[0]: r[1] for r in rows}

    def total(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(_CREATE_TABLE)
            for idx in _CREATE_INDEXES:
                conn.execute(idx)
            # Run migrations for existing databases
            self._migrate(conn)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Add new columns to existing databases without data loss."""
        existing_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(alerts)").fetchall()
        }
        for col_name, alter_sql in _MIGRATION_COLUMNS:
            if col_name not in existing_columns:
                try:
                    conn.execute(alter_sql)
                    logger.info("Migration: added column '%s' to alerts table", col_name)
                except sqlite3.OperationalError as exc:
                    # Column may already exist in a race condition; safe to ignore
                    logger.debug("Migration skipped for '%s': %s", col_name, exc)

    @contextmanager
    def _conn(self):
        """Yield a thread-safe connection with WAL mode enabled."""
        if self._db_path == ":memory:":
            if not hasattr(self, "_mem_conn") or self._mem_conn is None:
                self._mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
                self._mem_conn.row_factory = sqlite3.Row
            conn = self._mem_conn
            try:
                yield conn
                conn.commit()
            except sqlite3.Error:
                conn.rollback()
                raise
            return

        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")   # safe for concurrent reads
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except sqlite3.Error:
            conn.rollback()
            raise
        finally:
            conn.close()



def _row_to_alert(row: sqlite3.Row) -> Alert:
    return Alert(
        id                  = row["id"],
        timestamp           = row["timestamp"],
        src_ip              = row["src_ip"],
        dst_ip              = row["dst_ip"],
        src_port            = row["src_port"],
        dst_port            = row["dst_port"],
        protocol            = row["protocol"],
        attack_type         = row["attack_type"],
        attack_probability  = row["attack_probability"],
        rf_probability      = row["rf_probability"] if "rf_probability" in row.keys() else 0.0,
        xgb_probability     = row["xgb_probability"] if "xgb_probability" in row.keys() else 0.0,
        meta_probability    = row["meta_probability"] if "meta_probability" in row.keys() else 0.0,
        attack_confidence   = row["attack_confidence"],
        iso_score           = row["iso_score"],
        severity            = row["severity"],
        latency_ms          = row["latency_ms"],
    )
