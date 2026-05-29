"""SQLite-backed deduplication store.

Tracks IDs we've already processed so restarts don't repost digests or
re-forward channel messages. A single table with a namespace column keeps RSS
entries and Telegram messages cleanly separated.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path


class SeenStore:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False because APScheduler jobs and the Telethon
        # handler may touch the DB from different threads; guarded by a lock.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS seen (
                    namespace TEXT NOT NULL,
                    item_id   TEXT NOT NULL,
                    seen_at   REAL NOT NULL,
                    PRIMARY KEY (namespace, item_id)
                )
                """
            )
            self._conn.commit()

    def is_seen(self, namespace: str, item_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "SELECT 1 FROM seen WHERE namespace = ? AND item_id = ?",
                (namespace, str(item_id)),
            )
            return cur.fetchone() is not None

    def mark_seen(self, namespace: str, item_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO seen (namespace, item_id, seen_at) VALUES (?, ?, ?)",
                (namespace, str(item_id), time.time()),
            )
            self._conn.commit()

    def mark_seen_many(self, namespace: str, item_ids: list[str]) -> None:
        if not item_ids:
            return
        now = time.time()
        with self._lock:
            self._conn.executemany(
                "INSERT OR IGNORE INTO seen (namespace, item_id, seen_at) VALUES (?, ?, ?)",
                [(namespace, str(i), now) for i in item_ids],
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
