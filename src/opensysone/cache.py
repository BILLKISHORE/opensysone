"""Decision cache: same state, questions, options and model versions, same answer."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections import OrderedDict
from typing import Any


def cache_key(*parts: Any) -> str:
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class DecisionCache:
    def __init__(self, max_entries: int = 1000, path: str | None = None):
        self.max_entries = max_entries
        self._lru: OrderedDict[str, str] = OrderedDict()
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False) if path else None
        if self._db is not None:
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS decisions "
                "(key TEXT PRIMARY KEY, value TEXT NOT NULL, ts REAL NOT NULL)"
            )
            self._db.commit()

    def _evict(self) -> None:
        while len(self._lru) > self.max_entries:
            self._lru.popitem(last=False)

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            if key in self._lru:
                self._lru.move_to_end(key)
                return json.loads(self._lru[key])
            if self._db is not None:
                row = self._db.execute("SELECT value FROM decisions WHERE key = ?", (key,)).fetchone()
                if row is not None:
                    self._lru[key] = row[0]
                    self._evict()
                    return json.loads(row[0])
        return None

    def put(self, key: str, value: dict[str, Any]) -> None:
        blob = json.dumps(value, ensure_ascii=False)
        with self._lock:
            self._lru[key] = blob
            self._lru.move_to_end(key)
            self._evict()
            if self._db is not None:
                self._db.execute(
                    "INSERT OR REPLACE INTO decisions (key, value, ts) VALUES (?, ?, ?)",
                    (key, blob, time.time()),
                )
                self._db.commit()

    def clear(self) -> None:
        with self._lock:
            self._lru.clear()
            if self._db is not None:
                self._db.execute("DELETE FROM decisions")
                self._db.commit()

    def __len__(self) -> int:
        return len(self._lru)
