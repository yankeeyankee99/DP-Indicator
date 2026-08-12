from __future__ import annotations
import asyncio
import sqlite3
import time
import hashlib
import json
from pathlib import Path
class BaseClient:
    def __init__(self, base_url: str, rate_limit: int = 5,
                 cache_dir: str = "data/cache"):
        self.base_url = base_url
        self.rate_limit = rate_limit
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._cache_dir / f"{self.__class__.__name__.lower()}.db"
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        # Enable WAL mode for concurrent async access
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY, value TEXT, created REAL
            )
        """)
        self._conn.commit()
        self._last_call = 0
        self._min_interval = 1.0 / rate_limit
        # Unified cache TTL: 24 hours (matches Orchestrator cache policy)
        self._cache_ttl_seconds = 24 * 3600
        # Lock to prevent TOCTOU race in _wait_rate
        self._rate_lock = asyncio.Lock()
    async def _wait_rate(self):
        async with self._rate_lock:
            elapsed = time.time() - self._last_call
            if elapsed < self._min_interval:
                await asyncio.sleep(self._min_interval - elapsed)
            self._last_call = time.time()
    def _cache_key(self, params: dict) -> str:
        raw = json.dumps(params, sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()
    def _get_cached(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM cache WHERE key=? AND created > ?",
            (key, time.time() - self._cache_ttl_seconds)
        ).fetchone()
        return row[0] if row else None
    def _set_cached(self, key: str, value: str):
        self._conn.execute(
            "INSERT OR REPLACE INTO cache (key, value, created) VALUES (?,?,?)",
            (key, value, time.time())
        )
        self._conn.commit()
    def close(self):
        self._conn.close()
    def __del__(self):
        # Ensure SQLite connection is closed on garbage collection
        # (covers exception paths where close() is not called)
        try:
            if hasattr(self, '_conn') and self._conn:
                self._conn.close()
        except Exception:
            pass
