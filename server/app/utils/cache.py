"""
Result cache. Uses Redis when available, falls back to a process-local dict.
Async API so call sites don't change between modes.
"""
import json
import logging
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)


class _MemoryCache:
    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    async def get(self, key: str) -> Optional[str]:
        return self._store.get(key)

    async def set(self, key: str, value: str, ttl: int) -> None:
        # ttl ignored in memory mode (process is short-lived anyway)
        self._store[key] = value


class _RedisCache:
    def __init__(self, url: str) -> None:
        import redis.asyncio as aioredis
        self._r = aioredis.from_url(url, decode_responses=True)

    async def get(self, key: str) -> Optional[str]:
        try:
            return await self._r.get(key)
        except Exception as e:
            logger.warning("Redis get failed, ignoring: %s", e)
            return None

    async def set(self, key: str, value: str, ttl: int) -> None:
        try:
            await self._r.set(key, value, ex=ttl)
        except Exception as e:
            logger.warning("Redis set failed, ignoring: %s", e)


def build_cache():
    try:
        return _RedisCache(settings.redis_url)
    except Exception as e:
        logger.warning("Redis unavailable (%s), using in-memory cache", e)
        return _MemoryCache()


class ResultCache:
    def __init__(self) -> None:
        self._impl = build_cache()

    async def get_json(self, key: str) -> Optional[dict]:
        raw = await self._impl.get(key)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    async def set_json(self, key: str, value: dict) -> None:
        await self._impl.set(key, json.dumps(value), settings.cache_ttl_seconds)
