"""
engine/infrastructure/redis_adapter.py
--------------------------------------
Lightweight async Redis publisher for external event distribution.
Failures gracefully degrade and do not crash the engine.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

import redis.asyncio as redis

logger = logging.getLogger(__name__)


class EventEncoder(json.JSONEncoder):
    """JSON Encoder for complex types in trading events."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


class RedisAdapter:
    """
    Adapter to publish internal trading events to an external Redis pub/sub.
    """

    def __init__(self, url: str) -> None:
        self.url = url
        self._redis: redis.Redis | None = None

    async def connect(self) -> None:
        """Connect to Redis and ping to verify."""
        try:
            self._redis = redis.from_url(self.url, decode_responses=True, socket_timeout=2.0)
            await self._redis.ping()
            logger.info("RedisAdapter: Connected to %s", self.url)
        except Exception as e:
            logger.error("RedisAdapter: Connection failed: %s. Operating without Redis.", e)
            self._redis = None

    async def disconnect(self) -> None:
        """Close the Redis connection cleanly."""
        if self._redis:
            try:
                await self._redis.aclose()
            except Exception:
                pass
            logger.info("RedisAdapter: Disconnected")
            self._redis = None

    def publish_nowait(self, channel: str, event: Any) -> None:
        """
        Fire-and-forget publish. Does not block the caller.
        """
        if not self._redis:
            return

        try:
            if dataclasses.is_dataclass(event):
                payload = dataclasses.asdict(event)
            elif isinstance(event, dict):
                payload = event
            else:
                payload = {"data": str(event)}

            data_str = json.dumps(payload, cls=EventEncoder)

            task = asyncio.create_task(self._publish_task(channel, data_str))
            task.add_done_callback(self._handle_task_result)
        except Exception as e:
            logger.warning("RedisAdapter: Failed to serialize event for %s: %s", channel, e)

    async def _publish_task(self, channel: str, data: str) -> None:
        if self._redis:
            # We don't want a single slow publish to stall, so we wrap in wait_for
            await asyncio.wait_for(self._redis.publish(channel, data), timeout=1.0)

    def _handle_task_result(self, task: asyncio.Task) -> None:
        try:
            task.result()
        except asyncio.TimeoutError:
            logger.warning("RedisAdapter: Publish timed out")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning("RedisAdapter: Background publish failed: %s", e)
