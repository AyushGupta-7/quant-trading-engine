"""
engine/core/event_bus.py
------------------------
Typed publish/subscribe bus backed by asyncio.Queue.

Design decisions
~~~~~~~~~~~~~~~~
* One queue per *topic* (string tag) to avoid head-of-line blocking between
  unrelated consumers (e.g. tick consumers vs fill consumers).
* Back-pressure: each queue has a configurable ``maxsize``; producers use
  ``put()`` which will block when the queue is full rather than silently
  dropping events.
* Graceful shutdown: ``shutdown()`` cancels all pending consumer tasks and
  drains queues.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)

# Type alias for async event handler coroutines
Handler = Callable[[Any], Coroutine[Any, Any, None]]


class EventBus:
    """
    Lightweight asyncio-based publish/subscribe event bus.

    Usage::

        bus = EventBus()

        async def on_tick(tick):
            print(tick)

        bus.subscribe("tick", on_tick)
        await bus.publish("tick", some_tick)
    """

    def __init__(self, default_queue_size: int = 1000) -> None:
        self._default_queue_size = default_queue_size
        # topic → list of handler coroutine functions
        self._handlers: dict[str, list[Handler]] = defaultdict(list)
        # topic → asyncio.Queue
        self._queues: dict[str, asyncio.Queue] = {}
        # topic → consumer Task
        self._consumer_tasks: dict[str, asyncio.Task] = {}
        self._shutdown_event = asyncio.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def subscribe(self, topic: str, handler: Handler, queue_size: int | None = None) -> None:
        """Register *handler* for *topic*.  Multiple handlers per topic allowed."""
        self._handlers[topic].append(handler)
        if topic not in self._queues:
            size = queue_size if queue_size is not None else self._default_queue_size
            self._queues[topic] = asyncio.Queue(maxsize=size)

    async def publish(self, topic: str, event: Any) -> None:
        """
        Publish *event* to *topic*.

        If no queue exists for the topic (no subscribers), the event is
        silently dropped.  If the queue is full, this coroutine blocks until
        space is available (back-pressure).
        """
        if topic not in self._queues:
            return
        await self._queues[topic].put(event)

    def publish_nowait(self, topic: str, event: Any) -> bool:
        """
        Non-blocking publish; returns False (and drops event) if queue full.
        Prefer ``publish()`` for critical events.
        """
        if topic not in self._queues:
            return False
        try:
            self._queues[topic].put_nowait(event)
            return True
        except asyncio.QueueFull:
            logger.warning("EventBus: queue full, dropping event on topic=%s", topic)
            return False

    def start(self) -> None:
        """
        Spin up one consumer task per topic.
        Call after all subscriptions are registered and the event loop is running.
        """
        for topic in list(self._queues.keys()):
            if topic not in self._consumer_tasks:
                task = asyncio.create_task(
                    self._consume(topic), name=f"eventbus-consumer-{topic}"
                )
                self._consumer_tasks[topic] = task
                logger.debug("EventBus: started consumer for topic=%s", topic)

    async def shutdown(self) -> None:
        """
        Signal shutdown and wait for all consumer tasks to finish draining.
        """
        self._shutdown_event.set()
        # Unblock all queues with a sentinel
        for q in self._queues.values():
            await q.put(None)  # sentinel
        if self._consumer_tasks:
            await asyncio.gather(
                *self._consumer_tasks.values(), return_exceptions=True
            )
        self._consumer_tasks.clear()
        logger.info("EventBus: shutdown complete.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _consume(self, topic: str) -> None:
        """Drain the queue for *topic* and dispatch to all registered handlers."""
        q = self._queues[topic]
        handlers = self._handlers[topic]
        while True:
            event = await q.get()
            if event is None:  # shutdown sentinel
                q.task_done()
                break
            for handler in handlers:
                try:
                    await handler(event)
                except Exception:
                    logger.exception(
                        "EventBus: unhandled error in handler %s for topic=%s",
                        handler.__qualname__, topic,
                    )
            q.task_done()


# ---------------------------------------------------------------------------
# Topic name constants (avoids magic strings across codebase)
# ---------------------------------------------------------------------------

class Topics:
    TICK = "tick"
    BAR = "bar"
    ORDER_INTENT = "order_intent"
    ORDER = "order"
    FILL = "fill"
    KILL_SWITCH = "kill_switch"
    REGIME_CHANGE = "regime_change"
    ERROR = "error"
