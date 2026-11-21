"""Per-job pub/sub event broker for Server-Sent Events (SSE)."""

import asyncio
from typing import Dict, Set, Any
import json


class EventBroker:
    """Thread-safe event broker delivering streaming updates to SSE subscribers."""

    def __init__(self):
        self._subscribers: Dict[str, Set[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, job_id: str) -> asyncio.Queue:
        """Subscribe to events for a specific job."""
        async with self._lock:
            if job_id not in self._subscribers:
                self._subscribers[job_id] = set()
            queue: asyncio.Queue = asyncio.Queue()
            self._subscribers[job_id].add(queue)
            return queue

    async def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        """Remove a subscriber queue when client disconnects."""
        async with self._lock:
            if job_id in self._subscribers:
                self._subscribers[job_id].discard(queue)
                if not self._subscribers[job_id]:
                    del self._subscribers[job_id]

    async def publish(self, job_id: str, event_type: str, data: Any) -> None:
        """Broadcast an event to all active subscribers for job_id."""
        payload = {
            "event": event_type,
            "data": data if isinstance(data, str) else json.dumps(data)
        }
        async with self._lock:
            queues = list(self._subscribers.get(job_id, []))

        for q in queues:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                pass


event_broker = EventBroker()
