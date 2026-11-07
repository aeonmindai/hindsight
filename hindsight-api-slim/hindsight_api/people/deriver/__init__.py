"""The deriver: turns what people said into conclusions about them.

``enqueue_messages`` stores retained messages and queues them, ``QueueManager`` polls the queue and
runs the deriver on each work unit, ``process_available`` drains the queue once synchronously.
"""

from .enqueue import IncomingMessage, enqueue_dream, enqueue_messages
from .queue_manager import QueueManager, process_available

__all__ = [
    "IncomingMessage",
    "QueueManager",
    "enqueue_dream",
    "enqueue_messages",
    "process_available",
]
