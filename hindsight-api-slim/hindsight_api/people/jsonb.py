"""Reading and writing jsonb columns through the engine's connections.

asyncpg returns jsonb as text, and binds jsonb parameters from text. Write with ``to_jsonb`` and
``$n::jsonb``; read with ``from_jsonb``.
"""

import json
from typing import Any


def to_jsonb(value: Any) -> str:
    """A JSON string for a ``$n::jsonb`` parameter."""
    return json.dumps(value, default=str)


def from_jsonb(value: Any, default: Any = None) -> Any:
    """The Python value of a jsonb column (``default`` when it is NULL)."""
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value
