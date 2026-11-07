"""Identifiers for people rows: 21-character url-safe ids, the shape Honcho's nanoid produces."""

import secrets

_ALPHABET = "_-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
NANOID_LENGTH = 21


def generate_nanoid() -> str:
    """A random 21-character id over ``[A-Za-z0-9_-]``."""
    return "".join(secrets.choice(_ALPHABET) for _ in range(NANOID_LENGTH))
