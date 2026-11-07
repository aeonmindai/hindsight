"""Errors the people modules raise. The HTTP router maps them to 404, 422 and 409."""


class PeopleError(Exception):
    """Base class of the people errors."""


class ResourceNotFoundException(PeopleError):
    """A peer, session or conclusion does not exist in the bank."""


class ValidationException(PeopleError):
    """The input cannot be stored (bad name, NUL bytes, over-length text)."""


class ConflictException(PeopleError):
    """Concurrent writers collided and the retry did not resolve it."""
