"""A cheap token estimate for the dreamer's budgets."""

from __future__ import annotations


def estimate_tokens(text: str | list[str] | None) -> int:
    """Estimate the token count of a text or a list of texts (four characters per token)."""
    if not text:
        return 0
    if isinstance(text, list):
        text = "\n".join(text)
    return (len(text) + 3) // 4
