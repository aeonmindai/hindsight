"""A reply that is still not JSON after the provider's own retries is answered by the fallback model, like a refusal."""

import json

import pytest

from hindsight_api.engine.llm_interface import LLMInterface
from hindsight_api.engine.providers.broker_llm import RefusalFallbackLLM


class _Stub(LLMInterface):
    def __init__(self, model: str, outcome):
        super().__init__(provider="broker", api_key="k", base_url="http://x", model=model, reasoning_effort=None, timeout=1)
        self.outcome = outcome
        self.calls = 0

    async def verify_connection(self) -> None:
        return None

    async def call(self, messages, *args, **kwargs):
        self.calls += 1
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome

    async def cleanup(self) -> None:
        return None


def _not_json() -> json.JSONDecodeError:
    return json.JSONDecodeError("Expecting value", "", 0)


@pytest.mark.asyncio
async def test_a_reply_that_is_not_json_is_answered_by_the_fallback():
    primary, fallback = _Stub("sonnet", _not_json()), _Stub("terra", {"facts": []})
    assert await RefusalFallbackLLM(primary, fallback).call([{"role": "user", "content": "x"}]) == {"facts": []}
    assert (primary.calls, fallback.calls) == (1, 1)


@pytest.mark.asyncio
async def test_when_the_fallback_is_not_json_either_the_primarys_error_is_raised():
    primary, fallback = _Stub("sonnet", _not_json()), _Stub("terra", json.JSONDecodeError("Extra data", "", 2))
    with pytest.raises(json.JSONDecodeError, match="Expecting value"):
        await RefusalFallbackLLM(primary, fallback).call([{"role": "user", "content": "x"}])


@pytest.mark.asyncio
async def test_any_other_error_never_reaches_the_fallback():
    primary, fallback = _Stub("sonnet", TimeoutError("slow")), _Stub("terra", {"facts": []})
    with pytest.raises(TimeoutError):
        await RefusalFallbackLLM(primary, fallback).call([{"role": "user", "content": "x"}])
    assert fallback.calls == 0
