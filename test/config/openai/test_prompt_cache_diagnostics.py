# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A prompt cache miss says why, instead of leaving a caller to guess from a token count.

Responses only: `prompt_cache_diagnostics` is a field of the Responses API, and chat
completions carry no equivalent.
"""

from typing import Any

import pytest
from dirty_equals import IsPartialDict

from ag2 import MemoryStream
from ag2.config.openai.events import OpenAIPromptCacheDiagnostics
from ag2.events import Usage

from ._helpers import ask_client, capturing_config, recording, response

CACHE_MISS: dict[str, Any] = {
    "type": "cache_miss",
    "reason": "tools_changed",
    "cache_missed_tokens": 1792,
    "comparison_reusable_tokens": 256,
}


def _miss(**overrides: Any) -> OpenAIPromptCacheDiagnostics:
    """The event `CACHE_MISS` should produce, with `overrides` applied."""
    reported: dict[str, Any] = {
        "reason": "tools_changed",
        "cache_missed_tokens": 1792,
        "comparison_reusable_tokens": 256,
    }
    return OpenAIPromptCacheDiagnostics("cache_miss", **{**reported, **overrides})


async def _diagnostics_of(payload: dict[str, Any], *, stream: bool = False) -> list[OpenAIPromptCacheDiagnostics]:
    """The diagnostics one turn put on the stream — transient, so history holds none."""
    config, _ = capturing_config(payload, stream=stream)
    memory = MemoryStream()
    captured = recording(memory)

    await ask_client(config.create(), stream=memory)

    return [e for e in captured if isinstance(e, OpenAIPromptCacheDiagnostics)]


@pytest.mark.asyncio
class TestRequestingDiagnostics:
    """Diagnostics are asked for by naming a response to compare this one against."""

    async def test_nothing_is_sent_by_default(self) -> None:
        config, bodies = capturing_config(response())

        await ask_client(config.create())

        assert "prompt_cache_options" not in bodies[0]

    async def test_the_first_call_has_nothing_to_compare_against(self) -> None:
        config, bodies = capturing_config(response())

        await ask_client(config.copy(prompt_cache_diagnostics=True).create())

        # An empty options object is a 400 on a model without the feature, so none is sent.
        assert "prompt_cache_options" not in bodies[0]

    async def test_a_later_call_names_the_one_before_it(self) -> None:
        config, bodies = capturing_config(response(response_id="resp_1"), response(response_id="resp_2"))
        client = config.copy(prompt_cache_diagnostics=True).create()

        await ask_client(client)
        await ask_client(client)

        assert bodies[1] == IsPartialDict({"prompt_cache_options": {"comparison_response_id": "resp_1"}})

    async def test_the_callers_own_options_survive_the_injection(self) -> None:
        config, bodies = capturing_config(response(response_id="resp_1"), response(response_id="resp_2"))
        client = config.copy(
            prompt_cache_diagnostics=True,
            prompt_cache_options={"mode": "implicit", "ttl": "30m"},
        ).create()

        await ask_client(client)
        await ask_client(client)

        assert bodies[1] == IsPartialDict({
            "prompt_cache_options": {
                "mode": "implicit",
                "ttl": "30m",
                "comparison_response_id": "resp_1",
            }
        })

    async def test_an_explicit_comparison_id_outranks_the_chain(self) -> None:
        config, bodies = capturing_config(response(response_id="resp_1"), response(response_id="resp_2"))
        client = config.copy(
            prompt_cache_diagnostics=True,
            prompt_cache_options={"comparison_response_id": "resp_chosen"},
        ).create()

        await ask_client(client)
        await ask_client(client)

        assert bodies[1] == IsPartialDict({"prompt_cache_options": {"comparison_response_id": "resp_chosen"}})

    async def test_a_call_without_an_id_leaves_the_previous_one_standing(self) -> None:
        config, bodies = capturing_config(
            response(response_id="resp_1"),
            response(response_id=""),
            response(response_id="resp_3"),
        )
        client = config.copy(prompt_cache_diagnostics=True).create()

        await ask_client(client)
        await ask_client(client)
        await ask_client(client)

        assert bodies[2] == IsPartialDict({"prompt_cache_options": {"comparison_response_id": "resp_1"}})


@pytest.mark.asyncio
class TestReportingTheOutcome:
    """Every outcome reaches the application under its own name, including ones added later."""

    async def test_a_response_without_diagnostics_reports_nothing(self) -> None:
        assert await _diagnostics_of(response()) == []

    @pytest.mark.parametrize("outcome", ["cache_hit", "comparison_response_not_found", "unavailable"])
    async def test_an_outcome_without_a_miss_carries_no_estimates(self, outcome: str) -> None:
        payload = response(prompt_cache_diagnostics={"type": outcome})

        assert await _diagnostics_of(payload) == [OpenAIPromptCacheDiagnostics(outcome)]

    async def test_a_miss_carries_its_reason_and_its_estimates(self) -> None:
        assert await _diagnostics_of(response(prompt_cache_diagnostics=CACHE_MISS)) == [_miss()]

    async def test_an_unknown_reason_reaches_the_application_verbatim(self) -> None:
        payload = response(prompt_cache_diagnostics={**CACHE_MISS, "reason": "moon_phase_changed"})

        assert await _diagnostics_of(payload) == [_miss(reason="moon_phase_changed")]

    async def test_an_unknown_outcome_is_not_read_as_a_miss(self) -> None:
        """The SDK resolves an unknown `type` to the union's first variant, which is the miss."""
        payload = response(prompt_cache_diagnostics={**CACHE_MISS, "type": "cache_evicted"})

        # Named, but with none of the miss's fields harvested on its behalf.
        assert await _diagnostics_of(payload) == [OpenAIPromptCacheDiagnostics("cache_evicted")]

    async def test_diagnostics_without_a_discriminator_name_no_outcome(self) -> None:
        """There is no truthful name for an outcome the payload did not state."""
        assert await _diagnostics_of(response(prompt_cache_diagnostics={"cache_missed_tokens": 8})) == []

    async def test_a_streamed_turn_reports_the_same_outcome(self) -> None:
        payload = response(prompt_cache_diagnostics=CACHE_MISS)

        assert await _diagnostics_of(payload, stream=True) == [_miss()]


@pytest.mark.asyncio
class TestNamingTheComparison:
    """The report says which response it was compared against, whoever supplied the id."""

    async def test_the_reply_names_it(self) -> None:
        payload = response(
            prompt_cache_diagnostics=CACHE_MISS,
            prompt_cache_options={"mode": "implicit", "ttl": "30m", "comparison_response_id": "resp_0"},
        )

        assert await _diagnostics_of(payload) == [_miss(comparison_response_id="resp_0")]

    async def test_an_unechoed_id_falls_back_to_the_one_sent(self) -> None:
        """A reply that omits the options it was given still gets an attributable report."""
        config, _ = capturing_config(
            response(response_id="resp_1"),
            response(response_id="resp_2", prompt_cache_diagnostics=CACHE_MISS),
        )
        client = config.copy(prompt_cache_diagnostics=True).create()

        await ask_client(client)
        memory = MemoryStream()
        captured = recording(memory)
        await ask_client(client, stream=memory)

        assert [e for e in captured if isinstance(e, OpenAIPromptCacheDiagnostics)] == [
            _miss(comparison_response_id="resp_1")
        ]


@pytest.mark.asyncio
class TestWhatDiagnosticsAreNot:
    """An estimate about a counterfactual is not a measured token count."""

    async def test_estimates_stay_out_of_usage(self) -> None:
        config, _ = capturing_config(response(prompt_cache_diagnostics=CACHE_MISS))

        reply = await ask_client(config.create())

        # Whole-object comparison: no field of Usage absorbed 1792 or 256.
        assert reply.usage == Usage(
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            cache_read_input_tokens=0,
            thinking_tokens=0,
        )

    async def test_diagnostics_are_not_part_of_the_conversation(self) -> None:
        config, _ = capturing_config(response(prompt_cache_diagnostics=CACHE_MISS))
        memory = MemoryStream()

        await ask_client(config.create(), stream=memory)
        history = list(await memory.history.get_events())

        assert OpenAIPromptCacheDiagnostics.__transient__ is True
        assert not [e for e in history if isinstance(e, OpenAIPromptCacheDiagnostics)]
