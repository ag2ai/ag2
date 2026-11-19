# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for OpenAI Responses client usage normalization."""

import pytest
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails, ResponseUsage

from ag2 import Agent
from ag2.config.openai.mappers import normalize_responses_usage
from ag2.events import Usage

from ._helpers import capturing_config, config, message, response


class TestNormalizeUsage:
    def test_normalizes_input_output_keys(self):
        usage = ResponseUsage(
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            input_tokens_details=InputTokensDetails(cached_tokens=0, cache_write_tokens=0),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        )
        result = normalize_responses_usage(usage)
        assert result == Usage(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
            thinking_tokens=0,
        )

    def test_lifts_cached_tokens(self):
        usage = ResponseUsage(
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            input_tokens_details=InputTokensDetails(cached_tokens=80, cache_write_tokens=0),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        )
        result = normalize_responses_usage(usage)
        assert result == Usage(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            cache_read_input_tokens=80,
            cache_creation_input_tokens=0,
            thinking_tokens=0,
        )

    def test_lifts_reasoning_tokens(self):
        usage = ResponseUsage(
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            input_tokens_details=InputTokensDetails(cached_tokens=0, cache_write_tokens=0),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=10),
        )
        result = normalize_responses_usage(usage)
        # Reasoning tokens stay separate from the reported cache-write count.
        assert result == Usage(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
            thinking_tokens=10,
        )


@pytest.mark.asyncio
class TestCacheWriteRetention:
    async def test_12h_writes_are_reported_as_extended_cache_writes(self) -> None:
        agent = Agent(
            "assistant",
            config=config(
                response(
                    message("ok"),
                    usage={
                        "input_tokens": 1200,
                        "output_tokens": 20,
                        "total_tokens": 1220,
                        "input_tokens_details": {
                            "cached_tokens": 0,
                            "cache_write_tokens": 1000,
                            "cache_write_tokens_details": {"ttl_12h_tokens": 800, "ttl_30m_tokens": 200},
                        },
                        "output_tokens_details": {"reasoning_tokens": 0},
                    },
                )
            ),
        )

        reply = await agent.ask("Summarize the corpus.")

        assert (await reply.usage()).total == Usage(
            prompt_tokens=1200,
            completion_tokens=20,
            total_tokens=1220,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=1000,
            cache_creation_extended_input_tokens=800,
            thinking_tokens=0,
        )

    async def test_a_streaming_call_reports_them_from_the_completed_response(self) -> None:
        streaming, _ = capturing_config(
            response(
                message("ok"),
                usage={
                    "input_tokens": 1200,
                    "output_tokens": 20,
                    "total_tokens": 1220,
                    "input_tokens_details": {
                        "cached_tokens": 0,
                        "cache_write_tokens": 1000,
                        "cache_write_tokens_details": {"ttl_12h_tokens": 800, "ttl_30m_tokens": 200},
                    },
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            ),
            stream=True,
        )
        agent = Agent("assistant", config=streaming)

        reply = await agent.ask("Summarize the corpus.")

        assert (await reply.usage()).total == Usage(
            prompt_tokens=1200,
            completion_tokens=20,
            total_tokens=1220,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=1000,
            cache_creation_extended_input_tokens=800,
            thinking_tokens=0,
        )

    async def test_30m_writes_report_a_measured_zero(self) -> None:
        agent = Agent(
            "assistant",
            config=config(
                response(
                    message("ok"),
                    usage={
                        "input_tokens": 1200,
                        "output_tokens": 20,
                        "total_tokens": 1220,
                        "input_tokens_details": {
                            "cached_tokens": 0,
                            "cache_write_tokens": 1000,
                            "cache_write_tokens_details": {"ttl_12h_tokens": 0, "ttl_30m_tokens": 1000},
                        },
                        "output_tokens_details": {"reasoning_tokens": 0},
                    },
                )
            ),
        )

        reply = await agent.ask("Summarize the corpus.")

        assert (await reply.usage()).total == Usage(
            prompt_tokens=1200,
            completion_tokens=20,
            total_tokens=1220,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=1000,
            cache_creation_extended_input_tokens=0,
            thinking_tokens=0,
        )

    async def test_an_absent_breakdown_leaves_extended_writes_absent(self) -> None:
        agent = Agent(
            "assistant",
            config=config(
                response(
                    message("ok"),
                    usage={
                        "input_tokens": 1200,
                        "output_tokens": 20,
                        "total_tokens": 1220,
                        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 1000},
                        "output_tokens_details": {"reasoning_tokens": 0},
                    },
                )
            ),
        )

        reply = await agent.ask("Summarize the corpus.")

        assert (await reply.usage()).total == Usage(
            prompt_tokens=1200,
            completion_tokens=20,
            total_tokens=1220,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=1000,
            thinking_tokens=0,
        )
