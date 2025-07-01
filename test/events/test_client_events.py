# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional
from unittest.mock import MagicMock, call
from uuid import UUID

import pytest

from autogen.events.client_events import (
    StreamEvent,
    UsageSummaryEvent,
    _change_usage_summary_format,
)


@pytest.mark.parametrize(
    "actual_usage_summary, total_usage_summary, expected",
    [
        (
            {
                "gpt-4o-mini-2024-07-18": {
                    "completion_tokens": 25,
                    "cost": 4.23e-05,
                    "prompt_tokens": 182,
                    "total_tokens": 207,
                },
                "total_cost": 4.23e-05,
            },
            {
                "gpt-4o-mini-2024-07-18": {
                    "completion_tokens": 25,
                    "cost": 4.23e-05,
                    "prompt_tokens": 182,
                    "total_tokens": 207,
                },
                "total_cost": 4.23e-05,
            },
            {
                "actual": {
                    "usages": [
                        {
                            "model": "gpt-4o-mini-2024-07-18",
                            "completion_tokens": 25,
                            "cost": 4.23e-05,
                            "prompt_tokens": 182,
                            "total_tokens": 207,
                        }
                    ],
                    "total_cost": 4.23e-05,
                },
                "total": {
                    "usages": [
                        {
                            "model": "gpt-4o-mini-2024-07-18",
                            "completion_tokens": 25,
                            "cost": 4.23e-05,
                            "prompt_tokens": 182,
                            "total_tokens": 207,
                        }
                    ],
                    "total_cost": 4.23e-05,
                },
            },
        ),
        (None, None, {"actual": {"usages": None, "total_cost": None}, "total": {"usages": None, "total_cost": None}}),
    ],
)
def test__change_usage_summary_format(
    actual_usage_summary: Optional[dict[str, Any]],
    total_usage_summary: Optional[dict[str, Any]],
    expected: dict[str, dict[str, Any]],
) -> None:
    summary_dict = _change_usage_summary_format(actual_usage_summary, total_usage_summary)
    assert summary_dict == expected


class TestUsageSummaryEvent:
    @pytest.mark.parametrize(
        "actual_usage_summary, total_usage_summary",
        [
            (
                {
                    "gpt-4o-mini-2024-07-18": {
                        "completion_tokens": 25,
                        "cost": 4.23e-05,
                        "prompt_tokens": 182,
                        "total_tokens": 207,
                    },
                    "total_cost": 4.23e-05,
                },
                {
                    "gpt-4o-mini-2024-07-18": {
                        "completion_tokens": 25,
                        "cost": 4.23e-05,
                        "prompt_tokens": 182,
                        "total_tokens": 207,
                    },
                    "total_cost": 4.23e-05,
                },
            ),
        ],
    )
    def test_usage_summary_print_same_actual_and_total(
        self,
        actual_usage_summary: Optional[dict[str, Any]],
        total_usage_summary: Optional[dict[str, Any]],
        uuid: UUID,
    ) -> None:
        actual = UsageSummaryEvent(
            uuid=uuid, actual_usage_summary=actual_usage_summary, total_usage_summary=total_usage_summary, mode="both"
        )
        assert isinstance(actual, UsageSummaryEvent)

        expected_model_dump = {
            "type": "usage_summary",
            "content": {
                "uuid": uuid,
                "actual": {
                    "usages": [
                        {
                            "model": "gpt-4o-mini-2024-07-18",
                            "completion_tokens": 25,
                            "cost": 4.23e-05,
                            "prompt_tokens": 182,
                            "total_tokens": 207,
                        }
                    ],
                    "total_cost": 4.23e-05,
                },
                "total": {
                    "usages": [
                        {
                            "model": "gpt-4o-mini-2024-07-18",
                            "completion_tokens": 25,
                            "cost": 4.23e-05,
                            "prompt_tokens": 182,
                            "total_tokens": 207,
                        }
                    ],
                    "total_cost": 4.23e-05,
                },
                "mode": "both",
            },
        }
        assert actual.model_dump() == expected_model_dump

        mock = MagicMock()
        actual.print(f=mock)

        # print(mock.call_args_list)

        expected_call_args_list = [
            call(
                "----------------------------------------------------------------------------------------------------",
                flush=True,
            ),
            call("Usage summary excluding cached usage: ", flush=True),
            call("Total cost: 4e-05", flush=True),
            call(
                "* Model 'gpt-4o-mini-2024-07-18': cost: 4e-05, prompt_tokens: 182, completion_tokens: 25, total_tokens: 207",
                flush=True,
            ),
            call(),
            call(
                "All completions are non-cached: the total cost with cached completions is the same as actual cost.",
                flush=True,
            ),
            call(
                "----------------------------------------------------------------------------------------------------",
                flush=True,
            ),
        ]

        assert mock.call_args_list == expected_call_args_list

    @pytest.mark.parametrize(
        "actual_usage_summary, total_usage_summary",
        [
            (
                {
                    "gpt-4o-mini-2024-07-18": {
                        "completion_tokens": 25,
                        "cost": 4.23e-05,
                        "prompt_tokens": 182,
                        "total_tokens": 207,
                    },
                    "total_cost": 4.23e-05,
                },
                {
                    "gpt-4o-mini-2024-07-18": {
                        "completion_tokens": 25 * 40,
                        "cost": 4.23e-05 * 40,
                        "prompt_tokens": 182 * 40,
                        "total_tokens": 207 * 40,
                    },
                    "total_cost": 4.23e-05 * 40,
                },
            ),
        ],
    )
    def test_usage_summary_print_different_actual_and_total(
        self,
        actual_usage_summary: Optional[dict[str, Any]],
        total_usage_summary: Optional[dict[str, Any]],
        uuid: UUID,
    ) -> None:
        actual = UsageSummaryEvent(
            uuid=uuid, actual_usage_summary=actual_usage_summary, total_usage_summary=total_usage_summary, mode="both"
        )
        assert isinstance(actual, UsageSummaryEvent)

        expected_model_dump = {
            "type": "usage_summary",
            "content": {
                "uuid": uuid,
                "actual": {
                    "usages": [
                        {
                            "model": "gpt-4o-mini-2024-07-18",
                            "completion_tokens": 25,
                            "cost": 4.23e-05,
                            "prompt_tokens": 182,
                            "total_tokens": 207,
                        }
                    ],
                    "total_cost": 4.23e-05,
                },
                "total": {
                    "usages": [
                        {
                            "model": "gpt-4o-mini-2024-07-18",
                            "completion_tokens": 1000,
                            "cost": 0.001692,
                            "prompt_tokens": 7280,
                            "total_tokens": 8280,
                        }
                    ],
                    "total_cost": 0.001692,
                },
                "mode": "both",
            },
        }
        assert actual.model_dump() == expected_model_dump

        mock = MagicMock()
        actual.print(f=mock)

        # print(mock.call_args_list)

        expected_call_args_list = [
            call(
                "----------------------------------------------------------------------------------------------------",
                flush=True,
            ),
            call("Usage summary excluding cached usage: ", flush=True),
            call("Total cost: 4e-05", flush=True),
            call(
                "* Model 'gpt-4o-mini-2024-07-18': cost: 4e-05, prompt_tokens: 182, completion_tokens: 25, total_tokens: 207",
                flush=True,
            ),
            call(),
            call("Usage summary including cached usage: ", flush=True),
            call("Total cost: 0.00169", flush=True),
            call(
                "* Model 'gpt-4o-mini-2024-07-18': cost: 0.00169, prompt_tokens: 7280, completion_tokens: 1000, total_tokens: 8280",
                flush=True,
            ),
            call(
                "----------------------------------------------------------------------------------------------------",
                flush=True,
            ),
        ]

        assert mock.call_args_list == expected_call_args_list

    @pytest.mark.parametrize(
        "actual_usage_summary, total_usage_summary",
        [
            (
                None,
                None,
            ),
        ],
    )
    def test_usage_summary_print_none_actual_and_total(
        self,
        actual_usage_summary: Optional[dict[str, Any]],
        total_usage_summary: Optional[dict[str, Any]],
        uuid: UUID,
    ) -> None:
        actual = UsageSummaryEvent(
            uuid=uuid, actual_usage_summary=actual_usage_summary, total_usage_summary=total_usage_summary, mode="both"
        )
        assert isinstance(actual, UsageSummaryEvent)

        expected_model_dump = {
            "type": "usage_summary",
            "content": {
                "uuid": uuid,
                "actual": {"usages": None, "total_cost": None},
                "total": {"usages": None, "total_cost": None},
                "mode": "both",
            },
        }
        assert actual.model_dump() == expected_model_dump

        mock = MagicMock()
        actual.print(f=mock)

        # print(mock.call_args_list)

        expected_call_args_list = [call('No usage summary. Please call "create" first.', flush=True)]

        assert mock.call_args_list == expected_call_args_list


class TestStreamEvent:
    def test_print(self, uuid: UUID) -> None:
        content = "random stream chunk content"
        stream_message = StreamEvent(uuid=uuid, content=content)
        assert isinstance(stream_message, StreamEvent)

        expected_model_dump = {
            "type": "stream",
            "content": {
                "uuid": uuid,
                "content": content,
            },
        }
        assert stream_message.model_dump() == expected_model_dump

        mock = MagicMock()
        stream_message.print(f=mock)

        # print(mock.call_args_list)

        expected_call_args_list = [
            call("\x1b[32m", end=""),
            call("random stream chunk content", end="", flush=True),
            call("\x1b[0m\n"),
        ]

        assert mock.call_args_list == expected_call_args_list
