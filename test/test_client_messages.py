# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional
from unittest.mock import MagicMock, call

import pytest

from autogen.client_messages import (
    ActualUsageSummary,
    ModelUsageSummary,
    TotalUsageSummary,
    UsageSummary,
    _change_usage_summary_format,
    create_usage_summary_model,
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
    actual_usage_summary: Optional[dict[str, Any]],
    total_usage_summary: Optional[dict[str, Any]],
) -> None:
    actual = create_usage_summary_model(actual_usage_summary, total_usage_summary, "both")

    assert isinstance(actual, UsageSummary)
    assert isinstance(actual.actual, ActualUsageSummary)
    assert isinstance(actual.total, TotalUsageSummary)
    assert actual.mode == "both"

    assert isinstance(actual.actual.usages, list)
    assert len(actual.actual.usages) == 1
    assert isinstance(actual.actual.usages[0], ModelUsageSummary)
    assert actual.actual.total_cost == 4.23e-05
    assert actual.actual.usages[0].model == "gpt-4o-mini-2024-07-18"
    assert actual.actual.usages[0].completion_tokens == 25
    assert actual.actual.usages[0].cost == 4.23e-05
    assert actual.actual.usages[0].prompt_tokens == 182
    assert actual.actual.usages[0].total_tokens == 207

    assert isinstance(actual.total.usages, list)
    assert len(actual.total.usages) == 1
    assert isinstance(actual.total.usages[0], ModelUsageSummary)
    assert actual.total.total_cost == 4.23e-05
    assert actual.total.usages[0].model == "gpt-4o-mini-2024-07-18"
    assert actual.total.usages[0].completion_tokens == 25
    assert actual.total.usages[0].cost == 4.23e-05
    assert actual.total.usages[0].prompt_tokens == 182
    assert actual.total.usages[0].total_tokens == 207

    mock = MagicMock()
    actual.print(f=mock)

    print(mock.call_args_list)

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
    actual_usage_summary: Optional[dict[str, Any]],
    total_usage_summary: Optional[dict[str, Any]],
) -> None:
    actual = create_usage_summary_model(actual_usage_summary, total_usage_summary, "both")

    assert isinstance(actual, UsageSummary)
    assert isinstance(actual.actual, ActualUsageSummary)
    assert isinstance(actual.total, TotalUsageSummary)
    assert actual.mode == "both"

    assert isinstance(actual.actual.usages, list)
    assert len(actual.actual.usages) == 1
    assert isinstance(actual.actual.usages[0], ModelUsageSummary)
    assert actual.actual.total_cost == 4.23e-05
    assert actual.actual.usages[0].model == "gpt-4o-mini-2024-07-18"
    assert actual.actual.usages[0].completion_tokens == 25
    assert actual.actual.usages[0].cost == 4.23e-05
    assert actual.actual.usages[0].prompt_tokens == 182
    assert actual.actual.usages[0].total_tokens == 207

    assert isinstance(actual.total.usages, list)
    assert len(actual.total.usages) == 1
    assert isinstance(actual.total.usages[0], ModelUsageSummary)
    assert actual.total.total_cost == 4.23e-05 * 40
    assert actual.total.usages[0].model == "gpt-4o-mini-2024-07-18"
    assert actual.total.usages[0].completion_tokens == 25 * 40
    assert actual.total.usages[0].cost == 4.23e-05 * 40
    assert actual.total.usages[0].prompt_tokens == 182 * 40
    assert actual.total.usages[0].total_tokens == 207 * 40

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
    actual_usage_summary: Optional[dict[str, Any]],
    total_usage_summary: Optional[dict[str, Any]],
) -> None:
    actual = create_usage_summary_model(actual_usage_summary, total_usage_summary, "both")

    assert isinstance(actual, UsageSummary)
    assert isinstance(actual.actual, ActualUsageSummary)
    assert isinstance(actual.total, TotalUsageSummary)
    assert actual.mode == "both"

    assert actual.actual.usages is None
    assert actual.actual.total_cost is None

    assert actual.total.usages is None
    assert actual.total.total_cost is None

    mock = MagicMock()
    actual.print(f=mock)

    # print(mock.call_args_list)

    expected_call_args_list = [call('No usage summary. Please call "create" first.', flush=True)]

    assert mock.call_args_list == expected_call_args_list
