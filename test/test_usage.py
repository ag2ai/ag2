# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2.events import ModelMessage, ModelResponse, Usage, UsageEvent
from ag2.usage import UsageRecord, UsageReport


def _usage_event(prompt: int, completion: int, *, model: str, provider: str) -> UsageEvent:
    return UsageEvent(
        Usage(prompt_tokens=prompt, completion_tokens=completion),
        kind="model_call",
        model=model,
        provider=provider,
    )


class TestUsageArithmetic:
    def test_add_is_field_wise(self) -> None:
        a = Usage(prompt_tokens=10, completion_tokens=5)
        b = Usage(prompt_tokens=3, completion_tokens=7, total_tokens=20)
        # total present on only one side coerces the other to 0
        assert a + b == Usage(prompt_tokens=13, completion_tokens=12, total_tokens=20)

    def test_add_none_stays_none(self) -> None:
        assert Usage() + Usage() == Usage()

    def test_add_rejects_non_usage(self) -> None:
        with pytest.raises(TypeError):
            Usage() + 42  # type: ignore[operator]

    def test_builtin_sum_aggregates(self) -> None:
        usages = [
            Usage(prompt_tokens=1, completion_tokens=2),
            Usage(prompt_tokens=4, completion_tokens=8),
            Usage(),
        ]
        assert sum(usages, Usage()) == Usage(prompt_tokens=5, completion_tokens=10)


class TestUsageReport:
    def test_groups_by_model_and_provider(self) -> None:
        report = UsageReport.from_events([
            _usage_event(10, 4, model="claude", provider="anthropic"),
            _usage_event(6, 2, model="claude", provider="anthropic"),
            _usage_event(5, 1, model="gpt", provider="openai"),
        ])

        assert report.total == Usage(prompt_tokens=21, completion_tokens=7)
        assert report.by_model == {
            "claude": Usage(prompt_tokens=16, completion_tokens=6),
            "gpt": Usage(prompt_tokens=5, completion_tokens=1),
        }
        assert report.by_provider == {
            "anthropic": Usage(prompt_tokens=16, completion_tokens=6),
            "openai": Usage(prompt_tokens=5, completion_tokens=1),
        }
        assert report.by_kind == {"model_call": Usage(prompt_tokens=21, completion_tokens=7)}
        assert len(report.records) == 3

    def test_subtask_rollup_has_no_double_count(self) -> None:
        # Parent made one model call; a subtask rolled up its own usage as a
        # single subtask UsageEvent (child UsageEvents never reach parent
        # history). The report sums both with no double counting.
        report = UsageReport.from_events([
            _usage_event(10, 4, model="claude", provider="anthropic"),
            UsageEvent(
                Usage(prompt_tokens=100, completion_tokens=50),
                kind="subtask",
                label="worker",
            ),
        ])

        assert report.total == Usage(prompt_tokens=110, completion_tokens=54)
        assert report.by_kind == {
            "model_call": Usage(prompt_tokens=10, completion_tokens=4),
            "subtask": Usage(prompt_tokens=100, completion_tokens=50),
        }
        # subtask usage carries no single-model attribution
        assert report.by_model == {"claude": Usage(prompt_tokens=10, completion_tokens=4)}
        assert (
            UsageRecord(
                usage=Usage(prompt_tokens=100, completion_tokens=50),
                kind="subtask",
                label="worker",
            )
            in report.records
        )

    def test_internal_calls_are_captured_by_kind(self) -> None:
        # Compaction / aggregation calls run on throwaway streams but surface
        # their usage onto the agent stream, so the report must see them.
        report = UsageReport.from_events([
            _usage_event(10, 4, model="claude", provider="anthropic"),
            UsageEvent(Usage(prompt_tokens=30, completion_tokens=5), kind="compaction"),
            UsageEvent(Usage(prompt_tokens=20, completion_tokens=3), kind="aggregation"),
        ])

        assert report.total == Usage(prompt_tokens=60, completion_tokens=12)
        assert report.by_kind == {
            "model_call": Usage(prompt_tokens=10, completion_tokens=4),
            "compaction": Usage(prompt_tokens=30, completion_tokens=5),
            "aggregation": Usage(prompt_tokens=20, completion_tokens=3),
        }

    def test_response_usage_is_not_double_counted(self) -> None:
        # ModelResponse still carries usage for local inspection, but the
        # report aggregates UsageEvents alone — the single source of truth.
        report = UsageReport.from_events([
            ModelResponse(
                message=ModelMessage("hi"),
                usage=Usage(prompt_tokens=999, completion_tokens=999),
                model="claude",
                provider="anthropic",
            ),
            _usage_event(3, 1, model="claude", provider="anthropic"),
        ])
        assert len(report.records) == 1
        assert report.total == Usage(prompt_tokens=3, completion_tokens=1)

    def test_skips_empty_usage(self) -> None:
        report = UsageReport.from_events([
            UsageEvent(Usage(), kind="model_call"),
            _usage_event(3, 1, model="claude", provider="anthropic"),
        ])
        assert len(report.records) == 1
        assert report.total == Usage(prompt_tokens=3, completion_tokens=1)

    def test_empty_report(self) -> None:
        assert UsageReport.from_events([]) == UsageReport()
