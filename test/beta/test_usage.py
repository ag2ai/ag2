# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

from autogen.beta import Usage, UsageRecord, UsageReport
from autogen.beta.events import ModelMessage, ModelResponse, TaskCompleted


def _response(prompt: int, completion: int, *, model: str, provider: str) -> ModelResponse:
    return ModelResponse(
        message=ModelMessage("hi"),
        usage=Usage(prompt_tokens=prompt, completion_tokens=completion),
        model=model,
        provider=provider,
    )


def _task_completed(usage: Usage, *, agent_name: str) -> TaskCompleted:
    return TaskCompleted(
        task_id="t1",
        agent_name=agent_name,
        objective="do work",
        task_stream=uuid4(),
        usage=usage,
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
        assert Usage().__add__(42) is NotImplemented  # type: ignore[arg-type]

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
            _response(10, 4, model="claude", provider="anthropic"),
            _response(6, 2, model="claude", provider="anthropic"),
            _response(5, 1, model="gpt", provider="openai"),
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
        assert report.cost is None

    def test_subtask_rollup_has_no_double_count(self) -> None:
        # Parent made one model call; a subtask rolled up its own usage via
        # TaskCompleted (child ModelResponses never reach parent history).
        report = UsageReport.from_events([
            _response(10, 4, model="claude", provider="anthropic"),
            _task_completed(Usage(prompt_tokens=100, completion_tokens=50), agent_name="worker"),
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

    def test_skips_empty_usage(self) -> None:
        report = UsageReport.from_events([
            ModelResponse(message=ModelMessage("no usage")),
            _response(3, 1, model="claude", provider="anthropic"),
        ])
        assert len(report.records) == 1
        assert report.total == Usage(prompt_tokens=3, completion_tokens=1)

    def test_cost_model_hook_populates_cost(self) -> None:
        class FlatCost:
            def price(self, usage: Usage, model: str | None, provider: str | None) -> float | None:
                return (usage.prompt_tokens or 0) * 0.001

        report = UsageReport.from_events(
            [_response(1000, 0, model="claude", provider="anthropic")],
            cost_model=FlatCost(),
        )
        assert report.cost == 1.0

    def test_empty_report(self) -> None:
        assert UsageReport.from_events([]) == UsageReport()
