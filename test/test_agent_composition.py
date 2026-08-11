# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import BaseModel

from ag2 import Agent, TaskConfig, tool
from ag2.middleware import (
    ApprovalRequired,
    LoggingMiddleware,
    Middleware,
    TokenLimiter,
)
from ag2.observers import observer


def alpha(x: int) -> int:
    """Alpha."""
    return x


def beta(x: int) -> int:
    """Beta."""
    return x


def make_hook():  # type: ignore[no-untyped-def]
    async def hook(call_next, event, context):  # type: ignore[no-untyped-def]
        return await call_next(event, context)

    return hook


class Out(BaseModel):
    value: int


class TestAgentComposition:
    def test_reports_what_the_agent_is_made_of(self) -> None:
        agent = Agent(
            "bot",
            prompt="hello",
            tools=[alpha],
            dependencies={"db": object()},
            variables={"k": 1},
        )

        assert agent.system_prompt == ("hello",)
        assert [t.name for t in agent.tools] == ["alpha"]
        assert list(agent.dependencies) == ["db"]
        assert list(agent.variables) == ["k"]

    def test_unset_slots_read_as_defaults(self) -> None:
        agent = Agent("bot", prompt="p")

        assert agent.dynamic_prompt == ()
        assert agent.observers == ()
        assert agent.assembly == ()
        assert agent.tasks is None
        assert agent.response_schema is None

    def test_set_slots_are_reported(self) -> None:
        @observer
        def watcher(event, context) -> None:  # type: ignore[no-untyped-def]
            return None

        agent = Agent("bot", prompt="p", observers=[watcher], tasks=TaskConfig(), response_schema=Out)

        assert len(agent.observers) == 1
        assert isinstance(agent.tasks, TaskConfig)
        assert agent.response_schema is not None

    def test_dynamic_prompt_hooks_are_nameable(self) -> None:
        agent = Agent("bot", prompt="p")

        @agent.prompt
        async def greeting() -> str:
            return "hi"

        assert [hook.__name__ for hook in agent.dynamic_prompt] == ["greeting"]


class TestViewsAreReadOnly:
    def test_sequences_are_immutable_snapshots(self) -> None:
        agent = Agent("bot", prompt="p", tools=[alpha])

        assert isinstance(agent.system_prompt, tuple)
        assert isinstance(agent.dynamic_prompt, tuple)
        assert isinstance(agent.middleware, tuple)
        assert isinstance(agent.observers, tuple)
        assert isinstance(agent.assembly, tuple)

    def test_mappings_cannot_be_mutated_through_the_view(self) -> None:
        agent = Agent("bot", prompt="p", dependencies={"db": object()}, variables={"k": 1})

        with pytest.raises(TypeError):
            agent.dependencies["db"] = object()  # type: ignore[index]

        with pytest.raises(TypeError):
            agent.variables["k"] = 2  # type: ignore[index]

    def test_dependency_values_are_the_injected_objects(self) -> None:
        db = object()
        agent = Agent("bot", prompt="p", dependencies={"db": db})

        assert agent.dependencies["db"] is db


class TestMiddlewareEntries:
    def test_agent_middleware_pairs_object_with_description(self) -> None:
        limiter = TokenLimiter(max_tokens=10)
        agent = Agent("bot", prompt="p", middleware=[limiter])

        entry = agent.middleware[0]

        assert entry.middleware is limiter
        assert entry.description.kind == "TokenLimiter"
        assert entry.description.config == {"max_tokens": 10, "chars_per_token": 4}

    def test_tool_middleware_pairs_object_with_description(self) -> None:
        guard = ApprovalRequired(timeout=5)
        built = tool(alpha, middleware=[guard])

        entry = built.middleware[0]

        assert entry.middleware is guard
        assert entry.description.kind == "ApprovalRequired"

    def test_middleware_without_describe_reports_incomplete(self) -> None:
        agent = Agent("bot", prompt="p", middleware=[make_hook()])

        description = agent.middleware[0].description

        assert description.complete is False
        assert description.config == {}

    def test_entries_are_built_on_access(self) -> None:
        limiter = TokenLimiter(max_tokens=10)
        agent = Agent("bot", prompt="p", middleware=[limiter])

        # The entry is a fresh view each time; the middleware inside is not.
        assert agent.middleware[0] is not agent.middleware[0]
        assert agent.middleware[0].middleware is agent.middleware[0].middleware


class TestSharingIsObservable:
    def test_one_instance_across_two_tools_reads_as_the_same_object(self) -> None:
        shared = make_hook()
        agent = Agent(
            "bot",
            prompt="p",
            tools=[tool(alpha, middleware=[shared]), tool(beta, middleware=[shared])],
        )
        first, second = agent.tools

        assert first.middleware[0].middleware is second.middleware[0].middleware

    def test_separate_instances_with_equal_config_are_not_the_same_object(self) -> None:
        agent = Agent(
            "bot",
            prompt="p",
            tools=[tool(alpha, middleware=[make_hook()]), tool(beta, middleware=[make_hook()])],
        )
        first, second = agent.tools

        # Descriptions cannot tell these apart, which is why identity is exposed.
        assert first.middleware[0].description == second.middleware[0].description
        assert first.middleware[0].middleware is not second.middleware[0].middleware


class TestMiddlewareFactoryFields:
    def test_reports_the_wrapped_class_and_its_options(self) -> None:
        factory = Middleware(LoggingMiddleware, level=10)

        assert factory.cls is LoggingMiddleware
        assert dict(factory.options) == {"level": 10}

    def test_options_expose_values_that_the_description_withholds(self) -> None:
        factory = Middleware(LoggingMiddleware, api_key="sk-SECRET-123")

        # A description may be logged or committed as a fixture, so it reports
        # names only. Reading the factory you already hold is a deliberate act.
        assert factory.options["api_key"] == "sk-SECRET-123"
        assert "sk-SECRET-123" not in repr(factory.describe())

    def test_options_cannot_be_mutated_through_the_view(self) -> None:
        factory = Middleware(LoggingMiddleware, level=10)

        with pytest.raises(TypeError):
            factory.options["level"] = 20  # type: ignore[index]


class TestComparingTwoAgents:
    def test_independently_built_identical_agents_compare_equal(self) -> None:
        def build() -> Agent:
            return Agent(
                "bot",
                prompt="hello",
                tools=[alpha],
                middleware=[TokenLimiter(max_tokens=10)],
                dependencies={"db": "shared"},
            )

        def fingerprint(agent: Agent) -> object:
            return (
                agent.name,
                agent.system_prompt,
                tuple(t.name for t in agent.tools),
                tuple(m.description for m in agent.middleware),
                tuple(sorted(map(str, agent.dependencies))),
            )

        assert fingerprint(build()) == fingerprint(build())

    def test_a_changed_middleware_setting_is_caught(self) -> None:
        loose = Agent("bot", prompt="p", middleware=[TokenLimiter(max_tokens=10)])
        tight = Agent("bot", prompt="p", middleware=[TokenLimiter(max_tokens=99)])

        assert loose.middleware[0].description != tight.middleware[0].description
