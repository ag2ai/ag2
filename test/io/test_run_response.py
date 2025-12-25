# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for runtime_checkable protocols in run_response module."""

from collections.abc import AsyncIterable, Iterable
from typing import Any, Optional
from uuid import UUID, uuid4

from autogen.agentchat.group.context_variables import ContextVariables
from autogen.events.base_event import BaseEvent
from autogen.io.run_response import (
    AsyncRunResponse,
    AsyncRunResponseProtocol,
    Cost,
    Message,
    RunInfoProtocol,
    RunResponse,
    RunResponseProtocol,
)


class TestRunInfoProtocolRuntimeCheckable:
    """Tests for RunInfoProtocol runtime_checkable behavior."""

    def test_isinstance_with_compliant_class(self) -> None:
        """Test that isinstance works with a class implementing RunInfoProtocol."""

        class CompliantRunInfo:
            @property
            def uuid(self) -> UUID:
                return uuid4()

            @property
            def above_run(self) -> Optional["RunResponseProtocol"]:
                return None

        instance = CompliantRunInfo()
        assert isinstance(instance, RunInfoProtocol)

    def test_isinstance_with_non_compliant_class(self) -> None:
        """Test that isinstance returns False for non-compliant classes."""

        class NonCompliant:
            pass

        instance = NonCompliant()
        assert not isinstance(instance, RunInfoProtocol)

    def test_isinstance_with_partial_implementation(self) -> None:
        """Test that isinstance returns False for partial implementations."""

        class PartialImplementation:
            @property
            def uuid(self) -> UUID:
                return uuid4()

            # Missing above_run property

        instance = PartialImplementation()
        assert not isinstance(instance, RunInfoProtocol)


class TestRunResponseProtocolRuntimeCheckable:
    """Tests for RunResponseProtocol runtime_checkable behavior."""

    def test_isinstance_with_compliant_class(self) -> None:
        """Test that isinstance works with a class implementing RunResponseProtocol."""

        class CompliantRunResponse:
            @property
            def uuid(self) -> UUID:
                return uuid4()

            @property
            def above_run(self) -> Optional["RunResponseProtocol"]:
                return None

            @property
            def events(self) -> Iterable[BaseEvent]:
                return []

            @property
            def messages(self) -> Iterable[Message]:
                return []

            @property
            def summary(self) -> str | None:
                return None

            @property
            def context_variables(self) -> ContextVariables | None:
                return None

            @property
            def last_speaker(self) -> str | None:
                return None

            @property
            def cost(self) -> Cost | None:
                return None

            def process(self, processor: Any = None) -> None:
                pass

            def set_ui_tools(self, tools: Any) -> None:
                pass

        instance = CompliantRunResponse()
        assert isinstance(instance, RunResponseProtocol)

    def test_isinstance_with_non_compliant_class(self) -> None:
        """Test that isinstance returns False for non-compliant classes."""

        class NonCompliant:
            pass

        instance = NonCompliant()
        assert not isinstance(instance, RunResponseProtocol)

    def test_isinstance_with_partial_implementation(self) -> None:
        """Test that isinstance returns False for partial implementations."""

        class PartialImplementation:
            @property
            def uuid(self) -> UUID:
                return uuid4()

            @property
            def events(self) -> Iterable[BaseEvent]:
                return []

            # Missing other required properties and methods

        instance = PartialImplementation()
        assert not isinstance(instance, RunResponseProtocol)


class TestAsyncRunResponseProtocolRuntimeCheckable:
    """Tests for AsyncRunResponseProtocol runtime_checkable behavior."""

    def test_isinstance_with_compliant_class(self) -> None:
        """Test that isinstance works with a class implementing AsyncRunResponseProtocol."""

        class CompliantAsyncRunResponse:
            @property
            def uuid(self) -> UUID:
                return uuid4()

            @property
            def above_run(self) -> Optional["RunResponseProtocol"]:
                return None

            @property
            def events(self) -> AsyncIterable[BaseEvent]:
                async def gen() -> AsyncIterable[BaseEvent]:
                    return
                    yield  # type: ignore[misc]  # noqa: RET503 - makes this an async generator

                return gen()

            @property
            async def messages(self) -> Iterable[Message]:
                return []

            @property
            async def summary(self) -> str | None:
                return None

            @property
            async def context_variables(self) -> ContextVariables | None:
                return None

            @property
            async def last_speaker(self) -> str | None:
                return None

            @property
            async def cost(self) -> Cost | None:
                return None

            async def process(self, processor: Any = None) -> None:
                pass

            def set_ui_tools(self, tools: Any) -> None:
                pass

        instance = CompliantAsyncRunResponse()
        assert isinstance(instance, AsyncRunResponseProtocol)

    def test_isinstance_with_non_compliant_class(self) -> None:
        """Test that isinstance returns False for non-compliant classes."""

        class NonCompliant:
            pass

        instance = NonCompliant()
        assert not isinstance(instance, AsyncRunResponseProtocol)


class TestProtocolInheritance:
    """Test that protocol inheritance works correctly with runtime_checkable."""

    def test_run_response_protocol_inherits_run_info(self) -> None:
        """Test that RunResponseProtocol instances are also RunInfoProtocol instances."""

        class CompliantRunResponse:
            @property
            def uuid(self) -> UUID:
                return uuid4()

            @property
            def above_run(self) -> Optional["RunResponseProtocol"]:
                return None

            @property
            def events(self) -> Iterable[BaseEvent]:
                return []

            @property
            def messages(self) -> Iterable[Message]:
                return []

            @property
            def summary(self) -> str | None:
                return None

            @property
            def context_variables(self) -> ContextVariables | None:
                return None

            @property
            def last_speaker(self) -> str | None:
                return None

            @property
            def cost(self) -> Cost | None:
                return None

            def process(self, processor: Any = None) -> None:
                pass

            def set_ui_tools(self, tools: Any) -> None:
                pass

        instance = CompliantRunResponse()
        assert isinstance(instance, RunResponseProtocol)
        assert isinstance(instance, RunInfoProtocol)


class TestConcreteClassesAreProtocolInstances:
    """Test that the concrete RunResponse classes are recognized as protocol instances.

    Note: Protocols with non-method members (properties) don't support issubclass(),
    only isinstance() checks. We use MagicMock to create instances without needing
    the full constructor dependencies.
    """

    def test_run_response_isinstance_run_response_protocol(self) -> None:
        """Test that RunResponse instance is recognized as RunResponseProtocol."""
        from unittest.mock import MagicMock

        # Create a mock iostream and agents to satisfy the constructor
        mock_iostream = MagicMock()
        mock_iostream.input_stream = MagicMock()
        mock_agents: list[Any] = []

        instance = RunResponse(mock_iostream, mock_agents)
        assert isinstance(instance, RunResponseProtocol)

    def test_run_response_isinstance_run_info_protocol(self) -> None:
        """Test that RunResponse instance is recognized as RunInfoProtocol."""
        from unittest.mock import MagicMock

        mock_iostream = MagicMock()
        mock_iostream.input_stream = MagicMock()
        mock_agents: list[Any] = []

        instance = RunResponse(mock_iostream, mock_agents)
        assert isinstance(instance, RunInfoProtocol)

    def test_async_run_response_isinstance_async_run_response_protocol(self) -> None:
        """Test that AsyncRunResponse instance is recognized as AsyncRunResponseProtocol."""
        from unittest.mock import MagicMock

        mock_iostream = MagicMock()
        mock_iostream.input_stream = MagicMock()
        mock_agents: list[Any] = []

        instance = AsyncRunResponse(mock_iostream, mock_agents)
        assert isinstance(instance, AsyncRunResponseProtocol)

    def test_async_run_response_isinstance_run_info_protocol(self) -> None:
        """Test that AsyncRunResponse instance is recognized as RunInfoProtocol."""
        from unittest.mock import MagicMock

        mock_iostream = MagicMock()
        mock_iostream.input_stream = MagicMock()
        mock_agents: list[Any] = []

        instance = AsyncRunResponse(mock_iostream, mock_agents)
        assert isinstance(instance, RunInfoProtocol)
