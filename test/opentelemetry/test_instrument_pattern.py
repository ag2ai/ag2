# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for autogen.opentelemetry.instrumentators.pattern module (instrument_pattern, instrument_groupchat)."""

import contextlib
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from autogen.opentelemetry import instrument_pattern
from autogen.opentelemetry.consts import SpanType
from autogen.opentelemetry.instrumentators.pattern import instrument_groupchat
from test.opentelemetry.conftest import InMemorySpanExporter


@pytest.fixture()
def otel_setup():
    """Create an in-memory OTEL exporter/provider for capturing spans."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter, provider


def _make_mock_agent(agent_name: str) -> MagicMock:
    """Create a mock agent with a proper .name attribute."""
    agent = MagicMock()
    agent.name = agent_name
    return agent


def _make_mock_groupchat(agents=None):
    """Create a mock GroupChat with the methods that instrument_groupchat wraps."""
    gc = MagicMock()
    gc.agents = agents or []
    # Ensure the methods do NOT already have __otel_wrapped__
    gc._create_internal_agents = MagicMock()
    gc._auto_select_speaker = MagicMock()
    gc.a_auto_select_speaker = AsyncMock()
    # Remove the __otel_wrapped__ attribute that MagicMock auto-creates via hasattr
    del gc._create_internal_agents.__otel_wrapped__
    del gc._auto_select_speaker.__otel_wrapped__
    del gc.a_auto_select_speaker.__otel_wrapped__
    return gc


def _make_mock_pattern(prepare_return_value=None):
    """Create a mock Pattern with a prepare_group_chat method."""
    pattern = MagicMock()
    pattern.prepare_group_chat = MagicMock(return_value=prepare_return_value)
    # Remove __otel_wrapped__ so the guard in instrument_pattern sees it as unwrapped
    del pattern.prepare_group_chat.__otel_wrapped__
    return pattern


# ---------------------------------------------------------------------------
# instrument_pattern: basic instrumentation
# ---------------------------------------------------------------------------
class TestInstrumentPatternBasic:
    """Basic tests for instrument_pattern."""

    def test_returns_same_pattern_object(self, otel_setup) -> None:
        _exporter, provider = otel_setup
        pattern = _make_mock_pattern()
        result = instrument_pattern(pattern, tracer_provider=provider)
        assert result is pattern

    def test_marks_prepare_group_chat_as_wrapped(self, otel_setup) -> None:
        _exporter, provider = otel_setup
        pattern = _make_mock_pattern()
        instrument_pattern(pattern, tracer_provider=provider)
        assert hasattr(pattern.prepare_group_chat, "__otel_wrapped__")
        assert pattern.prepare_group_chat.__otel_wrapped__ is True


# ---------------------------------------------------------------------------
# instrument_pattern: idempotency
# ---------------------------------------------------------------------------
class TestInstrumentPatternIdempotency:
    """Tests that double-instrumenting does not double-wrap."""

    def test_double_instrument_does_not_double_wrap(self, otel_setup) -> None:
        _exporter, provider = otel_setup
        pattern = _make_mock_pattern()
        instrument_pattern(pattern, tracer_provider=provider)
        first_prepare = pattern.prepare_group_chat
        instrument_pattern(pattern, tracer_provider=provider)
        assert pattern.prepare_group_chat is first_prepare

    def test_double_instrument_returns_same_pattern(self, otel_setup) -> None:
        _exporter, provider = otel_setup
        pattern = _make_mock_pattern()
        result1 = instrument_pattern(pattern, tracer_provider=provider)
        result2 = instrument_pattern(pattern, tracer_provider=provider)
        assert result1 is result2


# ---------------------------------------------------------------------------
# instrument_pattern: prepare_group_chat traced behaviour
# ---------------------------------------------------------------------------
class TestInstrumentPatternPrepareGroupChat:
    """Tests that the wrapped prepare_group_chat instruments agents and groupchat."""

    def test_wrapped_prepare_group_chat_calls_original(self, otel_setup) -> None:
        _exporter, provider = otel_setup

        agent1 = _make_mock_agent("agent1")
        agent2 = _make_mock_agent("agent2")

        mock_gc = _make_mock_groupchat(agents=[agent1, agent2])
        # _create_internal_agents and speaker selection don't have __otel_wrapped__
        # already handled by _make_mock_groupchat

        mock_manager = MagicMock()
        mock_manager.name = "manager"
        mock_manager._reply_func_list = []
        # Remove __otel_wrapped__ from methods that instrument_agent would check
        for attr_name in [
            "initiate_chat",
            "generate_reply",
            "a_generate_reply",
            "execute_function",
            "get_human_input",
            "a_initiate_chat",
        ]:
            if hasattr(getattr(mock_manager, attr_name, None), "__otel_wrapped__"):
                with contextlib.suppress(AttributeError, TypeError):
                    del getattr(mock_manager, attr_name).__otel_wrapped__

        prepare_return = (
            [agent1, agent2],  # agents
            [agent1, agent2],  # wrapped_agents
            None,  # user_agent
            MagicMock(),  # context_variables
            agent1,  # initial_agent
            MagicMock(),  # group_after_work
            MagicMock(),  # tool_executor
            mock_gc,  # groupchat
            mock_manager,  # manager
            [],  # processed_messages
            agent1,  # last_agent
            ["agent1", "agent2"],  # group_agent_names
            [],  # temp_user_list
        )

        pattern = _make_mock_pattern(prepare_return_value=prepare_return)
        instrument_pattern(pattern, tracer_provider=provider)

        # Call the wrapped function
        with (
            patch("autogen.opentelemetry.instrumentators.pattern.instrument_agent") as mock_inst_agent,
            patch("autogen.opentelemetry.instrumentators.pattern.instrument_groupchat") as mock_inst_gc,
        ):
            mock_inst_agent.side_effect = lambda agent, **kw: agent
            mock_inst_gc.side_effect = lambda gc, **kw: gc

            result = pattern.prepare_group_chat(10)

        # The original prepare_group_chat should have been called
        assert pattern.prepare_group_chat is not prepare_return  # It's a wrapper now
        assert result is not None
        assert len(result) == 13  # 13-tuple

    def test_wrapped_prepare_group_chat_instruments_groupchat_agents(self, otel_setup) -> None:
        _exporter, provider = otel_setup

        agent1 = _make_mock_agent("agent1")
        agent2 = _make_mock_agent("agent2")

        mock_gc = _make_mock_groupchat(agents=[agent1, agent2])
        mock_manager = MagicMock()
        mock_manager.name = "manager"
        mock_manager._reply_func_list = []

        prepare_return = (
            [agent1, agent2],
            [agent1, agent2],
            None,
            MagicMock(),
            agent1,
            MagicMock(),
            MagicMock(),
            mock_gc,
            mock_manager,
            [],
            agent1,
            ["agent1", "agent2"],
            [],
        )

        pattern = _make_mock_pattern(prepare_return_value=prepare_return)
        instrument_pattern(pattern, tracer_provider=provider)

        with (
            patch("autogen.opentelemetry.instrumentators.pattern.instrument_agent") as mock_inst_agent,
            patch("autogen.opentelemetry.instrumentators.pattern.instrument_groupchat") as mock_inst_gc,
        ):
            mock_inst_agent.side_effect = lambda agent, **kw: agent
            mock_inst_gc.side_effect = lambda gc, **kw: gc

            pattern.prepare_group_chat(10)

        # instrument_agent should be called for each agent in groupchat.agents + manager
        # That's 2 agents + 1 manager = 3 calls
        assert mock_inst_agent.call_count == 3

    def test_wrapped_prepare_group_chat_instruments_manager_groupchat_copies(self, otel_setup) -> None:
        """When manager._reply_func_list contains a GroupChat copy, it gets instrumented too."""
        _exporter, provider = otel_setup

        from autogen.agentchat.groupchat import GroupChat

        agent1 = _make_mock_agent("agent1")
        mock_gc = _make_mock_groupchat(agents=[agent1])

        # Create a GroupChat-spec'd mock so isinstance(gc_copy, GroupChat) works
        gc_copy = MagicMock(spec=GroupChat)
        gc_copy.agents = [agent1]
        gc_copy._create_internal_agents = MagicMock()
        gc_copy._auto_select_speaker = MagicMock()
        gc_copy.a_auto_select_speaker = AsyncMock()

        mock_manager = MagicMock()
        mock_manager.name = "manager"
        mock_manager._reply_func_list = [
            {"config": gc_copy},  # This is a GroupChat copy, different object
        ]

        prepare_return = (
            [agent1],
            [agent1],
            None,
            MagicMock(),
            agent1,
            MagicMock(),
            MagicMock(),
            mock_gc,
            mock_manager,
            [],
            agent1,
            ["agent1"],
            [],
        )

        pattern = _make_mock_pattern(prepare_return_value=prepare_return)
        instrument_pattern(pattern, tracer_provider=provider)

        with (
            patch("autogen.opentelemetry.instrumentators.pattern.instrument_agent") as mock_inst_agent,
            patch("autogen.opentelemetry.instrumentators.pattern.instrument_groupchat") as mock_inst_gc,
        ):
            mock_inst_agent.side_effect = lambda agent, **kw: agent
            mock_inst_gc.side_effect = lambda gc, **kw: gc

            pattern.prepare_group_chat(10)

        # instrument_groupchat should be called for the main gc and also for the copy
        assert mock_inst_gc.call_count == 2


# ---------------------------------------------------------------------------
# instrument_groupchat: basic instrumentation
# ---------------------------------------------------------------------------
class TestInstrumentGroupchatBasic:
    """Basic tests for instrument_groupchat."""

    def test_returns_same_groupchat(self, otel_setup) -> None:
        _exporter, provider = otel_setup
        gc = _make_mock_groupchat()
        result = instrument_groupchat(gc, tracer_provider=provider)
        assert result is gc

    def test_wraps_create_internal_agents(self, otel_setup) -> None:
        _exporter, provider = otel_setup
        gc = _make_mock_groupchat()
        instrument_groupchat(gc, tracer_provider=provider)
        assert hasattr(gc._create_internal_agents, "__otel_wrapped__")
        assert gc._create_internal_agents.__otel_wrapped__ is True

    def test_wraps_auto_select_speaker(self, otel_setup) -> None:
        _exporter, provider = otel_setup
        gc = _make_mock_groupchat()
        instrument_groupchat(gc, tracer_provider=provider)
        assert hasattr(gc._auto_select_speaker, "__otel_wrapped__")
        assert gc._auto_select_speaker.__otel_wrapped__ is True

    def test_wraps_a_auto_select_speaker(self, otel_setup) -> None:
        _exporter, provider = otel_setup
        gc = _make_mock_groupchat()
        instrument_groupchat(gc, tracer_provider=provider)
        assert hasattr(gc.a_auto_select_speaker, "__otel_wrapped__")
        assert gc.a_auto_select_speaker.__otel_wrapped__ is True


# ---------------------------------------------------------------------------
# instrument_groupchat: idempotency
# ---------------------------------------------------------------------------
class TestInstrumentGroupchatIdempotency:
    """Tests that double-instrumenting groupchat does not double-wrap."""

    def test_double_instrument_does_not_double_wrap_create_internal_agents(self, otel_setup) -> None:
        _exporter, provider = otel_setup
        gc = _make_mock_groupchat()
        instrument_groupchat(gc, tracer_provider=provider)
        first_fn = gc._create_internal_agents
        instrument_groupchat(gc, tracer_provider=provider)
        assert gc._create_internal_agents is first_fn

    def test_double_instrument_does_not_double_wrap_auto_select_speaker(self, otel_setup) -> None:
        _exporter, provider = otel_setup
        gc = _make_mock_groupchat()
        instrument_groupchat(gc, tracer_provider=provider)
        first_fn = gc._auto_select_speaker
        instrument_groupchat(gc, tracer_provider=provider)
        assert gc._auto_select_speaker is first_fn

    def test_double_instrument_does_not_double_wrap_a_auto_select_speaker(self, otel_setup) -> None:
        _exporter, provider = otel_setup
        gc = _make_mock_groupchat()
        instrument_groupchat(gc, tracer_provider=provider)
        first_fn = gc.a_auto_select_speaker
        instrument_groupchat(gc, tracer_provider=provider)
        assert gc.a_auto_select_speaker is first_fn


# ---------------------------------------------------------------------------
# instrument_groupchat: _create_internal_agents traced
# ---------------------------------------------------------------------------
class TestCreateInternalAgentsTraced:
    """Tests that the wrapped _create_internal_agents instruments temporary agents."""

    def test_create_internal_agents_calls_original_and_returns_agents(self, otel_setup) -> None:
        _exporter, provider = otel_setup

        checking_agent = _make_mock_agent("checking_agent")
        speaker_selection_agent = _make_mock_agent("speaker_selection_agent")

        gc = _make_mock_groupchat()
        gc._create_internal_agents.return_value = (checking_agent, speaker_selection_agent)

        instrument_groupchat(gc, tracer_provider=provider)

        with patch("autogen.opentelemetry.instrumentators.pattern.instrument_agent") as mock_inst_agent:
            mock_inst_agent.side_effect = lambda agent, **kw: agent

            result = gc._create_internal_agents(
                agents=[],
                max_attempts=3,
                messages=[],
                validate_speaker_name=MagicMock(),
            )

        assert result == (checking_agent, speaker_selection_agent)
        # Both temporary agents should be instrumented
        assert mock_inst_agent.call_count == 2

    def test_create_internal_agents_passes_selector_kwarg(self, otel_setup) -> None:
        _exporter, provider = otel_setup

        checking_agent = _make_mock_agent("checking_agent")
        speaker_agent = _make_mock_agent("speaker_agent")

        original_fn = MagicMock(return_value=(checking_agent, speaker_agent))

        gc = MagicMock()
        gc.agents = []
        gc._create_internal_agents = original_fn
        gc._auto_select_speaker = MagicMock()
        gc.a_auto_select_speaker = AsyncMock()
        # Ensure none of them have __otel_wrapped__
        for attr_name in ["_create_internal_agents", "_auto_select_speaker", "a_auto_select_speaker"]:
            mock_attr = getattr(gc, attr_name)
            if hasattr(mock_attr, "__otel_wrapped__"):
                with contextlib.suppress(AttributeError):
                    del mock_attr.__otel_wrapped__

        instrument_groupchat(gc, tracer_provider=provider)

        mock_selector = _make_mock_agent("selector")
        mock_validator = MagicMock()

        with patch("autogen.opentelemetry.instrumentators.pattern.instrument_agent") as mock_inst_agent:
            mock_inst_agent.side_effect = lambda agent, **kw: agent

            gc._create_internal_agents(
                agents=[],
                max_attempts=3,
                messages=[],
                validate_speaker_name=mock_validator,
                selector=mock_selector,
            )

        # Verify original was called with selector
        original_fn.assert_called_once()
        call_args = original_fn.call_args
        assert call_args[1].get("selector") is mock_selector or call_args[0][-1] is mock_selector


# ---------------------------------------------------------------------------
# Speaker selection spans (_auto_select_speaker sync)
# ---------------------------------------------------------------------------
class TestSyncAutoSelectSpeaker:
    """Tests that _auto_select_speaker creates proper speaker_selection spans."""

    def test_creates_speaker_selection_span(self, otel_setup) -> None:
        exporter, provider = otel_setup

        agent1 = _make_mock_agent("researcher")
        agent2 = _make_mock_agent("writer")
        selected = agent1

        gc = _make_mock_groupchat(agents=[agent1, agent2])

        # Set the original to return the selected agent
        original_fn = MagicMock(return_value=selected)
        gc._create_internal_agents = MagicMock()
        del gc._create_internal_agents.__otel_wrapped__
        gc._auto_select_speaker = original_fn
        del gc._auto_select_speaker.__otel_wrapped__
        gc.a_auto_select_speaker = AsyncMock()
        del gc.a_auto_select_speaker.__otel_wrapped__

        instrument_groupchat(gc, tracer_provider=provider)

        last_speaker = _make_mock_agent("last_speaker")
        selector = _make_mock_agent("selector")

        gc._auto_select_speaker(
            last_speaker=last_speaker,
            selector=selector,
            messages=None,
            agents=[agent1, agent2],
        )

        spans = exporter.get_finished_spans()
        speaker_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.SPEAKER_SELECTION.value]
        assert len(speaker_spans) == 1

    def test_span_name_is_speaker_selection(self, otel_setup) -> None:
        exporter, provider = otel_setup

        agent1 = _make_mock_agent("agent1")
        selected = agent1

        gc = _make_mock_groupchat(agents=[agent1])
        original_fn = MagicMock(return_value=selected)
        gc._auto_select_speaker = original_fn
        del gc._auto_select_speaker.__otel_wrapped__

        instrument_groupchat(gc, tracer_provider=provider)

        gc._auto_select_speaker(
            last_speaker=_make_mock_agent("last"),
            selector=_make_mock_agent("sel"),
            messages=None,
            agents=[agent1],
        )

        spans = exporter.get_finished_spans()
        speaker_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.SPEAKER_SELECTION.value]
        assert speaker_spans[0].name == "speaker_selection"

    def test_span_sets_operation_name(self, otel_setup) -> None:
        exporter, provider = otel_setup

        agent1 = _make_mock_agent("agent1")
        gc = _make_mock_groupchat(agents=[agent1])
        gc._auto_select_speaker = MagicMock(return_value=agent1)
        del gc._auto_select_speaker.__otel_wrapped__

        instrument_groupchat(gc, tracer_provider=provider)

        gc._auto_select_speaker(
            last_speaker=_make_mock_agent("last"),
            selector=_make_mock_agent("sel"),
            messages=None,
            agents=[agent1],
        )

        spans = exporter.get_finished_spans()
        speaker_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.SPEAKER_SELECTION.value]
        assert speaker_spans[0].attributes["gen_ai.operation.name"] == "speaker_selection"

    def test_span_records_candidate_agents(self, otel_setup) -> None:
        exporter, provider = otel_setup

        agent1 = _make_mock_agent("researcher")
        agent2 = _make_mock_agent("writer")
        agent3 = _make_mock_agent("critic")

        gc = _make_mock_groupchat(agents=[agent1, agent2, agent3])
        gc._auto_select_speaker = MagicMock(return_value=agent2)
        del gc._auto_select_speaker.__otel_wrapped__

        instrument_groupchat(gc, tracer_provider=provider)

        gc._auto_select_speaker(
            last_speaker=_make_mock_agent("last"),
            selector=_make_mock_agent("sel"),
            messages=None,
            agents=[agent1, agent2, agent3],
        )

        spans = exporter.get_finished_spans()
        speaker_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.SPEAKER_SELECTION.value]
        candidates = json.loads(speaker_spans[0].attributes["ag2.speaker_selection.candidates"])
        assert candidates == ["researcher", "writer", "critic"]

    def test_span_records_selected_speaker(self, otel_setup) -> None:
        exporter, provider = otel_setup

        agent1 = _make_mock_agent("researcher")
        agent2 = _make_mock_agent("writer")

        gc = _make_mock_groupchat(agents=[agent1, agent2])
        gc._auto_select_speaker = MagicMock(return_value=agent2)
        del gc._auto_select_speaker.__otel_wrapped__

        instrument_groupchat(gc, tracer_provider=provider)

        gc._auto_select_speaker(
            last_speaker=_make_mock_agent("last"),
            selector=_make_mock_agent("sel"),
            messages=None,
            agents=[agent1, agent2],
        )

        spans = exporter.get_finished_spans()
        speaker_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.SPEAKER_SELECTION.value]
        assert speaker_spans[0].attributes["ag2.speaker_selection.selected"] == "writer"

    def test_span_uses_groupchat_agents_when_agents_arg_is_none(self, otel_setup) -> None:
        """When agents argument is None, candidates should come from groupchat.agents."""
        exporter, provider = otel_setup

        agent1 = _make_mock_agent("alpha")
        agent2 = _make_mock_agent("beta")

        gc = _make_mock_groupchat(agents=[agent1, agent2])
        gc._auto_select_speaker = MagicMock(return_value=agent1)
        del gc._auto_select_speaker.__otel_wrapped__

        instrument_groupchat(gc, tracer_provider=provider)

        gc._auto_select_speaker(
            last_speaker=_make_mock_agent("last"),
            selector=_make_mock_agent("sel"),
            messages=None,
            agents=None,
        )

        spans = exporter.get_finished_spans()
        speaker_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.SPEAKER_SELECTION.value]
        candidates = json.loads(speaker_spans[0].attributes["ag2.speaker_selection.candidates"])
        assert candidates == ["alpha", "beta"]

    def test_calls_original_auto_select_speaker(self, otel_setup) -> None:
        _exporter, provider = otel_setup

        agent1 = _make_mock_agent("agent1")
        original_fn = MagicMock(return_value=agent1)

        gc = _make_mock_groupchat(agents=[agent1])
        gc._auto_select_speaker = original_fn
        del gc._auto_select_speaker.__otel_wrapped__

        instrument_groupchat(gc, tracer_provider=provider)

        last_speaker = _make_mock_agent("last")
        selector = _make_mock_agent("sel")
        messages = [{"role": "user", "content": "hello"}]

        result = gc._auto_select_speaker(
            last_speaker=last_speaker,
            selector=selector,
            messages=messages,
            agents=[agent1],
        )

        assert result is agent1
        original_fn.assert_called_once_with(last_speaker, selector, messages, [agent1])


# ---------------------------------------------------------------------------
# Speaker selection spans (a_auto_select_speaker async)
# ---------------------------------------------------------------------------
class TestAsyncAutoSelectSpeaker:
    """Tests that a_auto_select_speaker creates proper speaker_selection spans."""

    @pytest.mark.asyncio
    async def test_creates_speaker_selection_span(self, otel_setup) -> None:
        exporter, provider = otel_setup

        agent1 = _make_mock_agent("researcher")
        agent2 = _make_mock_agent("writer")

        gc = _make_mock_groupchat(agents=[agent1, agent2])
        gc.a_auto_select_speaker = AsyncMock(return_value=agent1)
        del gc.a_auto_select_speaker.__otel_wrapped__

        instrument_groupchat(gc, tracer_provider=provider)

        await gc.a_auto_select_speaker(
            last_speaker=_make_mock_agent("last"),
            selector=_make_mock_agent("sel"),
            messages=None,
            agents=[agent1, agent2],
        )

        spans = exporter.get_finished_spans()
        speaker_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.SPEAKER_SELECTION.value]
        assert len(speaker_spans) == 1

    @pytest.mark.asyncio
    async def test_span_records_candidate_agents(self, otel_setup) -> None:
        exporter, provider = otel_setup

        agent1 = _make_mock_agent("alice")
        agent2 = _make_mock_agent("bob")

        gc = _make_mock_groupchat(agents=[agent1, agent2])
        gc.a_auto_select_speaker = AsyncMock(return_value=agent1)
        del gc.a_auto_select_speaker.__otel_wrapped__

        instrument_groupchat(gc, tracer_provider=provider)

        await gc.a_auto_select_speaker(
            last_speaker=_make_mock_agent("last"),
            selector=_make_mock_agent("sel"),
            messages=None,
            agents=[agent1, agent2],
        )

        spans = exporter.get_finished_spans()
        speaker_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.SPEAKER_SELECTION.value]
        candidates = json.loads(speaker_spans[0].attributes["ag2.speaker_selection.candidates"])
        assert candidates == ["alice", "bob"]

    @pytest.mark.asyncio
    async def test_span_records_selected_speaker(self, otel_setup) -> None:
        exporter, provider = otel_setup

        agent1 = _make_mock_agent("alice")
        agent2 = _make_mock_agent("bob")

        gc = _make_mock_groupchat(agents=[agent1, agent2])
        gc.a_auto_select_speaker = AsyncMock(return_value=agent2)
        del gc.a_auto_select_speaker.__otel_wrapped__

        instrument_groupchat(gc, tracer_provider=provider)

        await gc.a_auto_select_speaker(
            last_speaker=_make_mock_agent("last"),
            selector=_make_mock_agent("sel"),
            messages=None,
            agents=[agent1, agent2],
        )

        spans = exporter.get_finished_spans()
        speaker_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.SPEAKER_SELECTION.value]
        assert speaker_spans[0].attributes["ag2.speaker_selection.selected"] == "bob"

    @pytest.mark.asyncio
    async def test_span_uses_groupchat_agents_when_agents_arg_is_none(self, otel_setup) -> None:
        exporter, provider = otel_setup

        agent1 = _make_mock_agent("gamma")
        agent2 = _make_mock_agent("delta")

        gc = _make_mock_groupchat(agents=[agent1, agent2])
        gc.a_auto_select_speaker = AsyncMock(return_value=agent1)
        del gc.a_auto_select_speaker.__otel_wrapped__

        instrument_groupchat(gc, tracer_provider=provider)

        await gc.a_auto_select_speaker(
            last_speaker=_make_mock_agent("last"),
            selector=_make_mock_agent("sel"),
            messages=None,
            agents=None,
        )

        spans = exporter.get_finished_spans()
        speaker_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.SPEAKER_SELECTION.value]
        candidates = json.loads(speaker_spans[0].attributes["ag2.speaker_selection.candidates"])
        assert candidates == ["gamma", "delta"]

    @pytest.mark.asyncio
    async def test_calls_original_a_auto_select_speaker(self, otel_setup) -> None:
        _exporter, provider = otel_setup

        agent1 = _make_mock_agent("agent1")
        original_fn = AsyncMock(return_value=agent1)

        gc = _make_mock_groupchat(agents=[agent1])
        gc.a_auto_select_speaker = original_fn
        del gc.a_auto_select_speaker.__otel_wrapped__

        instrument_groupchat(gc, tracer_provider=provider)

        last_speaker = _make_mock_agent("last")
        selector = _make_mock_agent("sel")

        result = await gc.a_auto_select_speaker(
            last_speaker=last_speaker,
            selector=selector,
            messages=None,
            agents=[agent1],
        )

        assert result is agent1
        original_fn.assert_called_once_with(last_speaker, selector, None, [agent1])

    @pytest.mark.asyncio
    async def test_async_span_name_is_speaker_selection(self, otel_setup) -> None:
        exporter, provider = otel_setup

        agent1 = _make_mock_agent("agent1")

        gc = _make_mock_groupchat(agents=[agent1])
        gc.a_auto_select_speaker = AsyncMock(return_value=agent1)
        del gc.a_auto_select_speaker.__otel_wrapped__

        instrument_groupchat(gc, tracer_provider=provider)

        await gc.a_auto_select_speaker(
            last_speaker=_make_mock_agent("last"),
            selector=_make_mock_agent("sel"),
            messages=None,
            agents=[agent1],
        )

        spans = exporter.get_finished_spans()
        speaker_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.SPEAKER_SELECTION.value]
        assert speaker_spans[0].name == "speaker_selection"

    @pytest.mark.asyncio
    async def test_async_span_sets_operation_name(self, otel_setup) -> None:
        exporter, provider = otel_setup

        agent1 = _make_mock_agent("agent1")

        gc = _make_mock_groupchat(agents=[agent1])
        gc.a_auto_select_speaker = AsyncMock(return_value=agent1)
        del gc.a_auto_select_speaker.__otel_wrapped__

        instrument_groupchat(gc, tracer_provider=provider)

        await gc.a_auto_select_speaker(
            last_speaker=_make_mock_agent("last"),
            selector=_make_mock_agent("sel"),
            messages=None,
            agents=[agent1],
        )

        spans = exporter.get_finished_spans()
        speaker_spans = [s for s in spans if s.attributes.get("ag2.span.type") == SpanType.SPEAKER_SELECTION.value]
        assert speaker_spans[0].attributes["gen_ai.operation.name"] == "speaker_selection"
