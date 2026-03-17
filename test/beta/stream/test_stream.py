# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from autogen.beta import Context, MemoryStream
from autogen.beta.events import ModelMessage, ToolCall


class TestStreamSend:
    @pytest.mark.asyncio
    async def test_send_event_to_single_subscriber(self, mock: MagicMock) -> None:
        stream = MemoryStream()

        stream.subscribe(lambda ev: mock(ev))
        event = ToolCall(name="func1", arguments="test")
        await stream.send(event, context=Context(stream))

        mock.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_send_event_to_multiple_subscribers(self, mock: MagicMock) -> None:
        stream = MemoryStream()

        stream.subscribe(lambda ev: mock.listener1(ev))
        stream.subscribe(lambda ev: mock.listener2(ev))
        event = ToolCall(name="func1", arguments="test")
        await stream.send(event, context=Context(stream))

        mock.listener1.assert_called_once_with(event)
        mock.listener2.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_send_multiple_events(self, mock: MagicMock) -> None:
        stream = MemoryStream()

        stream.subscribe(mock)
        event1 = ToolCall(name="func1", arguments="test1")
        event2 = ToolCall(name="func2", arguments="test2")
        event3 = ModelMessage(content="response")

        await stream.send(event1, context=Context(stream))
        await stream.send(event2, context=Context(stream))
        await stream.send(event3, context=Context(stream))

        assert [c[0][0] for c in mock.call_args_list] == [event1, event2, event3]


class TestStreamWhereTypeFilter:
    @pytest.mark.asyncio
    async def test_where_type_filter_by_type(self, mock: MagicMock) -> None:
        stream = MemoryStream()

        tool_stream = stream.where(ToolCall)
        tool_stream.subscribe(mock)

        event1 = ToolCall(name="func1", arguments="test1")
        event2 = ModelMessage(content="response")
        event3 = ToolCall(name="func2", arguments="test2")
        await stream.send(event1, context=Context(stream))
        await stream.send(event2, context=Context(stream))
        await stream.send(event3, context=Context(stream))

        assert [c[0][0] for c in mock.call_args_list] == [event1, event3]

    @pytest.mark.asyncio
    async def test_where_type_filter_by_union_type(self, mock: MagicMock) -> None:
        stream = MemoryStream()

        tool_stream = stream.where(ToolCall | ModelMessage)
        tool_stream.subscribe(mock)

        event1 = ToolCall(name="func1", arguments="test1")
        event2 = ModelMessage(content="response")
        await stream.send(event1, context=Context(stream))
        await stream.send(event2, context=Context(stream))

        assert [c[0][0] for c in mock.call_args_list] == [event1, event2]

    @pytest.mark.asyncio
    async def test_where_type_filter_no_match(self, mock: MagicMock) -> None:
        stream = MemoryStream()

        tool_stream = stream.where(ToolCall)
        tool_stream.subscribe(mock)

        await stream.send(ModelMessage(content="response"), context=Context(stream))
        await stream.send(ModelMessage(content="response2"), context=Context(stream))

        mock.assert_not_called()


class TestStreamWhereConditionFilter:
    @pytest.mark.asyncio
    async def test_where_condition_filter_by_condition(self, mock: MagicMock) -> None:
        stream = MemoryStream()

        tool_stream = stream.where(ToolCall)
        func1_stream = tool_stream.where(ToolCall.name == "func1")
        func1_stream.subscribe(mock)

        event1 = ToolCall(name="func1", arguments="test1")
        event3 = ToolCall(name="func1", arguments="test3")
        await stream.send(event1, context=Context(stream))
        await stream.send(ToolCall(name="func2", arguments="test2"), context=Context(stream))
        await stream.send(event3, context=Context(stream))
        await stream.send(ModelMessage(content="response"), context=Context(stream))

        assert [c[0][0] for c in mock.call_args_list] == [event1, event3]

    @pytest.mark.asyncio
    async def test_where_condition_filter_toolcall_name_no_match(self, mock: MagicMock) -> None:
        stream = MemoryStream()

        tool_stream = stream.where(ToolCall)
        func1_stream = tool_stream.where(ToolCall.name == "func1")
        func1_stream.subscribe(mock)

        await stream.send(ToolCall(name="func2", arguments="test1"), context=Context(stream))
        await stream.send(ToolCall(name="func3", arguments="test2"), context=Context(stream))
        await stream.send(ModelMessage(content="response"), context=Context(stream))

        mock.assert_not_called()


class TestStreamChainedFilters:
    @pytest.mark.asyncio
    async def test_chained_type_and_condition_filters(self, mock: MagicMock) -> None:
        stream = MemoryStream()

        stream.subscribe(mock.all)
        tool_stream = stream.where(ToolCall)
        tool_stream.subscribe(mock.tool)
        tool_stream.where(ToolCall.name == "func1").subscribe(mock.func)

        await stream.send(ToolCall(name="func1", arguments="test1"), context=Context(stream))
        await stream.send(ToolCall(name="func2", arguments="test2"), context=Context(stream))
        await stream.send(ModelMessage(content="response"), context=Context(stream))

        assert mock.all.call_count == 3
        assert mock.tool.call_count == 2
        assert mock.func.call_count == 1

    @pytest.mark.asyncio
    async def test_unreachable_filter_scenario(self, mock: MagicMock) -> None:
        stream = MemoryStream()

        stream.where(ToolCall).where(ModelMessage).subscribe(mock)

        await stream.send(ToolCall(name="func1", arguments="test1"), context=Context(stream))
        await stream.send(ModelMessage(content="response"), context=Context(stream))
        await stream.send(ToolCall(name="func2", arguments="test2"), context=Context(stream))

        mock.assert_not_called()


class TestStreamMultipleSubscribers:
    @pytest.mark.asyncio
    async def test_multiple_subscribers_same_stream(self, mock: MagicMock) -> None:
        stream = MemoryStream()

        stream.subscribe(mock.one)
        stream.subscribe(mock.two)

        await stream.send(ToolCall(name="func1", arguments="test"), context=Context(stream))
        await stream.send(ModelMessage(content="response"), context=Context(stream))

        assert mock.one.call_count == 2
        assert mock.two.call_count == 2


class TestStreamPlayScenario:
    @pytest.mark.asyncio
    async def test_play_py_scenario(self) -> None:
        stream = MemoryStream()
        all_listener = MagicMock()
        tool_listener = MagicMock()
        tool_func1_listener = MagicMock()
        model_listener = MagicMock()
        unreachable_listener = MagicMock()

        stream.subscribe(all_listener)

        tool_stream = stream.where(ToolCall)
        tool_stream.subscribe(tool_listener)
        tool_stream.where(ToolCall.name == "func1").subscribe(tool_func1_listener)

        stream.where(ModelMessage).subscribe(model_listener)
        tool_stream.where(ModelMessage).subscribe(unreachable_listener)

        await stream.send(ToolCall(name="func1", arguments="Wtf1"), context=Context(stream))
        await stream.send(ToolCall(name="func2", arguments="Wtf2"), context=Context(stream))
        await stream.send(ModelMessage(content="Test"), context=Context(stream))

        assert all_listener.call_count == 3
        assert tool_listener.call_count == 2
        assert tool_func1_listener.call_count == 1
        assert model_listener.call_count == 1
        unreachable_listener.assert_not_called()

        all_calls = all_listener.call_args_list
        assert all_calls[0][0][0].name == "func1"
        assert all_calls[1][0][0].name == "func2"
        assert all_calls[2][0][0].content == "Test"

        tool_calls = tool_listener.call_args_list
        assert tool_calls[0][0][0].name == "func1"
        assert tool_calls[1][0][0].name == "func2"

        assert tool_func1_listener.call_args[0][0].name == "func1"
        assert model_listener.call_args[0][0].content == "Test"


class TestStreamUnsubscribe:
    @pytest.mark.asyncio
    async def test_unsubscribe_stops_receiving_events(self, mock: MagicMock) -> None:
        stream = MemoryStream()

        sub_id = stream.subscribe(lambda ev: mock(ev))
        event = ToolCall(name="func1", arguments="test1")
        await stream.send(event, context=Context(stream))

        stream.unsubscribe(sub_id)
        await stream.send(ToolCall(name="func2", arguments="test2"), context=Context(stream))

        mock.assert_called_once_with(event)


class TestStreamContextPropagation:
    @pytest.mark.asyncio
    async def test_context_propagates_to_substream(self, mock: MagicMock) -> None:
        stream = MemoryStream()

        def listener(ctx: Context):
            mock(ctx)

        tool_stream = stream.where(ToolCall)
        tool_stream.subscribe(listener)

        custom_ctx = Context(stream)
        await stream.send(ToolCall(name="func1", arguments="test"), custom_ctx)

        mock.assert_called_once_with(custom_ctx)
