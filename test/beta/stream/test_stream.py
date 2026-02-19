from unittest.mock import AsyncMock

import pytest

from autogen.beta.events import ModelResponse, ToolCall
from autogen.beta.stream import Context, Stream


class TestStreamSend:
    @pytest.mark.asyncio
    async def test_send_event_to_single_subscriber(self, async_mock: AsyncMock):
        stream = Stream()

        stream.subscribe(lambda ev, _: async_mock(ev))
        event = ToolCall(name="func1", arguments="test")
        await stream.send(event)

        async_mock.assert_awaited_once_with(event)

    @pytest.mark.asyncio
    async def test_send_event_to_multiple_subscribers(self, async_mock: AsyncMock):
        stream = Stream()

        stream.subscribe(lambda ev, _: async_mock.listener1(ev))
        stream.subscribe(lambda ev, _: async_mock.listener2(ev))
        event = ToolCall(name="func1", arguments="test")
        await stream.send(event)

        async_mock.listener1.assert_awaited_once_with(event)
        async_mock.listener2.assert_awaited_once_with(event)

    @pytest.mark.asyncio
    async def test_send_multiple_events(self, async_mock: AsyncMock):
        stream = Stream()

        stream.subscribe(async_mock)
        event1 = ToolCall(name="func1", arguments="test1")
        event2 = ToolCall(name="func2", arguments="test2")
        event3 = ModelResponse(response="response")

        await stream.send(event1)
        await stream.send(event2)
        await stream.send(event3)

        assert [c[0][0] for c in async_mock.await_args_list] == [event1, event2, event3]


class TestStreamWhereTypeFilter:
    @pytest.mark.asyncio
    async def test_where_type_filter_by_type(self, async_mock: AsyncMock):
        stream = Stream()

        tool_stream = stream.where(ToolCall)
        tool_stream.subscribe(async_mock)

        event1 = ToolCall(name="func1", arguments="test1")
        event2 = ModelResponse(response="response")
        event3 = ToolCall(name="func2", arguments="test2")
        await stream.send(event1)
        await stream.send(event2)
        await stream.send(event3)

        assert [c[0][0] for c in async_mock.await_args_list] == [event1, event3]

    @pytest.mark.asyncio
    async def test_where_type_filter_by_union_type(self, async_mock: AsyncMock):
        stream = Stream()

        tool_stream = stream.where(ToolCall | ModelResponse)
        tool_stream.subscribe(async_mock)

        event1 = ToolCall(name="func1", arguments="test1")
        event2 = ModelResponse(response="response")
        await stream.send(event1)
        await stream.send(event2)

        assert [c[0][0] for c in async_mock.await_args_list] == [event1, event2]

    @pytest.mark.asyncio
    async def test_where_type_filter_no_match(self, async_mock: AsyncMock):
        stream = Stream()

        tool_stream = stream.where(ToolCall)
        tool_stream.subscribe(async_mock)

        await stream.send(ModelResponse(response="response"))
        await stream.send(ModelResponse(response="response2"))

        async_mock.assert_not_called()


class TestStreamWhereConditionFilter:
    @pytest.mark.asyncio
    async def test_where_condition_filter_by_condition(self, async_mock: AsyncMock):
        stream = Stream()

        tool_stream = stream.where(ToolCall)
        func1_stream = tool_stream.where(ToolCall.name == "func1")
        func1_stream.subscribe(async_mock)

        event1 = ToolCall(name="func1", arguments="test1")
        event3 = ToolCall(name="func1", arguments="test3")
        await stream.send(event1)
        await stream.send(ToolCall(name="func2", arguments="test2"))
        await stream.send(event3)
        await stream.send(ModelResponse(response="response"))

        assert [c[0][0] for c in async_mock.await_args_list] == [event1, event3]

    @pytest.mark.asyncio
    async def test_where_condition_filter_toolcall_name_no_match(self, async_mock: AsyncMock):
        stream = Stream()

        tool_stream = stream.where(ToolCall)
        func1_stream = tool_stream.where(ToolCall.name == "func1")
        func1_stream.subscribe(async_mock)

        await stream.send(ToolCall(name="func2", arguments="test1"))
        await stream.send(ToolCall(name="func3", arguments="test2"))
        await stream.send(ModelResponse(response="response"))

        async_mock.assert_not_called()


class TestStreamChainedFilters:
    @pytest.mark.asyncio
    async def test_chained_type_and_condition_filters(self, async_mock: AsyncMock):
        stream = Stream()

        stream.subscribe(async_mock.all)
        tool_stream = stream.where(ToolCall)
        tool_stream.subscribe(async_mock.tool)
        tool_stream.where(ToolCall.name == "func1").subscribe(async_mock.func)

        await stream.send(ToolCall(name="func1", arguments="test1"))
        await stream.send(ToolCall(name="func2", arguments="test2"))
        await stream.send(ModelResponse(response="response"))

        assert async_mock.all.call_count == 3
        assert async_mock.tool.call_count == 2
        assert async_mock.func.call_count == 1

    @pytest.mark.asyncio
    async def test_unreachable_filter_scenario(self, async_mock: AsyncMock):
        stream = Stream()

        stream.where(ToolCall).where(ModelResponse).subscribe(async_mock)

        await stream.send(ToolCall(name="func1", arguments="test1"))
        await stream.send(ModelResponse(response="response"))
        await stream.send(ToolCall(name="func2", arguments="test2"))

        async_mock.assert_not_called()


class TestStreamMultipleSubscribers:
    @pytest.mark.asyncio
    async def test_multiple_subscribers_same_stream(self, async_mock: AsyncMock):
        stream = Stream()

        stream.subscribe(async_mock.one)
        stream.subscribe(async_mock.two)

        await stream.send(ToolCall(name="func1", arguments="test"))
        await stream.send(ModelResponse(response="response"))

        assert async_mock.one.call_count == 2
        assert async_mock.two.call_count == 2


class TestStreamPlayScenario:
    @pytest.mark.asyncio
    async def test_play_py_scenario(self):
        stream = Stream()
        all_listener = AsyncMock()
        tool_listener = AsyncMock()
        tool_func1_listener = AsyncMock()
        model_listener = AsyncMock()
        unreachable_listener = AsyncMock()

        stream.subscribe(all_listener)

        tool_stream = stream.where(ToolCall)
        tool_stream.subscribe(tool_listener)
        tool_stream.where(ToolCall.name == "func1").subscribe(tool_func1_listener)

        stream.where(ModelResponse).subscribe(model_listener)
        tool_stream.where(ModelResponse).subscribe(unreachable_listener)

        await stream.send(ToolCall(name="func1", arguments="Wtf1"))
        await stream.send(ToolCall(name="func2", arguments="Wtf2"))
        await stream.send(ModelResponse(response="Test"))

        assert all_listener.call_count == 3
        assert tool_listener.call_count == 2
        assert tool_func1_listener.call_count == 1
        assert model_listener.call_count == 1
        unreachable_listener.assert_not_called()

        all_calls = all_listener.call_args_list
        assert all_calls[0][0][0].name == "func1"
        assert all_calls[1][0][0].name == "func2"
        assert all_calls[2][0][0].response == "Test"

        tool_calls = tool_listener.call_args_list
        assert tool_calls[0][0][0].name == "func1"
        assert tool_calls[1][0][0].name == "func2"

        assert tool_func1_listener.call_args[0][0].name == "func1"
        assert model_listener.call_args[0][0].response == "Test"


class TestStreamUnsubscribe:
    @pytest.mark.asyncio
    async def test_unsubscribe_stops_receiving_events(self, async_mock: AsyncMock):
        stream = Stream()

        sub_id = stream.subscribe(lambda ev, _: async_mock(ev))
        event = ToolCall(name="func1", arguments="test1")
        await stream.send(event)

        stream.unsubscribe(sub_id)
        await stream.send(ToolCall(name="func2", arguments="test2"))

        async_mock.assert_awaited_once_with(event)


class TestStreamContextPropagation:
    @pytest.mark.asyncio
    async def test_context_propagates_to_substream(self):
        stream = Stream()
        listener = AsyncMock()

        tool_stream = stream.where(ToolCall)
        tool_stream.subscribe(listener)

        custom_ctx = Context(stream)
        await stream.send(ToolCall(name="func1", arguments="test"), custom_ctx)

        listener.assert_called_once()
        assert listener.call_args[0][1] is custom_ctx
