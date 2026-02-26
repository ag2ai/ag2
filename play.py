import asyncio
from pprint import pprint

from autogen.beta import Agent, Context, Stream
from autogen.beta.events import (
    HITL,
    BaseEvent,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolResult,
    UserMessage,
)
from autogen.beta.llms import LLMClient
from autogen.beta.tools import tool


class MockClient(LLMClient):
    async def __call__(self, *messages: BaseEvent, ctx: Context) -> None:
        print("Model call:")
        pprint(messages)
        last_msg = messages[-1]
        if isinstance(last_msg, ModelRequest):
            await ctx.send(ToolCall(name="func", arguments='{"cmd": "Call me a user\\n"}'))
        elif isinstance(last_msg, ToolResult):
            await ctx.send(ModelResponse(response="generated text"))
        elif isinstance(last_msg, UserMessage):
            await ctx.send(ModelResponse(response="Hi, user!"))


async def hitl_subscriber(event: HITL, ctx: Context) -> None:
    user_message = input(event.message)
    event = UserMessage(content=user_message)
    await ctx.send(event)


stream = Stream()


# @stream.where(ToolCall).subscribe(interrupt=True)
# async def patch_data(event: ToolCall, ctx: Context) -> BaseEvent | None:
#     print("interrupt:", event.arguments)
#     return event


@tool
async def func(cmd: str, ctx: Context) -> str:
    print()
    r = await ctx.input(cmd, timeout=10.0)
    print()
    return r


agent = Agent(
    "test",
    # client=OpenAIClient("gpt-5", reasoning_effort="high"),
    client=MockClient(),
    tools=[func],
)


async def main() -> None:
    with stream.where(HITL).sub_scope(hitl_subscriber):
        conversation = await agent.ask("Hi, agent!", stream=stream)
        print("\nFinal history:")
        final_events = list(await conversation.history.get_events())
        pprint(final_events)
        print("\nResult:", conversation.message, "\n", "=" * 80, "\n")

        result = await conversation.ask("Next turn")
        print("\nResult:", result.message, "\n", "=" * 80, "\n")

        # alternatively
        # result = await agent.ask("Next turn", stream=conversation.stream)
        # print("\nResult:", result.message, "\n", "=" * 80, "\n")

        # restore process from partialhistory
        # await conversation.stream.history.replace(final_events[:-4])
        # pprint(await conversation.stream.history.get_events())
        # result = await agent.restore(stream=stream)
        # print("\nFinal history:")
        # pprint(await stream.history.get_events())
        # print("\nResult", result.message, "\n", "=" * 80, "\n")


asyncio.run(main())
