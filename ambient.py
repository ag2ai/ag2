import logging

logger = logging.getLogger("autogen")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

from typing import Annotated

from autogen.beta import Agent, Context, Inject, MemoryStream
from autogen.beta.config import OpenAIConfig
from autogen.beta.events import BaseEvent, ModelResponse
from autogen.beta.middleware import LoggingMiddleware

agent = Agent(
    "test-agent",
    prompt="You are a helpful agent!",
    config=OpenAIConfig(
        "gpt-5-nano",
        reasoning_effort="low",
        streaming=True,
    ),
    middleware=[LoggingMiddleware()],
)


@agent.prompt
async def get_prompt(event: BaseEvent, ctx: Context) -> str:
    print(event)
    return "Do your best to be helpful!"


# ambient runtime
stream = MemoryStream()


class CallModel(BaseEvent):
    prompt: str


@stream.where(CallModel).subscribe()
async def call_agent(
    event: CallModel,
    ctx: Context,
    agent: Annotated[Agent, Inject("agent")],
) -> None:
    await agent.ask(
        event.prompt,
        stream=ctx.stream,
        dependencies=ctx.dependencies,
        variables=ctx.variables,
    )

    # # get original history events
    # history_events = await ctx.stream.history.get_events()

    # # create new stream with subhistory
    # storage = ctx.stream.history.storage
    # stream = MemoryStream(storage=storage)
    # if len(history_events) > 3:
    #     history_events = history_events[-3:]
    # # set subhistory to the new stream
    # await storage.set_history(stream.id, history_events)

    # # broadcast substream events to the original stream
    # with stream.sub_scope(ctx.send):
    #     await agent.ask(
    #         event.prompt,
    #         stream=stream,
    #         dependencies=ctx.dependencies,
    #         variables=ctx.variables,
    #     )

    # # drop subhistory from the storage
    # await storage.drop_history(stream.id)


async def main() -> None:
    context = Context(
        stream,
        dependencies={"agent": agent},
    )

    # trigger agent
    await context.send(CallModel(prompt="Hi! Tell me a joke!"))

    # trigger and capture output stream
    with context.stream.join(max_events=3) as events:
        await context.send(CallModel(prompt="Good one! Tell me one more"))

        async for event in events:
            print("event: ", event)

    # trigger and capture final result
    async with context.stream.get(ModelResponse) as result:
        await context.send(CallModel(prompt="And the best one!"))
        print("result: ", await result)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
