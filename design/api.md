# BaseEvent Driven Agent communication

```python
from typing import Protocol, Any, Self, Intersection, Callable, Awaitable

type EventType = Any


class Filterable(Protocol):
    def until(
        self,
        *,
        type: type[EventType] | tuple[EventType, ...] | None = None,
        **options: Any
    ) -> Self: ...

    def include(
        self,
        *,
        type: type[EventType] | tuple[EventType, ...] | None = None,
        **options: Any
    ) -> Self: ...

    def exclude(
        self,
        *,
        type: type[EventType] | tuple[EventType, ...] | None = None,
        **options: Any
    ) -> Self: ...


class Stream(StreamSubscriber, Filterable, Protocol):
    async def send(self, event: EventType) -> None: ...

    # listen methods
    def listen(self) -> AsyncIterator[EventType]: ...

    def join(self) -> AsyncIterator[EventType]: ...

    # fetch methods
    async def get(self) -> EventType: ...

    async def get_batch(self) -> list[EventType]: ...

    # stop methods
    async def pause(self):
        """Locks `send` method."""
        ...

    async def unpause(self): ...

    # subscribe methods
    def subscribe(self, listener: Callable[[EventType], Awaitable[Any]]) -> SubId: ...

    def unsubscriber(self, subscription: SubId) -> None: ...
```

# Direct Request

```python
conversation_result = await agent.ask("Hi, agent!")

# or
conversation = agent.ask("Hi, agent!")
result = await conversation
```

# Listen events

## Streaming

```python
stream = Stream()

conversation = agent.ask("Hi, agent!", stream=stream)
await conversation.start()

# listen all events from stream start
async for event in stream.listen():
    match event:
        case Thinking():
            ...
        case StreamEvent():
            ...
        case TextMessage():
            ...
```

Alternatively you can use `stream.join` method with same API to listen new events from join moment

```python
stream = Stream()

conversation = agent.ask("Hi, agent!", stream=stream)
await conversation.start()

# listen new events
async for event in stream.join():
    match event:
        case Thinking():
            ...
        case StreamEvent():
            ...
        case TextMessage():
            ...
```

## Bidirectional

```python
stream = Stream()

conversation = agent.ask("Hi, agent!", stream=stream)
await conversation.start()

async for event in stream.listen():
    ...
    await stream.send(TextMessage("Hi!"))
```

## Get next event

```python
stream = Stream()

conversation = agent.ask("Hi, agent!", stream=stream)
await conversation.start()

event = await stream.get(timeout=10.0)
```

## Filtering

### By type

```python
stream = Stream()

conversation = agent.ask("Hi, agent!", stream=stream)
await conversation.start()

stream = stream.where(type=ToolCall)

# Iter over `ToolCall`
async for event in stream.listen():
    ...

# Get next `ToolCall`
tool_call_event = await stream.get():
```

### By types

```python
stream = Stream()

conversation = agent.ask("Hi, agent!", stream=stream)
await conversation.start()

stream = stream.include(
    type=(ToolCall, ToolResult)
)

# Iter over `ToolCall` | `ToolResult`
async for event in stream.listen():
    ...

# Get next `ToolCall` | `ToolResult`
tool_call_event = await stream.get():
```

### By argument

```python
stream = Stream()

conversation = agent.ask("Hi, agent!", stream=stream)
await conversation.start()

stream = stream.include(speaker="agentname")

# Iter over `BaseEvent(..., speaker="agentname")`
async for event in stream.listen():
    ...

# Get next `BaseEvent(..., speaker="agentname")`
tool_call_event = await stream.get():
```

### By argument and type

```python
stream = Stream()

conversation = agent.ask("Hi, agent!", stream=stream)
await conversation.start()

stream = stream.include(
    type=TextMessage,
    speaker="agentname",
)

# Iter over `TextMessage(..., speaker="agentname")`
async for event in stream.listen():
    ...

# Get next `TextMessage(..., speaker="agentname")`
tool_call_event = await stream.get():
```

### By excluded type

```python
stream = Stream()

conversation = agent.ask("Hi, agent!", stream=stream)
await conversation.start()

stream = stream.exclude(type=StreamEvent)

# Iter over `BaseEvent != StreamEvent`
async for event in stream.listen():
    ...

# Get next `BaseEvent != StreamEvent`
tool_call_event = await stream.get():
```

### By complex condition

```python
stream = Stream()

conversation = agent.ask("Hi, agent!", stream=stream)
await conversation.start()

stream = stream.include(
    type=(ToolCall, ToolResult)
).exclude(
    tool_name="get_weather"
)

# Iter over `ToolCall(..., tool_name!="get_weather")` | `ToolResult(..., tool_name!="get_weather")`
async for event in stream.listen():
    ...

# Get next `ToolCall(..., tool_name!="get_weather")` | `ToolResult(..., tool_name!="get_weather")`
tool_call_event = await stream.get():
```

# History

## Agent-level

```python
storage = MemoryStorage(max_messages=100)  # max messages per stream (by id)

stream = Stream(storage=storage)

agent = Agent("agentname", stream=stream)

answer = await agent.ask("Hi, agent!")

user_input = input(answer)

# load history from the same stream
next_answer = await agent.ask(user_input)

# drop stream with history and replace by a new one
agent.clear_history()
```

## Each request

```python
storage = MemoryStorage(max_messages=100)

stream = Stream(storage=storage)

# load history from the same stream
@app.post("/")
async def endpoint(message: str):
    return await agent.ask(user_input, stream=stream)

# or use another stream object with the same id
@app.post("/{stream_id}")
async def endpoint(message: str, stream_id: StreamId):
    another_stream = Stream(storage=storage, id=stream_id)
    return await agent.ask(user_input, stream=another_stream)
```

## Details

```python
class MemoryStorage:
    def __init__(self) -> None:
        self.__storage: dict[StreamId, list[EventType]] = defaultdict(list)

    async def get_messages(self, stream_id: StreamId) -> AsyncIterator[EventType]:
        for event in self.__storage[stream_id]:
            yield event

    async def write(self, stream_id: StreamId, event: EventType) -> None:
        self.__storage[stream_id].append(event)


class History:
    def __init__(self, storage: MemoryStorage, stream_id: StreamId) -> None:
        self.storage = storage
        self.stream_id = stream_id

    async def get_messages(self) -> AsyncIterator[EventType]:
        async for event in self.storage.get_messages(self.stream_id):
            yield event


class Stream:
    def __init__(self, storage: MemoryStorage) -> None:
        self.id = str(uuid4())
        self.history = History(storage, self.id)
        self.subscribe(lambda event: storage.write(self.id, event))

    async def listen(self) -> AsyncIterator[EventType]:
        async for old_event in self.history.get_messages():
            yield old_event

        # iter over new messages
        async for event in self.join():
            yield event
```


# LLMClient

```python
class LLMClient:
    def __call__(self, *msgs, stream, speaker) -> RunId:
        ...

class Agent:
    async def ask(self, *msgs, stream: Stream | None = None) -> EventType:
        stream = stream or self.stream

        run_id = await self.llm_client(
            *(await stream.history.get_messages()),
            *msgs,
            stream=stream,
            speaker=self.name
        )

        return await stream.include(
            type=TextMessage,
            run_id=run_id,
        ).get()
```

# Cases

## Concurrent Agents execution

```python
stream = Stream()

agent1 = Agent("agent1")
agent2 = Agent("agent2")

await agent1.ask("Hi, agent1!", stream=stream).start()
await agent2.ask("Hi, agent2!", stream=stream).start()

# listen all events from 2 agents
async for event in stream.listen():
    match event:
        case Thinking():
            ...
        case StreamEvent():
            ...
        case TextMessage():
            ...
```

## Subagents with scoped context

```python
parent_stream = Stream()

agent1 = Agent("agent1")
agent1_result = await agent.ask("Hi, agent1!", stream=parent_stream)

new_agent = Agent("agent2")

# create new stream with clear history
substream = Stream()
# send agent2 stream events to parent_stream
substream.subscribe(parent_stream.send)

# ask agent2 with agent1 answer
agent2_result = await agent.ask(agent1_result, stream=substream)
```

## AG-UI integration

```python
agent = Agent("agent")

@app.post("/")
async def chat(input):
    stream = Stream()
    await agent.ask(*input.messages, stream=stream)
    return StreamingResponse(AGUIAdapter(stream))
```

## Background agent AG-UI integration

```python
stream = Stream()  # persistent history

await agent.ask("Run forever", stream=stream).start()

@app.post("/")
async def chat(input):
    # connect to running agent stream
    return StreamingResponse(AGUIAdapter(stream))
```

## BaseEvent Driven agent call

```python
stream = Stream()

agent = Agent("agent", stream=stream)

# subscribe agent on `UserInput` events
stream.include(type=UserInput).subscribe(agent.ask)

# run agent forewer
await agent.ask().run()

...

# send event to schedule for agent processing
await stream.send("Set agent a task")
```

## Tools Implementation

```python
stream = Stream()

@tool
async def func(): ...

# subscribe tool on calls
stream.include(
    type=ToolCall,
    tool_name="func"
).subscribe(
    lambda event: func(event, stream=stream)
)

# send tool execution
tool_call = ToolCall(name="func", arguments="")
await stream.send(tool_call)

# wait for answer
tool_result = await stream.include(
    type=ToolResult,
    call_id=tool_call.id
).get()
```

## HITL Implementation

```python
stream = Stream()

async def console_hitl(event: AskUserInput) -> TextMessage:
    return input(event.message)

# subscribe console HITL on events
stream.include(type=AskUserInput).subscribe(console_hitl)
```
