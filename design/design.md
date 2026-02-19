# AG2 Redesign

## Questions to discuss

1. Sync / async / both?


## New Configuration API

We should provide users explicit, predictable and clear way to configure Model, Model and Conversation settings.
The difference between various LLM providers should be type-based for better typing support and clear DX.

Suggested API:

1. Simple example - it is very important to create clear API for common parameters: API Key, preferred model, connection url
    These parameters are the most common to find / edit, so we should provide the same experience for all providers

    ```python
    from autogen.v1 import OpenAIModel, Agent
    # from autogen.v1 import GoogleModel

    model = OpenAIModel(model="gpt-5", api_key="sk-...", base_url="https:/...")
    # model = GoogleModel(model="gemini-3", api_key="...", base_url="https:/...")
    agent = Agent(model=model)
    ```

2. Options override example - user should be able to reuse the same model / provider for multiple agents with just a small difference

    ```python
    from autogen.v1 import OpenAIModel, Agent

    model = OpenAIModel(api_key="sk-...", base_url="https:/...")

    agent1 = Agent(
        model=model.update(model="gpt-5"),  # creates a new updated instance
    )

    agent1 = Agent(
        model=model.update(model="gpt-5-mini", temperature=0.8),
    )
    ```

3. Environment variables usage - each model should try to get API_KEY from env variables automatically if it was not provided. We should
    respect common naming practices, but allow to use common for our product naming:

    ```python
    from autogen.v1 import OpenAIModel, GoogleModel

    # fallback to OPENAI_API_KEY or AG2_API_KEY
    OpenAIModel(model="gpt-5")

    # fallback to GEMINI_API_KEY or AG2_API_KEY
    GoogleModel(model="gemini-3")
    ```

4. Models Composition. Right now `LLMConfig` is too complex for common cases. It doesn't hide fallback mechanism from final user.
    in general cases we just need a single model. So, I suggest to hide complex logic under "advanced" API:

    ```python
    from autogen.v1 import OpenAIModel, Agent, ModelComposition

    model = OpenAIModel(model="gpt-5")

    agent = Agent(model=ModelComposition(
        model.update(api_key="sk-...", base_url="https://..."),
        model.update(api_key="sk-...", base_url="https://proxy.com"),  # fallback model
    ))
    ```

5. Delayed configuration - it is a common pattern to separated logic of configuration. So, we should provide
    users with an ability to configure agent & config separately and join when on action

    ```python
    from autogen.v1 import OpenAIModel, Agent

    model = OpenAIModel(model="gpt-5")

    agent = Agent(tools=[...], system_message="...")

    @post("/")
    await def ask_agent(
        message: str,
        model: DI[OpenAIModel],  # provide settings from DI
    ):
        return await agent.ask(message, model=model)
    ```

    Also, it could be helpful for cases, when we allow users to operate their own credentials for our agent

    ```python
    from autogen.v1 import OpenAIModel, Agent

    model = OpenAIModel(model="gpt-5")

    agent = Agent(tools=[...], system_message="...")

    @post("/")
    await def ask_agent(message: str, user_id: UUID):
        model = await get_user_model(user_id)
        return await agent.ask(message, model=model)
    ```

    In case, when we have already configure model, provided one should override the current


    ```python
    from autogen.v1 import OpenAIModel, Agent

    model = OpenAIModel(model="gpt-5")
    agent = Agent(model=model)

    agent.ask("Hi!", model=model.update("gpt-4"))  # replace original "gpt-5" model
    ```

    But sometimes we want to update current configuration on call (tune prompt / temp / etc from admin UI).
    So, we need to provide an ability update / override original agent & model parameters on action

    ```python
    from autogen.v1 import OpenAIModel, Agent

    model = OpenAIModel(model="gpt-5")
    agent = Agent(model=model, tools=[tool1], system_message="1")

    @post("/")
    await def ask_agent(
        message: str,
        new_system_message: str,
        user_api_key: str,
    ):
        user_agent = agent.update(  # creates a new instance
            model=OpenAIModel(api_key=user_api_key),
            tools=[tool2]  # client-side additional tools
        )

        # Final configuration
        # model: GPT-5, user_api_key
        # system_message: 2
        # tools: tool1, tool2
        return await user_agent.ask(message, system_message="2")
    ```

### Suggested implementation

From the requirements above I can suggest the following design points
1. LLMModel should be stateless
2. LLMModel should be immutable (`update` creates a new instance)
3. Any LLMModel should follows a strict Interface
4. LLMModel should self-validate
5. LLMModel should be lazy - initialized

Code implementation could be like this

```python
from typing import Protocol

class LLMClient(Protocol):
    # to discuss. I believe, it should be bi-streaming
    async def ask(self, message: "InMessage") -> "OutMessage":
        ...

class OpenAIClient(LLMClient):
    pass

class GeminiClient(LLMClient):
    pass

# Model
import os
from functools import lru_cache
from typing import Any, runtime_checkable
from abc import abstractmethod
from dataclasses import dataclass, asdict, field


@runtime_checkable
class ModelInterface(Protocol):
    """Public Model API."""
    @abstractmethod
    def create(self) -> LLMClient: ...

    @abstractmethod
    def __or__(self, other: "ModelInterface") -> "ModelInterface": ...


class ModelConfig(ModelInterface):
    """Abstract class for specific LLM config implementation."""
    model: str
    api_key: str | None
    base_url: str | None

    @abstractmethod
    def update(self) -> "ModelConfig": ...


def filter_value(data: dict[str, Any], value: Any = None) -> dict[str, Any]:
    return {k: v for k, v in data.items() if v != value}


@dataclass(frozen=True, slots=True)
class OpenAIModel(ModelConfig):
    model: str

    api_key: str | None = field(default=None, kw_only=True)
    base_url: str | None = field(default=None, kw_only=True)

    def update(
        self,
        /,  # make options keyword-only
        model: str | None = None,
        api_key: str | None = None,
    ) -> "OpenAIModel":
        return OpenAIModel(**(
            asdict(self) |  # override original options by provided
            filter_value({
                "model": model,
                "api_key": api_key
            })
        ))

    def validate(self) -> dict[str, Any]:
        client_options = asdict(self)

        api_key = client_options.get("api_key") or \
                    os.getenv("OPENAI_API_KEY") or \
                    os.getenv("AG2_API_KEY")

        if not api_key:
            raise ValueError("You should provide `api_key` to use OpenAIModel")

        client_options["api_key"] = api_key
        return client_options

    @lru_cache(1)
    def create(self) -> OpenAIClient:
        return OpenAIClient(**self.validate())

    def __or__(self, other: "ModelInterface") -> "OpenAIModel":
        if not isinstance(other, OpenAIModel):
            return NotImplemented
        return self.update(**asdict(other))


# Agent config usage
class Agent:
    def __init__(self, model: ModelInterface):
        self.model = model

    def update(self, model: ModelInterface) -> "Agent":
        return Agent(model=self.model | model)

    async def ask(
        self,
        message: "str | InMessage",
        *,
        model: ModelInterface | None = None,
    ) -> "OutMessage":
        # lazy model construction on demand
        client = (model or self.model).create()

        if isinstance(message, str):
            message = TextMessage()

        return await client.ask(message)


# Model Composition
@dataclass(frozen=True, slots=True)
class ModelComposition(ModelInterface):
    def __init__(self, *models: ModelInterface):
        if not models:
            raise ValueError("You should provide at least 1 model")

        self.models = models

    @lru_cache(1)
    def create(self) -> LLMClient:
        return ClientComposition(*(m.create() for m in self.models))

    def __or__(self, other: "ModelInterface") -> "ModelComposition":
        if not isinstance(other, ModelInterface):
            return NotImplemented

        models_iter = iter(self.models)
        return ModelComposition(
            next(models_iter) | other,
            *models_iter
        )
```

## New Agent API

### Direct LLM Conversation (Low-level, but suitable)

First of all, user should be able to interact with LLM directly in a suitable way.
Suggested method shouldn't trigger any tools or process LLM response any other way.

It should just
- pass user message to LLM client
- return raw result to client
- trigger middlewares on go (Guardrails are middlewares too)

```python
answer = await agent.ask("Hi, agent!")

match answer.message:
    case Text() as msg:
        print(msg)

    case Binary(body):
        print(body)

    case default:
        print(default)
```

Also, user should be able to pass multiple messages in a turn:

```python
await agent.ask(
    "What company is this logo from?",
    Image(url="https://example.com/logo.png")
)
```

Incoming message types:  `Text` (`str` casted to `Text`) or `Image | Audio | Document | Video` (url or binary) or `Binary` fallback
Service message types: `SystemPrompt`, `ToolResult`
Response parts: `Text`, `ToolCall`, `BuiltinToolCall`, `BuiltinToolResult`, `Thinking`, `Binary`

To support structured conversation using such API, we can provide `history` property

```python
answer = await agent.ask("Hi, agent!")
print(answer.message)

next_turn = await agent.ask(
    *answer.history  # SystemPrompt and other parts will be reused,
    "Nice to see you!",
)
print(next_turn.message)
```

Alternatively (or in addition) we can return multitern-object

```python
conversation = await agent.ask("Hi, agent!")
print(conversation.message)  # current LLM answer

next_conversation = await conversation.ask("Nice to see you!")
print(next_conversation.message)  # next LLM answer
```

ResponseSchema - I believe, that response schema shouldn't be a part of Model / Agent config.
My opinion - it should be a conversation-level setting.

So, I suggest to pass `response_shema` on each agent call:

```python
@dataclass
class Answer:
    text: str
    description: str

conversation = await agent.ask("Hi, agent!", response_shema=Answer)
print(conversation.message.text)  # message field is `Answer` type now

next_conversation = await conversation.ask("Nice to see you!", response_shema=Answer)
print(next_conversation.message.text)  # next LLM answer
```

`Conversation` interface

```python
class Conversation[T=Message]:
    message: T
    history: list[Message]

    cost: Cost | None = None

    async def ask[T1=Message](self, *messages: Message | str, response_shema: type[T1] | None = None) -> Conversation[T1]:
        return make_new_conversation(*self.history, *messages, response_schema=response_shema)


class Agent:
    async def ask[T=Message](self, *messages: Message | str, response_shema: type[T1] | None = None) -> Conversation[T1]:
        return make_new_conversation(self.system, *messages, response_schema=response_shema)
```

### "Smart" LLM Conversation (Preferred API for simple cases)

Pretty the same API, but now agent turn conversation back to user only on HITL


```python
# send User message, run tools, execute code, shell automatically
conversation = await agent.process("Hi, agent!")
```

`conversation` - is the same object with previous example (with `history`, `message` and `ask` interface).
But now it not just returns direct LLM answer, but run cinversation automatically.


So, step by step agent conversation should looks like follow:

```python
message = "Hi, agent!"
conversation = await agent.process(message)

while message != "exit":
    message = input(conversation.message)
    conversation = await conversation.process(message)
```

### Streaming Response

If user wants to capture processing events or model returns streaming response, we can provide a special
object to listen events

```python
async with agent.stream() as stream:
    message = "Hi, agent!"

    while message != "exit":
        async for event in stream.send(message):
            match event:
                case Thinking(content):
                    print(f"Model thinking: {content}")
                case InputRequired:
                    message = input(event.message)
```

### Bidirection Streaming

To support bidirection streaming we can use the same API with additional object

```python
async with agent.stream() as stream:
    write_stream = stream.writer()

    await write_stream.send(message)

    async for event in stream.listen():
        match event:
            case Thinking(content):
                print(f"Model thinking: {content}")

            case InputRequired:
                user_input = input(event.message)
                await write_stream.send(user_input)
```

HTTP Usage example:

```python
@post("/chat")
async def init_chat(
    message: str,
    user_id: UUID,
    chat_manager: DI[ChatManager]
) -> StreamingResponse:
    async with agent.stream() as stream:
        write_stream = stream.writer()
        chat_manager.register(user_id, write_stream)

        await write_stream.send(message)

        return StreamingResponse(stream.listen())

@post("/push")
async def push_message_to_chat(
    message: str,
    user_id: UUID,
    chat_manager: DI[ChatManager]
) -> None:
    writer = chat_manager.get(user_id)
    if not writer:
        raise HttpError(404)
    await writer.send(message)
```


Streaming API:


```python
class Stream:
    def writer(self) -> "Writer": ...

    async def send(self, *messages: Message | str) -> AsyncIterator[OutMessage]:
        writer = self.writer()

        async for m in messages:
            await writer.send(m)

        async for msg in self.listen():
            match msg:
                case InputRequired():
                    yield msg
                    # break endless stream if input required
                    break
                case _:
                    yield msg

    async def listen(self) -> AsyncIterator[OutMessage]:
        """Endless stream iterator."""

    # to discuss
    async def pause(self) -> None: ...

    async def continue(self) -> None: ...

    async def break(self) -> None: ...
```

### History manipulation

Strong part of such API is that user able to control current context each step

```python
message = "Hi, agent!"
conversation = await agent.process(message)

while message != "exit":
    message = input(conversation.message)

    if len(onversation.history) > 10:
        conversation.history = [await summary_agent.ask(
            "Summary the following history",
            *conversation.history,
            response_type=Text,  # return single summary message
        )]

    conversation = await conversation.process(message)
```

To restore conversation from history (from DB as an example) user can use special factory

```python
history = await get_history_from_db()

conversation = await agent.restore_conversation(history)

user_input = input(conversation.message)  # print message on top

conversation = await agent.process(user_input)  # continue conversation
```
