import os

from autogen import ConversableAgent, LLMConfig

config = LLMConfig(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)

PYTHON_CODER_PROMPT = (
    "You are an expert Python developer. "
    "When asked to make changes to a code file, "
    "you should update the code to reflect the requested changes. "
    "Do not provide explanations or context; just return the updated code."
    "You should work in a single file. Just a code listing, without markdown markup."
    "Do not generate trailing whitespace or extra empty lines. "
    "Strongly follow provided recommendations for code quality. "
    "Do not generate code comments, unless required by linter. "
)

code_agent = ConversableAgent(
    name="code",
    system_message=PYTHON_CODER_PROMPT,
    llm_config=config,
    is_termination_msg=lambda x: "LGTM" in (x.get("content", "") if isinstance(x, dict) else x),
    human_input_mode="NEVER",
    silent=True,
)


from collections import defaultdict
from typing import Annotated, Any
from uuid import uuid4

from fastapi import Depends, FastAPI, Request

from autogen.remote import (
    AgentBusEvent,
    ProtocolEvents,
    SendEvent,
    StopEvent,
    serialize_event,
)

app = FastAPI()
chat_history: defaultdict[int, list[dict[str | Any] | str | None]] = defaultdict(list)


async def serialize_request_to_event(request: Request) -> AgentBusEvent:
    return serialize_event(await request.json())


def chat_id(request: Request) -> int:
    if "X-Chat-Id" in request.headers:
        return int(request.headers["X-Chat-Id"])
    return uuid4().int


@app.post("/")
async def handler(
    event: Annotated[AgentBusEvent, Depends(serialize_request_to_event)],
    chat_id: Annotated[int, Depends(chat_id)],
) -> AgentBusEvent | None:
    if event.event_type is ProtocolEvents.STOP_CHAT:
        del chat_history[chat_id]
        return

    messages = chat_history[chat_id]

    if event.event_type is ProtocolEvents.SEND_MESSAGE:
        messages.append(event.content)
        return

    if event.event_type is ProtocolEvents.NEXT_SPEAKER:
        reply = code_agent.generate_reply(
            messages,
            None,
            exclude={
                ConversableAgent.generate_function_call_reply,
                ConversableAgent.generate_tool_calls_reply,
                ConversableAgent.generate_code_execution_reply,
            },
        )

        # TODO: how to call local tools?

        if reply is None:
            return StopEvent()

        messages.append(reply)
        return SendEvent(content=reply)

    raise NotImplementedError("unsupported protocol method")
