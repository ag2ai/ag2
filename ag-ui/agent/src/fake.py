
from collections.abc import AsyncIterator
from uuid import uuid4

from ag_ui.core import (
    RunAgentInput,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    TextMessageChunkEvent,
    ToolCallArgsEvent,
    ToolCallChunkEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)
from ag_ui.encoder import EventEncoder
from fastapi import FastAPI, Header
from fastapi.responses import StreamingResponse

from autogen.a2a.remote.agent_service import AgentService
from autogen.a2a.remote.protocol import RequestMessage

from .ag import agent

app = FastAPI()

service = AgentService(agent)


@app.post("/chat")
async def request(
    data: RunAgentInput,
    accept: str | None = Header(None)
):
    encoder = EventEncoder(accept=accept)

    client_tools = []
    client_tools_names = set()
    for t in data.tools:
        func = t.model_dump(exclude_none=True)
        client_tools.append({
            "type": "function",
            "function": func,
        })
        client_tools_names.add(func["name"])

    message = RequestMessage(
        messages=[m.model_dump(exclude_none=True) for m in data.messages],
        client_tools=client_tools,
    )

    async def event_generator() -> AsyncIterator[str]:
        try:
            yield encoder.encode(RunStartedEvent(
                thread_id=data.thread_id,
                run_id=data.run_id
            ))

            async for response in service(message):
                msg_id = str(uuid4())

                if msg := response.message:
                    content = msg.get("content", "")

                    for tool_response in msg.get("tool_responses", []):
                        yield encoder.encode(ToolCallResultEvent(
                            tool_call_id=tool_response["tool_call_id"],
                            content=tool_response["content"],
                            message_id=msg_id
                        ))
                        yield encoder.encode(ToolCallEndEvent(
                            tool_call_id=tool_response["tool_call_id"],
                        ))
                        # do not send tool result as chat message
                        content = ""

                    for tool_call in msg.get("tool_calls", []):
                        func = tool_call["function"]

                        if (name := func.get("name")) in client_tools_names:
                            yield encoder.encode(ToolCallChunkEvent(
                                parent_message_id=msg_id,
                                tool_call_id=tool_call.get("id"),
                                tool_call_name=name,
                                delta=func.get("arguments"),
                            ))

                        else:
                            yield encoder.encode(ToolCallStartEvent(
                                tool_call_id=tool_call.get("id"),
                                tool_call_name=name,
                            ))
                            yield encoder.encode(ToolCallArgsEvent(
                                tool_call_id=tool_call.get("id"),
                                delta=func.get("arguments"),
                            ))

                    if content:
                        yield encoder.encode(TextMessageChunkEvent(
                            message_id=msg_id,
                            delta=content
                        ))

        except Exception as e:
            yield encoder.encode(RunErrorEvent(
                message=repr(e)
            ))
            raise e

        else:
            yield encoder.encode(RunFinishedEvent(
                thread_id=data.thread_id,
                run_id=data.run_id
            ))

    return StreamingResponse(
        event_generator(),
        media_type=encoder.get_content_type()
    )
