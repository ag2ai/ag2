import json
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import uuid4

from a2ui.core.schema.catalog import CatalogConfig
from a2ui.core.schema.manager import A2uiSchemaManager
from ag_ui.core import ActivitySnapshotEvent
from fastapi import FastAPI, Header
from fastapi.responses import StreamingResponse

from autogen import ConversableAgent, LLMConfig
from autogen.ag_ui import AGUIStream, RunAgentInput
from autogen.agentchat.remote import ServiceResponse

app = FastAPI()

_BASE = Path(__file__).resolve().parent
_A2UI_SPEC = _BASE / "specification" / "v0_8" / "json"


def add_catalog_id(schema):
    if schema is not None and "components" in schema and "catalogId" not in schema:
        schema["catalogId"] = "basic"
    return schema


schema_manager = A2uiSchemaManager(
    version="0.8",
    catalogs=[
        CatalogConfig.from_path(
            name="basic",
            catalog_path=str(_A2UI_SPEC / "standard_catalog_definition.json"),
            examples_path=str(_A2UI_SPEC / "catalogs" / "basic" / "examples"),
        )
    ],
    schema_modifiers=[add_catalog_id],
)

instruction = schema_manager.generate_system_prompt(
    role_description="You are a helpful AI assistant that generates rich UIs.",
    workflow_description="Analyze the user request and return an A2UI JSON payload to render interactive interfaces when appropriate.",
    ui_description="Use the provided A2UI components to structure the output.",
    include_schema=True,  # Injects the raw JSON schema
    include_examples=True,  # Injects few-shot examples
)

print(instruction)


agent = ConversableAgent(
    name="a2ui_support_bot",
    system_message=instruction,
    llm_config=LLMConfig({"model": "gpt-5"}),
)


async def a2ui_event_interceptor(response: ServiceResponse) -> AsyncIterator[ActivitySnapshotEvent]:
    if response.message and (data := response.message.get("content")):
        try:
            # TODO: how to check if the data is a A2UI JSON payload type?
            a2ui_data = json.loads(data)
            yield ActivitySnapshotEvent(
                message_id=str(uuid4()),
                activity_type="a2ui-surface",
                content={"operations": a2ui_data},
            )
            response.message = None
        except Exception:
            pass


# forwarded_props = {
#     "a2uiAction": {
#         "userAction": {
#             "name": "see_examples",
#             "sourceComponentId": "examples-btn",
#             "surfaceId": "greeting-surface",
#             "timestamp": "2026-03-05T17:53:14.947Z",
#             "context": {},
#         }
#     }
# }

# forwarded_props = {
#     "a2uiAction": {
#         "userAction": {
#             "name": "submit_query",
#             "sourceComponentId": "send-btn",
#             "surfaceId": "assistant-welcome",
#             "timestamp": "2026-03-05T17:57:21.998Z",
#             "context": {"query": "dasdasda"},
#         }
#     }
# }

stream = AGUIStream(
    agent,
    event_interceptors=[a2ui_event_interceptor],
)


@app.post("/chat")
async def run_agent(
    message: RunAgentInput,
    accept: str | None = Header(None),
) -> StreamingResponse:
    return StreamingResponse(
        stream.dispatch(message, accept=accept),
        media_type=accept or "text/event-stream",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8008)
