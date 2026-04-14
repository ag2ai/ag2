from fastapi import FastAPI, Header
from fastapi.responses import StreamingResponse

from autogen.beta import Agent, config
from autogen.beta.ag_ui import AGUIStream, RunAgentInput

agent = Agent(
    name="support_bot",
    prompt="You help users with billing questions.",
    config=config.OpenAIResponsesConfig(model="gpt-4o-mini", streaming=True),
)

stream = AGUIStream(agent)
app = FastAPI()


@app.post("/agentic_chat")
async def run_agent(
    message: RunAgentInput,
    accept: str | None = Header(None),
) -> StreamingResponse:
    return StreamingResponse(
        stream.dispatch(message, accept=accept),
        media_type=accept or "text/event-stream",
    )
