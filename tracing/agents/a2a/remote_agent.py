import os

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from opentelemetry.instrumentation.starlette import StarletteInstrumentor

from autogen import ConversableAgent, LLMConfig
from autogen.a2a import A2aAgentServer
from autogen.instrumentation import instrument_a2a_executor, setup_instrumentation

llm_config = LLMConfig(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)

tech_agent = ConversableAgent(
    name="tech_agent",
    system_message="""You solve technical problems like software bugs
    and hardware issues.""",
    llm_config=llm_config,
)

server = A2aAgentServer(tech_agent, url="http://localhost:8010/")

tracer, tracer_provider = setup_instrumentation("remote-tech-agent")
# executor = instrument_a2a_executor(server.executor, tracer)

app = server.build(
    request_handler=DefaultRequestHandler(
        agent_executor=server.executor,
        task_store=InMemoryTaskStore(),
    ),
)

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class OTELMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        print(request.headers)
        return await call_next(request)


# StarletteInstrumentor.instrument_app(app, tracer_provider=tracer_provider)
app.add_middleware(OTELMiddleware)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8010)
