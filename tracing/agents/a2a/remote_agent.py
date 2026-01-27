import os

from dotenv import load_dotenv
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from autogen import ConversableAgent, LLMConfig
from autogen.a2a import A2aAgentServer
from autogen.opentelemetry import instrument_a2a_server

load_dotenv()

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

server = A2aAgentServer(tech_agent, url="http://localhost:18123/")
resource = Resource.create(attributes={"service.name": "remote-tech-agent"})
tracer_provider = TracerProvider(resource=resource)
exporter = OTLPSpanExporter(endpoint="http://127.0.0.1:4317")
processor = BatchSpanProcessor(exporter)
tracer_provider.add_span_processor(processor)
trace.set_tracer_provider(tracer_provider)
server = instrument_a2a_server(server, tracer_provider=tracer_provider)
app = server.build()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=18123)
