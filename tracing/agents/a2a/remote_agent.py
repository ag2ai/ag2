import os

from autogen import ConversableAgent, LLMConfig
from autogen.a2a import A2aAgentServer
from autogen.instrumentation import instrument_a2a_server, setup_instrumentation

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
tracer = setup_instrumentation("remote-tech-agent")
server = instrument_a2a_server(server, tracer)
app = server.build()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8010)
