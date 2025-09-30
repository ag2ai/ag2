import os

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from autogen import ConversableAgent, LLMConfig
from autogen.remote.a2a import AutogenAgentExecutor

llm_config = LLMConfig(
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
    name="coder",
    system_message=PYTHON_CODER_PROMPT,
    llm_config=llm_config,
    is_termination_msg=lambda x: "LGTM" in x.get("content", ""),
    human_input_mode="NEVER",
    silent=True,
)

if __name__ == "__main__":
    skill = AgentSkill(
        id="hello_world",
        name="Returns hello world",
        description="just returns hello world",
        tags=["hello world"],
        examples=["hi", "hello world"],
    )

    public_agent_card = AgentCard(
        name="Hello World Agent",
        description="Just a hello world agent",
        url="http://magic-useless-url/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
        supports_authenticated_extended_card=False,
    )

    request_handler = DefaultRequestHandler(
        agent_executor=AutogenAgentExecutor(code_agent),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=public_agent_card,
        http_handler=request_handler,
    )

    uvicorn.run(server.build(), host="0.0.0.0", port=9999)
