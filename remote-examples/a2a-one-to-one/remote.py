import os

import uvicorn

from autogen import ConversableAgent, LLMConfig
from autogen.a2a import A2aAgentServer

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
    a2a_agent_server = A2aAgentServer(code_agent)
    uvicorn.run(a2a_agent_server.build(), host="0.0.0.0", port=9999)
