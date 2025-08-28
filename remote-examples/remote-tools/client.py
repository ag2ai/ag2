import os

from autogen import ConversableAgent, LLMConfig
from autogen.remote import HTTPRemoteAgent

llm_config = LLMConfig(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)

agent = ConversableAgent(
    name="local",
    human_input_mode="NEVER",
    llm_config=llm_config,
)

remote_agent = HTTPRemoteAgent(
    url="http://localhost:8000",
    name="remote",
)


def generate_code(prompt: str) -> str:
    agent.initiate_chat(
        recipient=remote_agent,
        message={"role": "user", "content": prompt},
    )

    return agent.last_message()


if __name__ == "__main__":
    code = generate_code("I was born on 25th of March 1995, what day was it?")
    print(code)
