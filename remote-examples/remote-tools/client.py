import asyncio
import os

from autogen import ConversableAgent, LLMConfig
from autogen.a2a import A2aRemoteAgent

llm_config = LLMConfig(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)

agent = ConversableAgent(
    name="local",
    human_input_mode="NEVER",
    llm_config=llm_config,
)

remote_agent = A2aRemoteAgent(
    url="http://localhost:8000",
    name="remote",
)


async def generate_code(prompt: str) -> str:
    await agent.a_initiate_chat(
        recipient=remote_agent,
        message={"role": "user", "content": prompt},
    )

    return agent.last_message()


if __name__ == "__main__":
    code = asyncio.run(generate_code("I was born on 25th of March 1995, what day was it?"))
    print(code)
