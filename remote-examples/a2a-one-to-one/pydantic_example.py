import uvicorn
from pydantic_ai import Agent

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

agent = Agent("openai:gpt-4o-mini", system_prompt=PYTHON_CODER_PROMPT)

app = agent.to_a2a(url="http://localhost:9999")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9999)
