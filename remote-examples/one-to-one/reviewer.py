import os

from autogen import ConversableAgent, LLMConfig
from autogen.remote import RemoteAgent

llm_config = LLMConfig(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)


PYTHON_REVIEW_PROMPT = (
    "You are code reviewer. Analyze provided code and suggest your changes.\n"
    "Do not generate any code, only suggest improvements.\n"
    "Format suggestions as a task list with the following format:\n"
    "- [ ] Fix bug in `bar` function\n"
    "- [ ] Refactor `baz` class\n"
    "Mark issues as completed when they are addressed.\n"
    "You should try to decrease issues numbers between review rounds.\n"
    "You are a strongly fun of SOLID principles.\n"
    "Any python code have to be typed with proper type hints.\n"
    "Send NOT MERGE message if you see any issues. Otherwise send LGTM message.\n"
    "Never use LGTM keyword in text until you are absolutely sure that the code is ready to merge.\n"
    "Last line of review should always be the final decision: either 'NOT MERGE' or 'LGTM'.\n"
)

review_agent = ConversableAgent(
    name="reviewer",
    system_message=PYTHON_REVIEW_PROMPT,
    human_input_mode="NEVER",
    llm_config=llm_config,
)


code_agent = RemoteAgent(url="http://localhost:8000", name="coder")


def generate_code(prompt: str) -> str:
    review_agent.initiate_chat(
        recipient=code_agent,
        message={"role": "user", "content": prompt},
    )

    code_messages = [y for x in review_agent.chat_messages.values() for y in x if y["name"] == code_agent.name]
    last_message = code_messages[-1]["content"]
    lines = last_message.splitlines()
    return "\n".join(lines[1:-1])


if __name__ == "__main__":
    code = generate_code(
        "Create a simple calculator app in Python in console without any library. "
        "It should consume whole operation string with all operands and operators, "
        "like '2 + 3 * 4', and output the correct result."
        "Followed input without start digit should be allowed like '-5 + 3' or '+5 - 3' "
        "and applied to previous result. Example: '-5 + 3' -> '-2' "
        "and then '*2' -> '-4' or '+7' -> '5'. "
        "To exit the application user should enter 'exit' or 'quit'. "
        "The solution should be in a single file called result.py"
    )

    print(code)
