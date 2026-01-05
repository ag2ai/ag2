import asyncio

from dotenv import load_dotenv

from autogen import ConversableAgent, LLMConfig
from autogen.opentelemetry import instrument_agent, instrument_llm_wrapper, setup_instrumentation

load_dotenv()

# llm_config = LLMConfig(
#     model="gpt-4o-mini",
#     api_key=os.getenv("OPENAI_API_KEY"),
# )

llm_config = LLMConfig(
    model="devstral-small-2",
    api_type="ollama",
)

PYTHON_REVIEW_PROMPT = (
    "You are code reviewer. Analyze provided code and suggest your changes.\n"
    "Do not generate any code, only suggest improvements.\n"
    "Format suggestions as a task list with the following format:\n"
    "- [ ] Fix bug in `bar` function\n"
    "- [ ] Refactor `baz` class\n"
    "Mark issues as completed when they are addressed.\n"
    "Do not add new issues at review rounds - just check if the code is already fixed."
    "You should try to decrease issues numbers between review rounds.\n"
    "Accept any code after first review."
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


async def generate_code(prompt: str) -> str:
    await review_agent.a_initiate_chat(
        recipient=code_agent,
        message={"role": "user", "content": prompt},
        max_turns=3,
    )

    code_messages = [y for x in review_agent.chat_messages.values() for y in x if y["name"] == code_agent.name]
    last_message = code_messages[-1]["content"]
    lines = last_message.splitlines()
    return "\n".join(lines[1:-1])


if __name__ == "__main__":
    tracer = setup_instrumentation("local-agents-ollama", "http://127.0.0.1:14317")
    instrument_llm_wrapper(tracer)
    instrument_agent(review_agent, tracer)
    instrument_agent(code_agent, tracer)

    code = asyncio.run(
        generate_code(
            "Create a simple calculator app in Python in console without any library. "
            "It should consume whole operation string with all operands and operators, "
            "like '2 + 3 * 4', and output the correct result."
            "Followed input without start digit should be allowed like '-5 + 3' or '+5 - 3' "
            "and applied to previous result. Example: '-5 + 3' -> '-2' "
            "and then '*2' -> '-4' or '+7' -> '5'. "
            "To exit the application user should enter 'exit' or 'quit'. "
            "The solution should be in a single file called result.py"
        )
    )

    print(code)
