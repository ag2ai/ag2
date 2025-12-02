import os

from autogen import ConversableAgent, LLMConfig
from dotenv import load_dotenv
load_dotenv()

llm_config = LLMConfig(
    config_list={
        "api_type": "responses",
        "model": "gpt-5.1",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "built_in_tools": ["apply_patch"],
    },
)

coding_agent = ConversableAgent(
    name="coding_assistant",
    llm_config=llm_config,
    system_message="""You are a helpful coding assistant. You can create, edit, and delete files
    using the apply_patch tool. When making changes, always use the apply_patch tool rather than
    writing raw code blocks. Be precise with your file operations and explain what you're doing.""",
)

# Create a new project structure
result = coding_agent.initiate_chat(
    recipient=coding_agent,
    message="""
    Create a new Python project folder called 'calculator' with the following structure:
    1. Create a main.py file with a Calculator class that has methods for add, subtract, multiply, and divide  
    """,
    max_turns=2,
    clear_history=True,
)

print("Chat Summary:")
print(result.summary)