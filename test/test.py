from autogen import LLMConfig, ConversableAgent
import os
from dotenv import load_dotenv
load_dotenv()

llm_config = LLMConfig(
    config_list={
        "api_type": "responses",
        "model": "gpt-5.1",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "built_in_tools": ["apply_patch"],
        "workspace_dir": "./my_project_folder",
        "allowed_paths": ["src/**", "tests/**"],
    },
    # workspace_dir="./my_project_folder",
    # allowed_paths=["src/**", "*.py", "tests/**"],  # Only allow operations in these paths
)

# Create agent - no need to manually create editor or patch_tool
coding_agent = ConversableAgent(
    name="coding_assistant",
    llm_config=llm_config,
    system_message="""You are a helpful coding assistant...""",
)

result2 = coding_agent.initiate_chat(
    recipient=coding_agent,
    message="Create newdir/settings.py ",
    max_turns=2,
)

print(result2.summary)