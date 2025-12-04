import os

from dotenv import load_dotenv

from autogen import ConversableAgent, LLMConfig

load_dotenv()
# add async patches configuration
llm_config = LLMConfig(
    config_list={
        "api_type": "responses",
        "model": "gpt-5.1",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "built_in_tools": ["apply_patch_async"],
        "workspace_dir": "./my_project_folder",
    },
)

# Create a coding assistant agent
# initialise agents
from autogen.agentchat.group.patterns.auto import AutoPattern


devops_agent = ConversableAgent(
    name="devops_agent",
    llm_config=llm_config,
    system_message="""You are a helpful devops agent
    you will be responsible for devOPS tasks.
    """,
)

qa_agent = ConversableAgent(
    name="qa_agent",
    llm_config=llm_config,
    system_message="""You are a helpful qa agent
    you will be responsible for qa tasks.
    """,
)

coding_agent = ConversableAgent(
    name="coding_assistant",
    llm_config=llm_config,
    system_message="""You are a helpful coding assistant. You can create, edit, and delete files
    using the apply_patch tool. When making changes, always use the apply_patch tool rather than
    writing raw code blocks. Be precise with your file operations and explain what you're doing.""",
)

review_agent = ConversableAgent(
    name="review_agent",
    llm_config=llm_config,
    system_message="""
    you are a review agent. you will review changes made by coding agent
    """,
)
# create a groupchat


pattern = AutoPattern(
    initial_agent=coding_agent,
    agents=[devops_agent, qa_agent, review_agent, coding_agent],
    group_manager_args={"llm_config": llm_config}
)

from autogen.agentchat import initiate_group_chat


result, context, last_agent = initiate_group_chat(
    pattern=pattern,
    messages="""
    create a project.py file
    - in project.py add calculator code
    - create a tests folder
    - create a tests/test_main.py file
    - add .yaml file to project.py
    - add .sh file to project.py
    - add .gitignore file to project.py
    - add .md file to project.py
    review the code and make sure it is correct
    add test to the project.
    """,
    max_rounds=10
)