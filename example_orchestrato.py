"""This example demonstrates using a web surfer agent to search for and summarize
information about Python programming through an embedded browser."""

import asyncio
import os
from autogen.agentchat.contrib.magentic_one.websurfer import MultimodalWebSurfer
from autogen.agentchat.user_proxy_agent import UserProxyAgent

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# NOTE: Don't forget to 'playwright install --with-deps chromium'

import logging
import os
import sys

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('orchestrator.log')
    ]
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))


from autogen.agentchat.contrib.magentic_one.orchestrator_agent import OrchestratorAgent
from autogen.agentchat import AssistantAgent, UserProxyAgent
import json

async def main() -> None:
    config_list = [
        {
            "model": os.environ.get("MODEL"),
            "api_key": os.environ.get("API_KEY"),
            "base_url": os.environ.get("BASE_URL", None)
        }
    ]

    llm_config = {
        "config_list": config_list,
        "temperature": 0,
        "timeout": 120
    }



    # Create the web surfer agent
    websurfer = MultimodalWebSurfer(
        name="WebSurfer",
        human_input_mode="TERMINATE",
        llm_config=llm_config
    )

    # Initialize the browser
    await websurfer.init(
        headless=True,
        downloads_folder="./downloads",
        start_page="https://www.bing.com",
        browser_channel="chromium",
        to_save_screenshots=True
    )

    coder = AssistantAgent(
        name="coder",
        llm_config=llm_config,
        is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
        system_message="You are a helpful AI coding assistant. You write clean, efficient code and explain your solutions clearly. Return the code to the user. ",
        max_consecutive_auto_reply=10,
        human_input_mode="NEVER",
        code_execution_config={
                "work_dir": "coding",
                "use_docker": False,
            },
        description="coder. Writes and executes code to solve tasks.",
    )


    orchestrator = OrchestratorAgent(
        name="Orchestrator",
        llm_config=llm_config,
        agents=[websurfer],
        max_consecutive_auto_reply=10,
        max_stalls_before_replan=3,  
        max_replans=3, 
        return_final_answer=True,
        description="An orchestrator that manages conversation flow and handles errors."
    )

    task = """Find out some detailed informtion about the magentic one agent from microsoft"""
    #task = """Go to amazon.com and find a good beginner 3d printer, pick any good one for under 300 dollars. """
    massages = [
        {"role": "user", "content": task}]
    result = await orchestrator.a_generate_reply(messages=massages)
        


if __name__ == "__main__":
    asyncio.run(main())
