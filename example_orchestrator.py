# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

import asyncio
import logging
import os
import tempfile

from autogen import ConversableAgent
from autogen.agentchat.contrib.magentic_one.coder_agent import create_coder_agent
from autogen.agentchat.contrib.magentic_one.filesurfer_agent import create_file_surfer
from autogen.agentchat.contrib.magentic_one.orchestrator_agent import OrchestratorAgent
from autogen.agentchat.contrib.magentic_one.websurfer import MultimodalWebSurfer
from autogen.coding import DockerCommandLineCodeExecutor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# NOTE: Don't forget to 'playwright install --with-deps chromium'

import logging
import sys

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("orchestrator.log")],
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))


async def main() -> None:
    config_list = [
        {
            "model": os.environ.get("MODEL"),
            "api_key": os.environ.get("API_KEY"),
            "base_url": os.environ.get("BASE_URL", None),
        }
    ]

    llm_config = {"config_list": config_list, "temperature": 0, "timeout": 120}

    temp_dir = tempfile.TemporaryDirectory()

    executorENV = DockerCommandLineCodeExecutor(
        image="python:3.12-slim",
        timeout=10,
        work_dir=temp_dir.name,
    )

    executor = ConversableAgent(
        name="executor",
        description="""A computer terminal that performs no other action than running Python scripts (provided to it quoted in ```python code blocks), or sh shell scripts (provided to it quoted in ```sh code blocks)""",
        llm_config=False,
        code_execution_config={"executor": executorENV},
        human_input_mode="NEVER",
    )

    # Create the web surfer agent
    websurfer = MultimodalWebSurfer(name="WebSurfer", human_input_mode="TERMINATE", llm_config=llm_config)

    # Initialize the browser
    await websurfer.init(
        headless=True,
        downloads_folder="./downloads",
        start_page="https://www.bing.com",
        browser_channel="chromium",
        to_save_screenshots=True,
    )
    coder = create_coder_agent("Coder", llm_config=llm_config)
    filesurfer = create_file_surfer("FileSurfer", llm_config=llm_config)

    orchestrator = OrchestratorAgent(
        name="Orchestrator",
        llm_config=llm_config,
        agents=[websurfer, coder, filesurfer, executor],
        max_consecutive_auto_reply=10,
        max_stalls_before_replan=3,
        max_replans=3,
        return_final_answer=True,
        description="An orchestrator that manages conversation flow and handles errors.",
    )

    task = """Find out some detailed information about the magentic one agent from microsoft"""
    # task = """Go to amazon.com and find a good beginner 3d printer, pick any good one for under 300 dollars. """
    # task = """write the game sname in python """
    # task = """can you generate the python call to run all return all the prime numbers between 1 and 1000 and return the prime numbers ? (give the code to the Executor to run it, don't ask the user to run it. If it does not work with the executor, tell the user.)"""

    messages = [{"role": "user", "content": task}]

    # result = await coder.a_generate_reply(messages)
    # result = await executor.a_generate_reply(messages)

    result = await orchestrator.a_generate_reply(messages=messages)

    print(result)


if __name__ == "__main__":
    asyncio.run(main())
