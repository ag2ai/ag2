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

async def main() -> None:
    # Set up LLM configuration
    config_list = [
        {
            "model": os.environ.get("MODEL"),#, "gpt-4-vision-preview"),
            "api_key": os.environ.get("API_KEY"),
            #"base_url": os.environ.get("BASE_URL", None)
        }
    ]

    llm_config = {
        "config_list": config_list,
        "cache_seed": None,
        "temperature": 0,
        "timeout": 120
    }

    # Create the web surfer agent
    websurfer = MultimodalWebSurfer(
        name="WebSurfer",
        human_input_mode="TERMINATE",
        llm_config=llm_config
    )

    user = UserProxyAgent( 
        name="User",
        human_input_mode="NEVER",
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

    marked_screenshot = await websurfer.test_set_of_mark("https://zeit.de")                                                     
    marked_screenshot.save("marked_screenshot.png")                                                                                 

if __name__ == "__main__":
    asyncio.run(main())
