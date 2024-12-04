"""Interactive manual tool execution for MultimodalWebSurfer."""

import asyncio
import os
import json
import logging

from autogen.agentchat.contrib.magentic_one.websurfer import MultimodalWebSurfer
from autogen.agentchat.user_proxy_agent import UserProxyAgent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

async def main() -> None:
    # Set up LLM configuration
    config_list = [
        {
            "model": os.environ.get("MODEL"),
            "api_key": os.environ.get("API_KEY"),
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

    # Initialize the browser
    await websurfer.init(
        headless=False,  # Set to False to see browser interactions
        downloads_folder="./downloads",
        start_page="https://www.bing.com",
        browser_channel="chromium",
        to_save_screenshots=True
    )

    # Initial URL
    initial_url = input("Enter initial URL to visit (default: https://www.bing.com): ") or "https://www.bing.com"
    await websurfer._visit_page(initial_url)

    # Interactive tool execution loop
    while True:
        try:
            # Prompt for tool name
            print("\nAvailable tools:")
            print("1. visit_url")
            print("2. web_search")
            print("3. click")
            print("4. input_text")
            print("5. summarize_page")
            print("6. page_up")
            print("7. page_down")
            print("8. history_back")
            print("9. Exit")

            tool_choice = input("Enter tool number: ")
            
            if tool_choice == '9':
                break

            # Prepare tool arguments based on selection
            tool_map = {
                '1': 'visit_url',
                '2': 'web_search',
                '3': 'click',
                '4': 'input_text',
                '5': 'summarize_page',
                '6': 'page_up',
                '7': 'page_down',
                '8': 'history_back'
            }

            tool_name = tool_map.get(tool_choice)
            if not tool_name:
                print("Invalid tool selection.")
                continue

            # Get tool arguments dynamically
            tool_args = {}
            if tool_name in ['visit_url', 'web_search']:
                tool_args['url' if tool_name == 'visit_url' else 'query'] = input(f"Enter {tool_name} argument: ")
            elif tool_name in ['click', 'input_text']:
                tool_args['target_id'] = input("Enter target element ID: ")
                if tool_name == 'input_text':
                    tool_args['text_value'] = input("Enter text to input: ")

            # Execute the tool
            print(f"\nExecuting {tool_name} with args: {tool_args}")
            request_halt, tool_response = await websurfer.manual_tool_execute(
                tool_name, 
                tool_args
            )

            # Perform set of marks after tool execution
            som_screenshot = await websurfer.test_set_of_mark(websurfer._page.url)
            print("\nSet of Marks screenshot generated.")

            # Print the tool response
            if isinstance(tool_response, dict):
                for content in tool_response.get('content', []):
                    if content['type'] == 'text':
                        print("\nTool Response Text:")
                        print(content['text'])
                    elif content['type'] == 'image_url':
                        print("\nScreenshot saved in debug directory.")
            else:
                print("\nTool Response:", tool_response)

        except Exception as e:
            print(f"Error executing tool: {e}")
            continue

if __name__ == "__main__":
    asyncio.run(main())
