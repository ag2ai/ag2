# Example: Multimodal NVIDIA hosted model with AG2's OpenAI compatible client
# Needs an NVIDIA API key from build.nvidia.com
# pip install ag2[openai]

import base64
import mimetypes
import os
from pathlib import Path

from dotenv import load_dotenv

from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent
from autogen.events.agent_events import TextEvent
from autogen.llm_config.config import LLMConfig

# NVIDIA_API_KEY stored in .env
load_dotenv(Path(__file__).parent / ".env")


# Take a file path, read the file, and convert to a base64-encoded data URI string that can be sent in a message for the MultiModalConversableAgent
def image_to_base64_data_uri(file_path: str) -> str:
    """Read an image file and return it as a base64-encoded data URI string."""
    path = Path(file_path)
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type is None:
        mime_type = "application/octet-stream"
    data = path.read_bytes()
    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


base64_encoded_image = image_to_base64_data_uri(Path(__file__).parent / "automatic-speech-recognition-diagram.png")

# Use OpenAI client as NVIDIA endpoints are OpenAI API compatible
llm_config = LLMConfig(
    {
        "api_type": "openai",
        "model": "moonshotai/kimi-k2.5",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key": os.environ["NVIDIA_API_KEY"],
    },
    temperature=1.0,
    top_p=0.95,
)

vision_agent = MultimodalConversableAgent(
    name="vision_agent",
    llm_config=llm_config,
)

response = vision_agent.run(
    message=f"What do you see? <img {base64_encoded_image}>",
    max_turns=1,
)

for event in response.events:
    if isinstance(event, TextEvent):
        if event.content.sender == "user":
            print(f"User:\n{event.content.content[0:100]}...")
        else:
            print(f"Vision Agent:\n{event.content.content}")
