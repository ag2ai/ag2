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

# SAMPLE OUTPUT:

# User: What do you see? <img data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnYAAACxCAYAAABJJvDcAAAErWlUWHRY...
# [autogen.oai.client: 02-13 19:24:17] {738} WARNING - Model moonshotai/kimi-k2.5 is not found. The cost will be 0. In your config_list, add field {"price" : [prompt_price_per_1k, completion_token_price_per_1k]} for customized pricing.

# Vision Agent:  I see a diagram illustrating an **Automatic Speech Recognition (ASR)** pipeline. The flow moves from left to right, showing how audio input is converted to text output through several processing stages:

# **Input:**
# - **Audio** (represented by a sound wave icon on the far left)

# **Processing Pipeline:**
# 1. **Custom Vocabulary** - Speech bubble icon with three dots
# 2. **Feature Extraction** - Visual representation of audio waves/spectrogram
# 3. **Acoustic Model** - Stacked layers resembling sound processing units
# 4. **Decoder or N-gram Language Model** - Network of circular nodes (neural network representation)
# 5. **BERT Punctuation Model** - Geometric network diagram (triangle with connections)

# **Output:**
# - **Text** (represented by a document icon on the far right)

# The entire workflow is enclosed in a green-bordered box with a green header labeled "Automatic Speech Recognition." Green arrows indicate the flow direction from audio input through each processing stage to the final text output.
