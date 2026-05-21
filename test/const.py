# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

from pathlib import Path

KEY_LOC = str((Path(__file__).parents[1] / "notebook").resolve())
OAI_CONFIG_LIST = Path(__file__).parents[1] / "OAI_CONFIG_LIST"
MOCK_OPEN_AI_API_KEY = "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly"
MOCK_AZURE_API_KEY = "mockazureAPIkeysinexpectedformatsfortestingonly"

reason = "requested to skip"

BLOG_POST_URL = "https://docs.ag2.ai/latest/docs/blog/2023/04/21/LLM-tuning-math/"
BLOG_POST_TITLE = "Does Model and Inference Parameter Matter in LLM Applications? - A Case Study for MATH - AG2"
BLOG_POST_STRING = "Large language models (LLMs) are powerful tools that can generate natural language texts for various applications, such as chatbots, summarization, translation, and more. GPT-4 is currently the state of the art LLM in the world. Is model selection irrelevant? What about inference parameters?"

WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/Microsoft"
WIKIPEDIA_TITLE = "Microsoft - Wikipedia"
WIKIPEDIA_STRING = "Redmond"

PLAIN_TEXT_URL = "https://raw.githubusercontent.com/ag2ai/ag2/main/README.md"
IMAGE_URL = "https://github.com/afourney.png"

PDF_URL = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
PDF_STRING = "Dummy PDF file"

BING_QUERY = "Microsoft"
BING_TITLE = f"{BING_QUERY} - Search"
BING_STRING = f"A Bing search for '{BING_QUERY}' found"
