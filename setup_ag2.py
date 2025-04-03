# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

# this file is autogenerated, please do not edit it directly
# instead, edit the corresponding setup.jinja file and run the ./scripts/build-setup-files.py script

import os

import setuptools

here = os.path.abspath(os.path.dirname(__file__))

with open("README.md", "r", encoding="UTF-8") as fh:
    long_description = fh.read()

# Get the code version
version = {}
with open(os.path.join(here, "autogen/version.py")) as fp:
    exec(fp.read(), version)
__version__ = version["__version__"]

setuptools.setup(
    name="ag2",
    version=__version__,
    description="Alias package for pyautogen",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=["pyautogen==" + __version__],
    extras_require={
        "flaml": ["pyautogen[flaml]==" + __version__],
        "openai": ["pyautogen[openai]==" + __version__],
        "openai-realtime": ["pyautogen[openai-realtime]==" + __version__],
        "jupyter-executor": ["pyautogen[jupyter-executor]==" + __version__],
        "retrievechat": ["pyautogen[retrievechat]==" + __version__],
        "retrievechat-pgvector": ["pyautogen[retrievechat-pgvector]==" + __version__],
        "retrievechat-mongodb": ["pyautogen[retrievechat-mongodb]==" + __version__],
        "retrievechat-qdrant": ["pyautogen[retrievechat-qdrant]==" + __version__],
        "retrievechat-couchbase": ["pyautogen[retrievechat-couchbase]==" + __version__],
        "graph-rag-falkor-db": ["pyautogen[graph-rag-falkor-db]==" + __version__],
        "rag": ["pyautogen[rag]==" + __version__],
        "crawl4ai": ["pyautogen[crawl4ai]==" + __version__],
        "browser-use": ["pyautogen[browser-use]==" + __version__],
        "google-client": ["pyautogen[google-client]==" + __version__],
        "google-api": ["pyautogen[google-api]==" + __version__],
        "google-search": ["pyautogen[google-search]==" + __version__],
        "wikipedia": ["pyautogen[wikipedia]==" + __version__],
        "neo4j": ["pyautogen[neo4j]==" + __version__],
        "twilio": ["pyautogen[twilio]==" + __version__],
        "mcp": ["pyautogen[mcp]==" + __version__],
        "interop-crewai": ["pyautogen[interop-crewai]==" + __version__],
        "interop-langchain": ["pyautogen[interop-langchain]==" + __version__],
        "interop-pydantic-ai": ["pyautogen[interop-pydantic-ai]==" + __version__],
        "interop": ["pyautogen[interop]==" + __version__],
        "autobuild": ["pyautogen[autobuild]==" + __version__],
        "blendsearch": ["pyautogen[blendsearch]==" + __version__],
        "mathchat": ["pyautogen[mathchat]==" + __version__],
        "captainagent": ["pyautogen[captainagent]==" + __version__],
        "teachable": ["pyautogen[teachable]==" + __version__],
        "lmm": ["pyautogen[lmm]==" + __version__],
        "graph": ["pyautogen[graph]==" + __version__],
        "gemini": ["pyautogen[gemini]==" + __version__],
        "gemini-realtime": ["pyautogen[gemini-realtime]==" + __version__],
        "together": ["pyautogen[together]==" + __version__],
        "websurfer": ["pyautogen[websurfer]==" + __version__],
        "redis": ["pyautogen[redis]==" + __version__],
        "cosmosdb": ["pyautogen[cosmosdb]==" + __version__],
        "websockets": ["pyautogen[websockets]==" + __version__],
        "long-context": ["pyautogen[long-context]==" + __version__],
        "anthropic": ["pyautogen[anthropic]==" + __version__],
        "cerebras": ["pyautogen[cerebras]==" + __version__],
        "mistral": ["pyautogen[mistral]==" + __version__],
        "groq": ["pyautogen[groq]==" + __version__],
        "cohere": ["pyautogen[cohere]==" + __version__],
        "ollama": ["pyautogen[ollama]==" + __version__],
        "bedrock": ["pyautogen[bedrock]==" + __version__],
        "deepseek": ["pyautogen[deepseek]==" + __version__],
        "commsagent-discord": ["pyautogen[commsagent-discord]==" + __version__],
        "commsagent-slack": ["pyautogen[commsagent-slack]==" + __version__],
        "commsagent-telegram": ["pyautogen[commsagent-telegram]==" + __version__],
        "test": ["pyautogen[test]==" + __version__],
        "docs": ["pyautogen[docs]==" + __version__],
        "types": ["pyautogen[types]==" + __version__],
        "lint": ["pyautogen[lint]==" + __version__],
        "dev": ["pyautogen[dev]==" + __version__],

    },
    url="https://github.com/ag2ai/ag2",
    author="Chi Wang & Qingyun Wu",
    author_email="support@ag2.ai",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    license="Apache Software License 2.0",
    python_requires=">=3.9,<3.14",
)
