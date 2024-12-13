# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

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
    name="autogen",
    version=__version__,
    description="Alias package for pyautogen",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=["pyautogen==" + __version__],
    extras_require={
        "test": ["pyautogen[test]==" + __version__],
        "blendsearch": ["pyautogen[blendsearch]==" + __version__],
        "mathchat": ["pyautogen[mathchat]==" + __version__],
        "retrievechat": ["pyautogen[retrievechat]==" + __version__],
        "retrievechat-pgvector": ["pyautogen[retrievechat-pgvector]==" + __version__],
        "retrievechat-mongodb": ["pyautogen[retrievechat-mongodb]==" + __version__],
        "retrievechat-qdrant": ["pyautogen[retrievechat-qdrant]==" + __version__],
        "graph-rag-falkor-db": ["pyautogen[graph-rag-falkor-db]==" + __version__],
        "autobuild": ["pyautogen[autobuild]==" + __version__],
        "captainagent": ["pyautogen[captainagent]==" + __version__],
        "teachable": ["pyautogen[teachable]==" + __version__],
        "lmm": ["pyautogen[lmm]==" + __version__],
        "graph": ["pyautogen[graph]==" + __version__],
        "gemini": ["pyautogen[gemini]==" + __version__],
        "together": ["pyautogen[together]==" + __version__],
        "websurfer": ["pyautogen[websurfer]==" + __version__],
        "redis": ["pyautogen[redis]==" + __version__],
        "cosmosdb": ["pyautogen[cosmosdb]==" + __version__],
        "websockets": ["pyautogen[websockets]==" + __version__],
        "jupyter-executor": ["pyautogen[jupyter-executor]==" + __version__],
        "types": ["pyautogen[types]==" + __version__],
        "long-context": ["pyautogen[long-context]==" + __version__],
        "anthropic": ["pyautogen[anthropic]==" + __version__],
        "cerebras": ["pyautogen[cerebras]==" + __version__],
        "mistral": ["pyautogen[mistral]==" + __version__],
        "groq": ["pyautogen[groq]==" + __version__],
        "cohere": ["pyautogen[cohere]==" + __version__],
        "ollama": ["pyautogen[ollama]==" + __version__],
        "bedrock": ["pyautogen[bedrock]==" + __version__],
        "interop-crewai": ["pyautogen[interop-crewai]==" + __version__],
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
    python_requires=">=3.8,<3.14",
)
