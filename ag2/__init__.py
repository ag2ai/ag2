# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from fast_depends import Depends

from .agent import Agent, AgentReply, AgentRun, KnowledgeConfig, TaskConfig
from .annotations import Context, Inject, Variable
from .events import (
    AudioInput,
    BinaryInput,
    DataInput,
    DocumentInput,
    ImageInput,
    TextInput,
    VideoInput,
)
from .files import FilesAPI
from .middleware import Middleware
from .observers import observer
from .plugin import Plugin
from .response import PromptedSchema, ResponseSchema, response_schema
from .spec import AgentSpec
from .stream import MemoryStream
from .task import Task, TaskInject, TaskSpec
from .tools import ToolResult, Toolkit, tool
from .version import __version__

__all__ = (
    "Agent",
    "AgentReply",
    "AgentRun",
    "AgentSpec",
    "AudioInput",
    "BinaryInput",
    "Context",
    "DataInput",
    "Depends",
    "DocumentInput",
    "FilesAPI",
    "ImageInput",
    "Inject",
    "KnowledgeConfig",
    "MemoryStream",
    "Middleware",
    "Plugin",
    "PromptedSchema",
    "ResponseSchema",
    "Task",
    "TaskConfig",
    "TaskInject",
    "TaskSpec",
    "TextInput",
    "ToolResult",
    "Toolkit",
    "Variable",
    "VideoInput",
    "__version__",
    "observer",
    "response_schema",
    "tool",
)
