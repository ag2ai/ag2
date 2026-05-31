# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Original portions of this file are derived from https://github.com/microsoft/autogen under the MIT License.
# SPDX-License-Identifier: MIT
import logging

from .base import CodeBlock, CodeExecutor, CodeExtractor, CodeResult
from .factory import CodeExecutorFactory
from .local_commandline_code_executor import LocalCommandLineCodeExecutor
from .markdown_code_extractor import MarkdownCodeExtractor

logger = logging.getLogger(__name__)

__all__ = [
    "CodeBlock",
    "CodeExecutor",
    "CodeExecutorFactory",
    "CodeExtractor",
    "CodeResult",
    "LocalCommandLineCodeExecutor",
    "MarkdownCodeExtractor",
]

# Try to import Docker executor and add to __all__ if available
try:
    from .docker_commandline_code_executor import DockerCommandLineCodeExecutor  # noqa: F401

    __all__.append("DockerCommandLineCodeExecutor")
except ImportError:
    logger.debug(
        "DockerCommandLineCodeExecutor not available: missing dependencies. Install with: pip install ag2[docker]"
    )
    pass

# Try to import YepCode executor and add to __all__ if available
try:
    from .yepcode_code_executor import YepCodeCodeExecutor, YepCodeCodeResult  # noqa: F401

    __all__.extend(["YepCodeCodeExecutor", "YepCodeCodeResult"])
except ImportError:
    pass

# Try to import Remyx executor and add to __all__ if available
try:
    from .remyx_code_executor import RemyxCodeExecutor, RemyxCodeResult  # noqa: F401

    __all__.extend(["RemyxCodeExecutor", "RemyxCodeResult"])
except ImportError:
    logger.debug("RemyxCodeExecutor not available: missing dependencies. Install with: pip install ag2[remyx]")
    pass

# Try to import Daytona executor and add to __all__ if available
try:
    from .daytona_code_executor import DaytonaCodeExecutor, DaytonaCodeResult, DaytonaSandboxResources  # noqa: F401

    __all__.extend(["DaytonaCodeExecutor", "DaytonaCodeResult", "DaytonaSandboxResources"])
except ImportError:
    logger.debug("DaytonaCodeExecutor not available: missing dependencies. Install with: pip install ag2[daytona]")
    pass
