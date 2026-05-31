# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Original portions of this file are derived from https://github.com/microsoft/autogen under the MIT License.
# SPDX-License-Identifier: MIT

import logging

from .base import JupyterConnectable, JupyterConnectionInfo
from .embedded_ipython_code_executor import EmbeddedIPythonCodeExecutor
from .jupyter_client import JupyterClient
from .jupyter_code_executor import JupyterCodeExecutor
from .local_jupyter_server import LocalJupyterServer

logger = logging.getLogger(__name__)

__all__ = [
    "EmbeddedIPythonCodeExecutor",
    "JupyterClient",
    "JupyterCodeExecutor",
    "JupyterConnectable",
    "JupyterConnectionInfo",
    "LocalJupyterServer",
]

# Try to import DockerJupyterServer and add to __all__ if available
try:
    from .docker_jupyter_server import DockerJupyterServer  # noqa: F401

    __all__.append("DockerJupyterServer")
except ImportError:
    logger.debug(
        "DockerJupyterServer not available: missing dependencies. Install with: pip install ag2[docker,jupyter-executor]"
    )
    pass
