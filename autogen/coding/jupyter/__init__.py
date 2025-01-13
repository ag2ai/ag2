# Copyright (c) 2023 - 2024, Owners of https://github.com/autogen-ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Original portions of this file are derived from https://github.com/microsoft/autogen under the MIT License.
# SPDX-License-Identifier: MIT
from .helpers import is_jupyter_kernel_gateway_installed

if not is_jupyter_kernel_gateway_installed():
    raise ImportError(
        "jupyter-kernel-gateway is required for JupyterCodeExecutor, please install it with `pip install ag2[jupyter-executor]`"
    )


from .base import JupyterConnectable, JupyterConnectionInfo
from .docker_jupyter_server import DockerJupyterServer
from .embedded_ipython_code_executor import EmbeddedIPythonCodeExecutor
from .jupyter_client import JupyterClient
from .jupyter_code_executor import JupyterCodeExecutor
from .local_jupyter_server import LocalJupyterServer

__all__ = [
    "DockerJupyterServer",
    "EmbeddedIPythonCodeExecutor",
    "JupyterClient",
    "JupyterCodeExecutor",
    "JupyterConnectable",
    "JupyterConnectionInfo",
    "LocalJupyterServer",
]
