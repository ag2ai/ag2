# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import subprocess
from logging import getLogger

logger = getLogger(__name__)

__all__ = ["is_jupyter_kernel_gateway_installed"]


def is_jupyter_kernel_gateway_installed() -> bool:
    """Check if jupyter-kernel-gateway is installed."""
    try:
        subprocess.run(
            ["jupyter", "kernelgateway", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning(
            "jupyter-kernel-gateway is required for JupyterCodeExecutor, please install it with `pip install ag2[jupyter-executor]`"
        )
        return False
