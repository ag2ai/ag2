# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import subprocess

__all__ = ["is_jupyter_kernel_gateway_installed"]


def is_jupyter_kernel_gateway_installed() -> bool:
    try:
        subprocess.run(
            ["jupyter-kernel-gateway", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
