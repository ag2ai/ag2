# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/https://github.com/Lancetnik/FastDepends are under the MIT License.
# SPDX-License-Identifier: MIT
import warnings

from .dependencies import Provider, dependency_provider
from .use import Depends, inject

warnings.warn(
    "The 'autogen.fast_depends' package is deprecated and will be removed in version 0.11. "
    "Please use the regular 'fast_depends' library instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = (
    "Depends",
    "Provider",
    "dependency_provider",
    "inject",
)
