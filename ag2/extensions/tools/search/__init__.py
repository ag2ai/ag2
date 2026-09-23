# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from ag2.exceptions import missing_additional_dependency

# Imported twice on purpose: the checker is shown only the real import, because
# rebinding a name it has already bound to a class is an error, and the install hint
# is runtime behaviour. See website/docs/contributor-guide/type-checking.mdx.
if TYPE_CHECKING:
    from .exa import ExaToolkit
    from .tinyfish import TinyFishSearchToolkit
else:
    try:
        from .exa import ExaToolkit
    except ImportError as e:
        ExaToolkit = missing_additional_dependency("ExaToolkit", "exa-py>=2.12.1,<3", e)

    try:
        from .tinyfish import TinyFishSearchToolkit
    except ImportError as e:
        TinyFishSearchToolkit = missing_additional_dependency("TinyFishSearchToolkit", "tinyfish>=0.2.3", e)

from .serply import SerplySearchToolkit
from .xquik import XquikSearchToolkit

__all__ = (
    "ExaToolkit",
    "SerplySearchToolkit",
    "TinyFishSearchToolkit",
    "XquikSearchToolkit",
)
