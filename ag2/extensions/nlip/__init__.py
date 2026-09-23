# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from ag2.exceptions import missing_additional_dependency

# Imported twice on purpose: the checker is shown only the real import, because
# rebinding a name it has already bound to a class is an error, and the install hint
# is runtime behaviour. See website/docs/contributor-guide/type-checking.mdx.
if TYPE_CHECKING:
    from .config import NlipConfig
    from .server import NlipServer
else:
    try:
        from .config import NlipConfig
    except ImportError as e:
        NlipConfig = missing_additional_dependency("NlipConfig", 'nlip-sdk>=0.1.0,<1" "nlip-server>=0.1.3,<1', e)

    try:
        from .server import NlipServer
    except ImportError as e:
        NlipServer = missing_additional_dependency("NlipServer", 'nlip-sdk>=0.1.0,<1" "nlip-server>=0.1.3,<1', e)

__all__ = (
    "NlipConfig",
    "NlipServer",
)
