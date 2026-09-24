# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from ag2.exceptions import missing_additional_dependency, missing_optional_dependency

# The fallback rebinds a name mypy has bound to a class, which it rejects; it
# sees only the real import. See website/docs/contributor-guide/type-checking.mdx.
if TYPE_CHECKING:
    from .card import build_card
    from .config import A2AConfig
else:
    try:
        from .card import build_card
        from .config import A2AConfig
    except ImportError as e:
        build_card = missing_optional_dependency("build_card", "a2a", e)
        A2AConfig = missing_optional_dependency("A2AConfig", "a2a", e)

if TYPE_CHECKING:
    from .server import A2AServer
else:
    try:
        from .server import A2AServer
    except ImportError as e:
        A2AServer = missing_optional_dependency("A2AServer", "a2a", e)

if TYPE_CHECKING:
    from .transports.grpc import secure_grpc_channel_factory
else:
    try:
        from .transports.grpc import secure_grpc_channel_factory
    except ImportError as e:
        secure_grpc_channel_factory = missing_additional_dependency("secure_grpc_channel_factory", "a2a-sdk[grpc]", e)

__all__ = (
    "A2AConfig",
    "A2AServer",
    "build_card",
    "secure_grpc_channel_factory",
)
