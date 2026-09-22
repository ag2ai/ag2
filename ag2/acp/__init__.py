# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from ag2.exceptions import missing_additional_dependency, missing_optional_dependency

# The fallback rebinds a name mypy has bound to a class, which it rejects; it
# sees only the real import. See website/docs/contributor-guide/type-checking.mdx.
if TYPE_CHECKING:
    from .agent import ACPAgent, PromptContent, SessionObserver, SessionOrigin
    from .auth import AuthProvider, AuthenticationFailedError, StaticTokenAuth
    from .config import ACPConfig, ClaudeCodeConfig, CodexConfig, KiloCodeConfig, OpenCodeConfig
    from .sessions import SessionConfig
    from .tool_gateway import MCPCapabilityError
    from .transport import ACPTransportError
else:
    try:
        from .agent import ACPAgent, PromptContent, SessionObserver, SessionOrigin
        from .auth import AuthProvider, AuthenticationFailedError, StaticTokenAuth
        from .config import ACPConfig, ClaudeCodeConfig, CodexConfig, KiloCodeConfig, OpenCodeConfig
        from .sessions import SessionConfig
        from .tool_gateway import MCPCapabilityError
        from .transport import ACPTransportError
    except ImportError as e:  # pragma: no cover - exercised only when ag2[acp] is absent
        ACPConfig = missing_optional_dependency("ACPConfig", "acp", e)
        ACPTransportError = missing_optional_dependency("ACPTransportError", "acp", e)
        ACPAgent = missing_optional_dependency("ACPAgent", "acp", e)
        PromptContent = missing_optional_dependency("PromptContent", "acp", e)
        AuthProvider = missing_optional_dependency("AuthProvider", "acp", e)
        AuthenticationFailedError = missing_optional_dependency("AuthenticationFailedError", "acp", e)
        ClaudeCodeConfig = missing_optional_dependency("ClaudeCodeConfig", "acp", e)
        CodexConfig = missing_optional_dependency("CodexConfig", "acp", e)
        KiloCodeConfig = missing_optional_dependency("KiloCodeConfig", "acp", e)
        OpenCodeConfig = missing_optional_dependency("OpenCodeConfig", "acp", e)
        MCPCapabilityError = missing_optional_dependency("MCPCapabilityError", "acp", e)
        SessionConfig = missing_optional_dependency("SessionConfig", "acp", e)
        SessionObserver = missing_optional_dependency("SessionObserver", "acp", e)
        SessionOrigin = missing_optional_dependency("SessionOrigin", "acp", e)
        StaticTokenAuth = missing_optional_dependency("StaticTokenAuth", "acp", e)

if TYPE_CHECKING:
    from .remote import ACPRemoteConfig
else:
    try:
        from .remote import ACPRemoteConfig
    except ImportError as e:  # pragma: no cover - exercised only when agent-client-protocol[http] is absent
        ACPRemoteConfig = missing_additional_dependency("ACPRemoteConfig", "agent-client-protocol[http]", e)

__all__ = [
    "ACPAgent",
    "ACPConfig",
    "ACPRemoteConfig",
    "ACPTransportError",
    "AuthProvider",
    "AuthenticationFailedError",
    "ClaudeCodeConfig",
    "CodexConfig",
    "KiloCodeConfig",
    "MCPCapabilityError",
    "OpenCodeConfig",
    "PromptContent",
    "SessionConfig",
    "SessionObserver",
    "SessionOrigin",
    "StaticTokenAuth",
]
