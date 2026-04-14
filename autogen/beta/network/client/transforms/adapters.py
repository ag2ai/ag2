# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Three MVP transform adapters — Phase 5a.1.

- :class:`NamedTransform` — resolves ``apply: "name"`` against the
  owning :class:`ActorClient`'s :class:`TransformRegistry`.
- :class:`PythonTransform` — imports a dotted-path module, instantiates
  a class with an optional ``config`` dict, calls the instance on
  every envelope.
- :class:`HttpTransform` — POSTs the envelope to a local sidecar URL;
  uses a pooled :class:`httpx.AsyncClient`. ``200`` → mutate,
  ``204`` → pass, ``409`` → reject, anything else → fail-as-reject.

Every adapter runs inside the recipient :class:`ActorClient` — never
in the hub process — so tenant code (especially ``PythonTransform``'s
user module) stays isolated.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

from ...envelope import Envelope
from .protocol import Transform, TransformContext, TransformError
from .registry import TransformRegistry

log = logging.getLogger(__name__)

__all__ = ("HttpTransform", "NamedTransform", "PythonTransform")


# ---------------------------------------------------------------------------
# NamedTransform
# ---------------------------------------------------------------------------


class NamedTransform:
    """Delegate to a named entry in the owning actor's
    :class:`TransformRegistry`.

    The registry lookup happens at construction time (i.e. when the
    pipeline is built in response to :class:`RuleChangedFrame`), so a
    rule pointing at an unknown name fails fast at rule-change time
    rather than when the first envelope flows.
    """

    __slots__ = ("_name", "_delegate")

    def __init__(self, name: str, registry: TransformRegistry) -> None:
        self._name = name
        # ``registry.create`` raises ``TransformLookupError`` for
        # unknown names — propagating is the intended behavior.
        self._delegate = registry.create(name)

    @property
    def name(self) -> str:
        return self._name

    async def __call__(
        self, envelope: Envelope, ctx: TransformContext
    ) -> Envelope | None:
        return await self._delegate(envelope, ctx)

    async def aclose(self) -> None:
        """Forward close to the delegate if it exposes one.

        A named transform's delegate is the long-lived instance from
        the registry factory. Stateful delegates (pooled clients,
        databases) get drained here when the pipeline is rebuilt or
        the :class:`ActorClient` disconnects.
        """

        close = getattr(self._delegate, "aclose", None)
        if close is not None:
            try:
                await close()
            except Exception:  # pragma: no cover
                log.warning(
                    "NamedTransform %s delegate aclose failed",
                    self._name,
                    exc_info=True,
                )


# ---------------------------------------------------------------------------
# PythonTransform
# ---------------------------------------------------------------------------


class PythonTransform:
    """Import a user class by dotted path and call its instances.

    The ``apply`` dict shape:

    .. code-block:: jsonc

        {"python": {"module": "myorg.guards", "class": "PromptGuard",
                    "config": {"max_tokens": 8000}}}

    ``module`` is imported via :func:`importlib.import_module`. The
    class is instantiated once with ``config`` as a single dict
    argument (the user class decides how to accept it — a zero-arg
    constructor is not required but an empty ``config`` is passed as
    ``{}``). The resulting instance is invoked on every envelope as
    an async callable, matching the :class:`Transform` protocol.

    **This adapter only runs inside an :class:`ActorClient`, never in
    the hub.** Even though the rule JSON is stored on the hub, the
    import happens in the recipient's process — the hub never calls
    :func:`importlib.import_module` on tenant data, keeping
    multi-tenant deployments safe.
    """

    __slots__ = ("_instance", "_module_name", "_class_name")

    def __init__(self, apply_dict: dict[str, Any]) -> None:
        module_name = apply_dict.get("module")
        class_name = apply_dict.get("class")
        config = apply_dict.get("config") or {}
        if not isinstance(module_name, str) or not module_name:
            raise TransformError(
                "python transform requires 'module' string"
            )
        if not isinstance(class_name, str) or not class_name:
            raise TransformError(
                "python transform requires 'class' string"
            )
        if not isinstance(config, dict):
            raise TransformError(
                "python transform 'config' must be a dict when present"
            )
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise TransformError(
                f"python transform module {module_name!r} not importable: {exc}"
            ) from exc
        try:
            cls = getattr(module, class_name)
        except AttributeError as exc:
            raise TransformError(
                f"python transform class "
                f"{module_name}.{class_name} not found"
            ) from exc
        try:
            instance = cls(**config) if config else cls()
        except TypeError:
            # Fall back to single-dict constructor for classes that
            # want the whole config as one argument.
            instance = cls(config)
        self._instance = instance
        self._module_name = module_name
        self._class_name = class_name

    @property
    def dotted_path(self) -> str:
        return f"{self._module_name}.{self._class_name}"

    async def __call__(
        self, envelope: Envelope, ctx: TransformContext
    ) -> Envelope | None:
        return await self._instance(envelope, ctx)

    async def aclose(self) -> None:
        """Close the underlying instance if it supports ``aclose``.

        Called by the pipeline on rule-change rebuild and on
        :meth:`ActorClient.disconnect` so long-lived instances
        (database connections, pooled clients) get torn down cleanly.
        """

        close = getattr(self._instance, "aclose", None)
        if close is not None:
            try:
                await close()
            except Exception:  # pragma: no cover
                log.warning(
                    "PythonTransform %s aclose failed",
                    self.dotted_path,
                    exc_info=True,
                )


# ---------------------------------------------------------------------------
# HttpTransform
# ---------------------------------------------------------------------------


class HttpTransform:
    """Delegate to a local HTTP sidecar.

    The ``apply`` form is a single URL string:

    .. code-block:: jsonc

        {"http": "http://localhost:9000/redact"}

    The adapter POSTs the envelope's JSON to that URL with
    ``Content-Type: application/json`` and honors the following
    response contract:

    - ``200 application/json`` — body is the (possibly mutated)
      envelope. Decoded via :meth:`Envelope.from_dict` and returned.
    - ``204 No Content`` — pass-through; the original envelope is
      returned unchanged.
    - ``409 Conflict`` — reject; the pipeline returns ``None`` so
      the ``pre_send`` / ``pre_receive`` path rejects. Response body
      is logged as the reason when available.
    - Any other status — fail-as-reject. The error is logged and the
      pipeline treats the envelope as rejected (matching §4.1's
      "unexpected errors are reject" stance).

    The :class:`httpx.AsyncClient` is lazily constructed on the first
    invocation and reused for every subsequent envelope via the
    configured connection pool — **no per-envelope TCP handshake**.
    """

    _DEFAULT_TIMEOUT_S = 5.0
    _DEFAULT_MAX_KEEPALIVE = 10
    _DEFAULT_MAX_CONNECTIONS = 50

    __slots__ = ("_url", "_client", "_timeout_s")

    def __init__(
        self,
        url: str,
        *,
        timeout_s: float | None = None,
    ) -> None:
        if not isinstance(url, str) or not url:
            raise TransformError("http transform requires a non-empty URL")
        self._url = url
        self._timeout_s = (
            timeout_s if timeout_s is not None else self._DEFAULT_TIMEOUT_S
        )
        self._client: Any = None  # lazily constructed httpx.AsyncClient

    @property
    def url(self) -> str:
        return self._url

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                import httpx
            except ImportError as exc:  # pragma: no cover
                raise TransformError(
                    "httpx is required for HttpTransform; "
                    "install with `pip install httpx`"
                ) from exc
            self._client = httpx.AsyncClient(
                timeout=self._timeout_s,
                limits=httpx.Limits(
                    max_keepalive_connections=self._DEFAULT_MAX_KEEPALIVE,
                    max_connections=self._DEFAULT_MAX_CONNECTIONS,
                ),
            )
        return self._client

    async def __call__(
        self, envelope: Envelope, ctx: TransformContext
    ) -> Envelope | None:
        client = self._ensure_client()
        try:
            response = await client.post(
                self._url,
                json=envelope.to_dict(),
            )
        except Exception as exc:
            log.warning(
                "HttpTransform POST to %s failed at %s: %s",
                self._url,
                ctx.stage.value,
                exc,
            )
            return None
        status = response.status_code
        if status == 204:
            return envelope
        if status == 200:
            try:
                body = response.json()
            except Exception as exc:
                log.warning(
                    "HttpTransform %s returned 200 with invalid JSON: %s",
                    self._url,
                    exc,
                )
                return None
            try:
                return Envelope.from_dict(body)
            except Exception as exc:
                log.warning(
                    "HttpTransform %s 200 body is not a valid envelope: %s",
                    self._url,
                    exc,
                )
                return None
        if status == 409:
            reason = ""
            try:
                reason = response.text
            except Exception:  # pragma: no cover
                pass
            log.info(
                "HttpTransform %s rejected envelope at %s: %s",
                self._url,
                ctx.stage.value,
                reason,
            )
            return None
        log.warning(
            "HttpTransform %s unexpected status %d at %s",
            self._url,
            status,
            ctx.stage.value,
        )
        return None

    async def aclose(self) -> None:
        """Close the pooled :class:`httpx.AsyncClient`.

        Called on pipeline rebuild (rule change) and on
        :meth:`ActorClient.disconnect`. Idempotent.
        """

        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:  # pragma: no cover
                log.warning(
                    "HttpTransform %s aclose failed",
                    self._url,
                    exc_info=True,
                )
            self._client = None
