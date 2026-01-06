# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import ssl
import typing
from typing import Protocol

from httpx._client import AsyncClient, Client, EventHook
from httpx._config import DEFAULT_LIMITS, DEFAULT_MAX_REDIRECTS, DEFAULT_TIMEOUT_CONFIG, Limits
from httpx._transports.base import AsyncBaseTransport
from httpx._types import AuthTypes, CertTypes, CookieTypes, HeaderTypes, ProxyTypes, QueryParamTypes, TimeoutTypes
from httpx._urls import URL

from autogen.doc_utils import export_module


class ClientFactory(Protocol):
    def __call__(self) -> AsyncClient: ...

    def make_sync(self) -> Client: ...


@export_module("autogen.a2a")
class HttpxClientFactory(ClientFactory):
    """
    An asynchronous HTTP client factory, with connection pooling, HTTP/2, redirects,
    cookie persistence, etc.

    It can be shared between tasks.

    Usage:

    ```python
    >>> factory = HttpxClientFactory()
    >>> async with factory() as client:
    >>>     response = await client.get('https://example.org')
    ```

    **Parameters:**

    * **auth** - *(optional)* An authentication class to use when sending
    requests.
    * **params** - *(optional)* Query parameters to include in request URLs, as
    a string, dictionary, or sequence of two-tuples.
    * **headers** - *(optional)* Dictionary of HTTP headers to include when
    sending requests.
    * **cookies** - *(optional)* Dictionary of Cookie items to include when
    sending requests.
    * **verify** - *(optional)* Either `True` to use an SSL context with the
    default CA bundle, `False` to disable verification, or an instance of
    `ssl.SSLContext` to use a custom context.
    * **http2** - *(optional)* A boolean indicating if HTTP/2 support should be
    enabled. Defaults to `False`.
    * **proxy** - *(optional)* A proxy URL where all the traffic should be routed.
    * **timeout** - *(optional)* The timeout configuration to use when sending
    requests.
    * **limits** - *(optional)* The limits configuration to use.
    * **max_redirects** - *(optional)* The maximum number of redirect responses
    that should be followed.
    * **base_url** - *(optional)* A URL to use as the base when building
    request URLs.
    * **transport** - *(optional)* A transport class to use for sending requests
    over the network.
    * **trust_env** - *(optional)* Enables or disables usage of environment
    variables for configuration.
    * **default_encoding** - *(optional)* The default encoding to use for decoding
    response text, if no charset information is included in a response Content-Type
    header. Set to a callable for automatic character set detection. Default: "utf-8".
    """

    def __init__(
        self,
        *,
        auth: AuthTypes | None = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        verify: ssl.SSLContext | str | bool = True,
        cert: CertTypes | None = None,
        http1: bool = True,
        http2: bool = False,
        proxy: ProxyTypes | None = None,
        mounts: None | (typing.Mapping[str, AsyncBaseTransport | None]) = None,
        timeout: TimeoutTypes = DEFAULT_TIMEOUT_CONFIG,
        follow_redirects: bool = False,
        limits: Limits = DEFAULT_LIMITS,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        event_hooks: None | (typing.Mapping[str, list[EventHook]]) = None,
        base_url: URL | str = "",
        transport: AsyncBaseTransport | None = None,
        trust_env: bool = True,
        default_encoding: str | typing.Callable[[bytes], str] = "utf-8",
        **kwargs: typing.Any,
    ) -> None:
        self.options = {
            "auth": auth,
            "params": params,
            "headers": headers,
            "cookies": cookies,
            "verify": verify,
            "cert": cert,
            "http1": http1,
            "http2": http2,
            "proxy": proxy,
            "mounts": mounts,
            "timeout": timeout,
            "follow_redirects": follow_redirects,
            "limits": limits,
            "max_redirects": max_redirects,
            "event_hooks": event_hooks,
            "base_url": base_url,
            "transport": transport,
            "trust_env": trust_env,
            "default_encoding": default_encoding,
            **kwargs,
        }

    def __call__(self) -> AsyncClient:
        return AsyncClient(**self.options)

    def make_sync(self) -> Client:
        return Client(**self.options)


class EmptyClientFactory(ClientFactory):
    def __call__(self) -> AsyncClient:
        return AsyncClient(timeout=30.0)

    def make_sync(self) -> Client:
        return Client(timeout=30.0)
