# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Zenrows scrape and extract extension for AG2.

Zenrows is a web scraping API. It fetches one page through rotating proxies with
anti-bot bypass and returns it as markdown, plaintext or HTML, as named fields
chosen by CSS selectors, or as the structured data its extractor already knows how
to read on that domain. This extension wraps that single endpoint as two tools, one
for reading a page and one for its structured data.

Maintainer: Zenrows (@ZenRows)
Docs: https://docs.ag2.ai/docs/user-guide/extensions/tools/search/zenrows/
"""

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal, NoReturn, TypeAlias
from urllib.parse import urlparse

import httpx
from pydantic import Field

from ag2.annotations import Context, Variable
from ag2.events import ToolResult
from ag2.middleware import ToolMiddleware
from ag2.tools.builtin._resolve import resolve_variable
from ag2.tools.final import Toolkit, tool
from ag2.tools.final.function_tool import FunctionTool

ResponseType: TypeAlias = Literal["markdown", "plaintext", "html"]
RefusalReason: TypeAlias = Literal[
    "unsupported_url_scheme",
    "domain_not_prepared",
    "page_not_recognised",
    "domain_not_enabled",
    "not_extractable",
    "no_data_extracted",
]

_API_URL = "https://api.zenrows.com/v1/"
_API_KEY_ENV_VAR = "ZENROWS_API_KEY"  # pragma: allowlist secret
_USER_AGENT = "ag2-zenrows-extension"
_SAFE_URL_SCHEMES = {"http", "https"}
_DEFAULT_MAX_CHARACTERS = 20_000

# Every next step names something the model can do now. None of them says to retry:
# a model has no clock, so "retry in a few minutes" becomes an immediate loop.
_NEXT_STEP: dict[RefusalReason, str] = {
    "unsupported_url_scheme": "Call this tool again with an http or https URL.",
    "domain_not_prepared": (
        "Extract does not serve this domain yet. Preparing it is a manual step an operator takes once "
        "in the Zenrows dashboard. Read the page with zenrows_scrape instead, passing css_extractor "
        "with a CSS selector per field you need."
    ),
    "page_not_recognised": (
        "Extract serves this domain but not this page. Ask for a product, article or listing page on the "
        "same domain, or read this page with zenrows_scrape, passing css_extractor with a CSS selector "
        "per field you need."
    ),
    "domain_not_enabled": (
        "Extract is not enabled for this domain, and preparing the domain does not change that. Read the "
        "page with zenrows_scrape instead, passing css_extractor with a CSS selector per field you need."
    ),
    "not_extractable": (
        "Extract could not read this page. Read it with zenrows_scrape instead, passing css_extractor "
        "with a CSS selector per field you need."
    ),
    "no_data_extracted": (
        "Extract read this page and filled no field. Ask for a page of the same kind that carries the data, "
        "or read this one with zenrows_scrape, passing css_extractor with a CSS selector per field you need."
    ),
}

_REQS007_REASONS: dict[str, RefusalReason] = {
    "domain_not_prepared": "domain_not_prepared",
    "page_not_recognised": "page_not_recognised",
    "domain_not_enabled": "domain_not_enabled",
}

# `reason` arrives with an upcoming Zenrows release. Until then the documented way to
# tell the three gate cases apart is the `title` prefix, so both are read.
_REQS007_TITLES: tuple[tuple[str, RefusalReason], ...] = (
    ("Extract is not prepared for the requested domain", "domain_not_prepared"),
    ("Extract did not recognise this page on", "page_not_recognised"),
    ("Extract is not enabled for the requested domain", "domain_not_enabled"),
)


@dataclass(slots=True)
class ZenRowsPage:
    """One page read as text."""

    url: str
    format: str
    content: str
    truncated: bool = False
    request_id: str | None = None


@dataclass(slots=True)
class ZenRowsData:
    """Named fields read from one page, by CSS selector, by output, or by Extract."""

    url: str
    data: dict[str, Any]
    truncated: bool = False
    request_id: str | None = None


@dataclass(slots=True)
class ZenRowsRefusal:
    """A verdict on the URL the model chose, with the one step that can change it."""

    url: str
    reason: RefusalReason
    next_step: str
    detail: str = ""
    request_id: str | None = None


def _http_url(url: str) -> bool:
    return urlparse(url).scheme.lower() in _SAFE_URL_SCHEMES


def _text(raw: Mapping[str, Any], key: str) -> str:
    """Return ``raw[key]`` when it holds a string there, else ``""``."""
    value = raw.get(key)
    return value if isinstance(value, str) else ""


def _as_csv(value: Sequence[str] | str) -> str:
    """Render an outputs list as the comma separated string the API takes."""
    return value if isinstance(value, str) else ",".join(value)


def _bound_text(text: str, max_characters: int | None) -> tuple[str, bool]:
    if max_characters is None or len(text) <= max_characters:
        return text, False
    return text[:max_characters], True


def _bound_data(data: dict[str, Any], max_characters: int | None) -> tuple[dict[str, Any], bool]:
    """Keep leading entries while their serialized length fits, so the result stays parseable."""
    if max_characters is None:
        return data, False
    kept: dict[str, Any] = {}
    used = 0
    for key, value in data.items():
        used += len(json.dumps({key: value}, ensure_ascii=False))
        if used > max_characters:
            return kept, True
        kept[key] = value
    return kept, False


def _literal(value: Any) -> Any:
    """Blank a ``Variable`` so a build time check sees only what is already known."""
    return None if isinstance(value, Variable) else value


def _deferred(**values: Any) -> frozenset[str]:
    """Name the arguments that are a ``Variable`` at this call site."""
    return frozenset(name for name, value in values.items() if isinstance(value, Variable))


def _check_params(
    *,
    response_type: Any,
    outputs: Any,
    js_render: Any,
    premium_proxy: Any,
    proxy_country: Any,
    deferred: frozenset[str],
) -> None:
    """Reject the combinations the Zenrows API rejects, before a request is billed.

    ``deferred`` names the arguments still carrying a ``Variable``. A rule that reads
    "A and B cannot both be set" is safe with those blanked to ``None``; a rule that
    reads "B requires A" is not, because ``None`` is then ambiguous between unset and
    not yet knowable, so that rule is skipped instead.
    """
    if response_type is not None and outputs is not None:
        raise ValueError("response_type and outputs cannot both be set")
    if proxy_country is None or deferred & {"js_render", "premium_proxy", "proxy_country"}:
        return
    # Adaptive Stealth Mode allows proxy_country on its own. Setting js_render or
    # premium_proxy turns that mode off, and the plain API wants premium_proxy with it.
    if (js_render is not None or premium_proxy is not None) and not premium_proxy:
        raise ValueError("proxy_country requires premium_proxy once js_render or premium_proxy is set")


def _request_params(
    *,
    url: str,
    response_type: str | None,
    outputs: Sequence[str] | str | None,
    css_extractor: Mapping[str, str] | None,
    js_render: bool | None,
    premium_proxy: bool | None,
    proxy_country: str | None,
    extract: bool,
) -> dict[str, str]:
    params: dict[str, str] = {"url": url}
    if extract:
        params["extract"] = "auto"
    elif response_type is None and outputs is None and css_extractor is None:
        response_type = "markdown"
    # HTML is the API's own default and has no response_type value of its own.
    if response_type is not None and response_type != "html":
        params["response_type"] = response_type
    if outputs is not None:
        params["outputs"] = _as_csv(outputs)
    if css_extractor is not None:
        params["css_extractor"] = json.dumps(dict(css_extractor), ensure_ascii=False)
    if js_render is not None:
        params["js_render"] = "true" if js_render else "false"
    if premium_proxy is not None:
        params["premium_proxy"] = "true" if premium_proxy else "false"
    if proxy_country is not None:
        params["proxy_country"] = proxy_country
    # Adaptive Stealth Mode picks rendering and proxies per request and bills only for
    # what succeeded. The API rejects it alongside either flag, so it goes only when
    # the operator has set neither.
    if js_render is None and premium_proxy is None:
        params["mode"] = "auto"
    return params


async def _get(api_key: str | None, api_url: str, timeout: float, params: dict[str, str]) -> httpx.Response:
    """Run one request. The key travels as a header, so it is in no URL and no error."""
    key = api_key or os.environ.get(_API_KEY_ENV_VAR)
    headers = {"User-Agent": _USER_AGENT}
    if key:
        headers["X-API-Key"] = key
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        return await client.get(api_url, params=params, headers=headers)


def _problem(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def _json_object(response: httpx.Response) -> dict[str, Any]:
    body = response.json()
    if not isinstance(body, dict):
        raise ValueError("Zenrows returned a JSON value that is not an object")
    return body


def _raise(response: httpx.Response) -> NoReturn:
    """Raise the API's own message. ``raise_for_status`` is avoided: it embeds the URL."""
    problem = _problem(response)
    message = f"Zenrows API returned HTTP {response.status_code}"
    code = _text(problem, "code")
    if code:
        message += f" ({code})"
    detail = _text(problem, "detail")
    if detail:
        message += f": {detail}"
    request_id = response.headers.get("X-Request-Id")
    if request_id:
        message += f" [request {request_id}]"
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        message += f" (Retry-After: {retry_after})"
    if "X-API-Key" not in response.request.headers:
        message += f". No API key was sent; pass api_key or set {_API_KEY_ENV_VAR}."
    raise httpx.HTTPStatusError(message, request=response.request, response=response)


def _reqs007_reason(problem: Mapping[str, Any]) -> RefusalReason:
    reason = _text(problem, "reason")
    if reason in _REQS007_REASONS:
        return _REQS007_REASONS[reason]
    title = _text(problem, "title")
    for prefix, mapped in _REQS007_TITLES:
        if title.startswith(prefix):
            return mapped
    return "not_extractable"


def _refusal(url: str, reason: RefusalReason, *, detail: str = "", request_id: str | None = None) -> ZenRowsRefusal:
    return ZenRowsRefusal(url=url, reason=reason, next_step=_NEXT_STEP[reason], detail=detail, request_id=request_id)


def _extract_refusal(url: str, response: httpx.Response) -> ZenRowsRefusal | None:
    """Name the Extract verdict behind an error response, or ``None`` when it is not one.

    Every case here is a verdict on the one URL the model chose. None of them is charged,
    an identical call gives an identical answer, and the model is the only party that can
    act on it, so each comes back as a result rather than a raised traceback.
    """
    problem = _problem(response)
    code = _text(problem, "code")
    reason: RefusalReason
    if code == "REQS007":
        reason = _reqs007_reason(problem)
    elif code == "AUTH010" and response.status_code == 402:
        reason = "domain_not_enabled"
    elif code == "no_data_extracted" and response.status_code == 422:
        reason = "no_data_extracted"
    elif code == "RESP003":
        reason = "not_extractable"
    else:
        return None
    return _refusal(
        _final_url(response, url),
        reason,
        detail=_text(problem, "detail"),
        request_id=response.headers.get("X-Request-Id"),
    )


def _final_url(response: httpx.Response, url: str) -> str:
    return response.headers.get("Zr-Final-Url") or url


class ZenRowsToolkit(Toolkit):
    """Toolkit that exposes Zenrows page reading and structured extraction as agent tools.

    The two tools share one Zenrows endpoint:
      - ``zenrows_scrape``: read one URL as markdown, plaintext or HTML, or as named
        fields when the call carries ``css_extractor`` or the tool carries ``outputs``
      - ``zenrows_extract``: read one URL as the structured data Zenrows already knows
        how to pull from that domain

    By default, passing the whole toolkit to an agent registers both tools. To use a
    subset, or to fix per-tool options, call the factory methods and pass the returned
    tools to the agent::

        toolkit = ZenRowsToolkit(api_key=...)

        # both tools
        agent = Agent("a", config=config, tools=[toolkit])

        # only scrape, reading pages as plaintext
        agent = Agent("a", config=config, tools=[toolkit.scrape(response_type="plaintext")])

    ``api_key`` falls back to ``ZENROWS_API_KEY``, read when a tool runs, so building the
    toolkit never needs a key. The key travels as an ``X-API-Key`` header and never as a
    query parameter, no client is kept on the toolkit, and redirects are not followed, so
    the key reaches no URL, no error message and no other origin.

    ``js_render`` and ``premium_proxy`` are operator options and are deliberately not on
    either tool's schema. Both turn Adaptive Stealth Mode off, which picks rendering and
    proxies per request and bills only for the configuration that succeeded; a model
    offered a boolean named ``js_render`` turns it on to be safe and bills every request
    at up to 25 times the base rate.

    ``max_characters`` bounds the content each tool returns and defaults to 20,000.
    Text is cut to that many characters and structured data to whole entries, so the
    data a model reads back is still parseable. Results that were cut carry
    ``truncated=True``. ``None`` turns the bound off.

    Both tools return a ``ZenRowsRefusal`` for a URL that is not http or https, and
    ``zenrows_extract`` also returns one for each verdict Extract passes on the page it
    was given. A refusal carries the one next step that can change the answer. Every
    other problem (a parameter combination the API rejects, a missing or invalid key,
    and any other API or transport failure) is raised.
    """

    __slots__ = ("_api_key", "_api_url", "_timeout")

    def __init__(
        self,
        api_key: str | None = None,
        *,
        api_url: str = _API_URL,
        timeout: float = 60.0,
        response_type: ResponseType | Variable | None = None,
        outputs: Sequence[str] | Variable | None = None,
        js_render: bool | Variable | None = None,
        premium_proxy: bool | Variable | None = None,
        proxy_country: str | Variable | None = None,
        max_characters: int | Variable | None = _DEFAULT_MAX_CHARACTERS,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        self._api_key = api_key
        self._api_url = api_url
        self._timeout = timeout

        super().__init__(
            self.scrape(
                response_type=response_type,
                outputs=outputs,
                js_render=js_render,
                premium_proxy=premium_proxy,
                proxy_country=proxy_country,
                max_characters=max_characters,
            ),
            self.extract(
                js_render=js_render,
                premium_proxy=premium_proxy,
                proxy_country=proxy_country,
                max_characters=max_characters,
            ),
            name="zenrows_toolkit",
            middleware=middleware,
        )

    def scrape(
        self,
        *,
        response_type: ResponseType | Variable | None = None,
        outputs: Sequence[str] | Variable | None = None,
        js_render: bool | Variable | None = None,
        premium_proxy: bool | Variable | None = None,
        proxy_country: str | Variable | None = None,
        max_characters: int | Variable | None = _DEFAULT_MAX_CHARACTERS,
        name: str = "zenrows_scrape",
        description: str = (
            "Read one web page through Zenrows and return its content, as markdown by default. "
            "Pass css_extractor to get named fields from the page instead of its whole content."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        """Create the ``zenrows_scrape`` tool.

        Args:
            response_type: How to return the page: ``"markdown"``, ``"plaintext"`` or
                ``"html"``. Defaults to ``"markdown"``. Cannot be combined with ``outputs``.
            outputs: Named Zenrows outputs to return instead of the page content, for
                example ``["headings", "links"]``. Cannot be combined with ``response_type``.
            js_render: Render the page in a browser. Turns Adaptive Stealth Mode off and
                bills at a higher rate, so leave it unset unless a page needs it.
            premium_proxy: Use residential proxies. Turns Adaptive Stealth Mode off and
                bills at a higher rate.
            proxy_country: Two letter country code to proxy from, for example ``"us"``.
                Needs ``premium_proxy`` once ``js_render`` or ``premium_proxy`` is set.
            max_characters: Cut the returned content to this many characters and set
                ``truncated=True``. ``None`` turns the bound off.
            name: Tool name exposed to the model.
            description: Tool description exposed to the model.
            middleware: Tool middleware applied to this tool.

        Raises:
            ValueError: If ``response_type`` and ``outputs`` are both set, or if
                ``proxy_country`` is set without ``premium_proxy`` while either flag is set.
        """
        api_key = self._api_key
        api_url = self._api_url
        timeout = self._timeout
        _check_params(
            response_type=_literal(response_type),
            outputs=_literal(outputs),
            js_render=_literal(js_render),
            premium_proxy=_literal(premium_proxy),
            proxy_country=_literal(proxy_country),
            deferred=_deferred(
                response_type=response_type,
                outputs=outputs,
                js_render=js_render,
                premium_proxy=premium_proxy,
                proxy_country=proxy_country,
            ),
        )

        @tool(name=name, description=description, middleware=middleware)
        async def zenrows_scrape(
            url: Annotated[str, Field(description="The http or https URL of the page to read.")],
            ctx: Context,
            css_extractor: Annotated[
                Mapping[str, str] | None,
                Field(
                    description=(
                        'CSS selectors keyed by the field name to return, for example {"title": "h1"}. '
                        "Returns those named fields instead of the page content."
                    )
                ),
            ] = None,
        ) -> ToolResult:
            """Read one web page through Zenrows."""
            if not _http_url(url):
                return ToolResult(_refusal(url, "unsupported_url_scheme", detail=f"{url!r} is not an http(s) URL."))

            page_format = resolve_variable(response_type, ctx, param_name="response_type")
            resolved_outputs = resolve_variable(outputs, ctx, param_name="outputs")
            resolved_js_render = resolve_variable(js_render, ctx, param_name="js_render")
            resolved_premium = resolve_variable(premium_proxy, ctx, param_name="premium_proxy")
            resolved_country = resolve_variable(proxy_country, ctx, param_name="proxy_country")
            limit = resolve_variable(max_characters, ctx, param_name="max_characters")
            _check_params(
                response_type=page_format,
                outputs=resolved_outputs,
                js_render=resolved_js_render,
                premium_proxy=resolved_premium,
                proxy_country=resolved_country,
                deferred=frozenset(),
            )

            params = _request_params(
                url=url,
                response_type=page_format,
                outputs=resolved_outputs,
                css_extractor=css_extractor,
                js_render=resolved_js_render,
                premium_proxy=resolved_premium,
                proxy_country=resolved_country,
                extract=False,
            )
            response = await _get(api_key, api_url, timeout, params)
            if not response.is_success:
                _raise(response)

            request_id = response.headers.get("X-Request-Id")
            # Which result type comes back is decided by the request that was built,
            # never by looking at the response.
            if resolved_outputs is not None or css_extractor is not None:
                data, cut = _bound_data(_json_object(response), limit)
                return ToolResult(
                    ZenRowsData(url=_final_url(response, url), data=data, truncated=cut, request_id=request_id)
                )
            content, cut = _bound_text(response.text, limit)
            return ToolResult(
                ZenRowsPage(
                    url=_final_url(response, url),
                    format=page_format or "markdown",
                    content=content,
                    truncated=cut,
                    request_id=request_id,
                )
            )

        return zenrows_scrape

    def extract(
        self,
        *,
        js_render: bool | Variable | None = None,
        premium_proxy: bool | Variable | None = None,
        proxy_country: str | Variable | None = None,
        max_characters: int | Variable | None = _DEFAULT_MAX_CHARACTERS,
        name: str = "zenrows_extract",
        description: str = (
            "Read one web page through Zenrows and return its structured data, such as the fields of a "
            "product, an article or a listing. Returns a refusal naming the next step when Zenrows "
            "cannot extract that page."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        """Create the ``zenrows_extract`` tool.

        Args:
            js_render: Render the page in a browser. Turns Adaptive Stealth Mode off and
                bills at a higher rate, so leave it unset unless a page needs it.
            premium_proxy: Use residential proxies. Turns Adaptive Stealth Mode off and
                bills at a higher rate.
            proxy_country: Two letter country code to proxy from, for example ``"us"``.
                Needs ``premium_proxy`` once ``js_render`` or ``premium_proxy`` is set.
            max_characters: Cut the returned data to whole entries fitting in this many
                characters and set ``truncated=True``. ``None`` turns the bound off.
            name: Tool name exposed to the model.
            description: Tool description exposed to the model.
            middleware: Tool middleware applied to this tool.

        Raises:
            ValueError: If ``proxy_country`` is set without ``premium_proxy`` while
                either flag is set.
        """
        api_key = self._api_key
        api_url = self._api_url
        timeout = self._timeout
        _check_params(
            response_type=None,
            outputs=None,
            js_render=_literal(js_render),
            premium_proxy=_literal(premium_proxy),
            proxy_country=_literal(proxy_country),
            deferred=_deferred(js_render=js_render, premium_proxy=premium_proxy, proxy_country=proxy_country),
        )

        @tool(name=name, description=description, middleware=middleware)
        async def zenrows_extract(
            url: Annotated[str, Field(description="The http or https URL of the page to extract.")],
            ctx: Context,
        ) -> ToolResult:
            """Read one web page's structured data through Zenrows."""
            if not _http_url(url):
                return ToolResult(_refusal(url, "unsupported_url_scheme", detail=f"{url!r} is not an http(s) URL."))

            resolved_js_render = resolve_variable(js_render, ctx, param_name="js_render")
            resolved_premium = resolve_variable(premium_proxy, ctx, param_name="premium_proxy")
            resolved_country = resolve_variable(proxy_country, ctx, param_name="proxy_country")
            limit = resolve_variable(max_characters, ctx, param_name="max_characters")
            _check_params(
                response_type=None,
                outputs=None,
                js_render=resolved_js_render,
                premium_proxy=resolved_premium,
                proxy_country=resolved_country,
                deferred=frozenset(),
            )

            params = _request_params(
                url=url,
                response_type=None,
                outputs=None,
                css_extractor=None,
                js_render=resolved_js_render,
                premium_proxy=resolved_premium,
                proxy_country=resolved_country,
                extract=True,
            )
            response = await _get(api_key, api_url, timeout, params)
            if not response.is_success:
                refusal = _extract_refusal(url, response)
                if refusal is None:
                    _raise(response)
                return ToolResult(refusal)

            # A success body carries the whole page source under `html` alongside `parsed`.
            # Returning it would put a full page in the model's context on every call, which
            # is the opposite of why anyone asks for structured data.
            parsed = _json_object(response).get("parsed")
            data, cut = _bound_data(parsed if isinstance(parsed, dict) else {}, limit)
            return ToolResult(
                ZenRowsData(
                    url=_final_url(response, url),
                    data=data,
                    truncated=cut,
                    request_id=response.headers.get("X-Request-Id"),
                )
            )

        return zenrows_extract
