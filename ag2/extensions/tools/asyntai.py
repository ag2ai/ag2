# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Asyntai website knowledge extension for AG2.

Asyntai is an AI chatbot for websites. It answers visitors from the website's own
pages, documents and knowledge base, and it captures leads. This extension wraps
the Asyntai REST API as agent tools, so an agent can ask that knowledge base, add
knowledge to it, and read the leads and conversations the chatbot collected.

Maintainer: asyntai
Docs: https://docs.ag2.ai/docs/user-guide/extensions/tools/asyntai/
"""

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Annotated, Any

import httpx
from pydantic import Field

from ag2.annotations import Context, Variable
from ag2.events import ToolResult
from ag2.middleware import ToolMiddleware
from ag2.tools.builtin._resolve import resolve_variable
from ag2.tools.final import Toolkit, tool
from ag2.tools.final.function_tool import FunctionTool

_API_KEY_ENV_VAR = "ASYNTAI_API_KEY"  # pragma: allowlist secret


@dataclass(slots=True)
class AsyntaiAnswer:
    answer: str
    session_id: str = ""


@dataclass(slots=True)
class AsyntaiWebsites:
    websites: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class AsyntaiKnowledgeEntries:
    entries: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class AsyntaiKnowledgeAdded:
    entry_id: str = ""
    title: str = ""
    url: str = ""
    chunks_created: int = 0


@dataclass(slots=True)
class AsyntaiLeads:
    leads: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class AsyntaiConversation:
    session_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)


def _items(raw: dict[str, Any], key: str) -> list[dict[str, Any]]:
    """Return the mappings inside the list at ``raw[key]``, skipping anything else."""
    items = raw.get(key)
    return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def _text(raw: dict[str, Any], key: str) -> str:
    """Return ``raw[key]`` when it holds a string, else ``""``."""
    value = raw.get(key)
    return value if isinstance(value, str) else ""


def _raise_for_status(response: httpx.Response, raw: Any) -> None:
    """Raise ``httpx.HTTPStatusError`` carrying Asyntai's own error message.

    The API key travels in a header, never in the URL, so neither the message
    nor the request URL attached to the error exposes it.
    """
    if response.is_success:
        return

    detail = _text(raw, "error") if isinstance(raw, dict) else ""
    message = f"Asyntai API returned HTTP {response.status_code}"
    if detail:
        message += f": {detail}"
    raise httpx.HTTPStatusError(message, request=response.request, response=response)


def _decode(response: httpx.Response) -> Any:
    """Decode a JSON body, returning ``None`` when the body is not JSON."""
    try:
        return response.json()
    except ValueError:
        return None


async def _request(
    client_kwargs: dict[str, Any],
    api_key: str,
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one request against the Asyntai REST API.

    Args:
        client_kwargs: Keyword arguments for the ``httpx.AsyncClient`` to open.
        api_key: Asyntai API key, sent as a bearer token.
        method: HTTP method.
        path: Endpoint path, resolved against the client's base URL.
        params: Query parameters. Entries with a ``None`` value are dropped.
        body: JSON request body. Entries with a ``None`` value are dropped.

    Returns:
        The decoded JSON object, or an empty dict when the payload is not an object.

    Raises:
        httpx.HTTPStatusError: If Asyntai answers with a non-2xx status code.
    """
    query = {k: v for k, v in params.items() if v is not None} if params is not None else None
    payload = {k: v for k, v in body.items() if v is not None} if body is not None else None

    async with httpx.AsyncClient(**client_kwargs) as client:
        response = await client.request(
            method,
            path,
            params=query,
            json=payload,
            headers={"Authorization": f"Bearer {api_key}", "User-Agent": "ag2-asyntai-toolkit"},
        )
    raw = _decode(response)
    _raise_for_status(response, raw)
    return raw if isinstance(raw, dict) else {}


class AsyntaiToolkit(Toolkit):
    """Toolkit that exposes an Asyntai website knowledge base as agent tools.

    The tools mirror Asyntai's REST endpoints:
      - ``asyntai_ask_website``: ask the knowledge base and get an answer
      - ``asyntai_list_websites``: list the websites on the account, with their ids
      - ``asyntai_add_text``: add a text to the knowledge base
      - ``asyntai_add_url``: add a web page to the knowledge base
      - ``asyntai_list_knowledge``: list the knowledge base entries
      - ``asyntai_get_leads``: read the leads the chatbot captured
      - ``asyntai_get_conversation``: read the messages of one conversation

    By default, passing the whole toolkit to an agent registers the two read-only
    knowledge tools, ``asyntai_ask_website`` and ``asyntai_list_websites``. The
    other tools change the knowledge base or read visitor data, so they are opt-in:
    pass ``all_tools=True``, or call the factory methods and pass the returned tools
    to the agent::

        toolkit = AsyntaiToolkit()

        # ask and list websites
        agent = Agent("a", config=config, tools=[toolkit])

        # ask one website, and read its leads
        agent = Agent("a", config=config, tools=[toolkit.ask(website_id=12), toolkit.get_leads(website_id=12)])

    When ``api_key`` is omitted, ``ASYNTAI_API_KEY`` is read from the environment.
    """

    __slots__ = ("_api_key", "_base_url", "_timeout", "_proxy", "_verify")

    def __init__(
        self,
        api_key: str | None = None,
        *,
        website_id: int | Variable | None = None,
        all_tools: bool = False,
        base_url: str = "https://asyntai.com/api/v1",
        timeout: float = 60.0,
        proxy: str | None = None,
        verify: bool = True,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        """Build the toolkit and its default tools.

        Args:
            api_key: Asyntai API key. Falls back to ``ASYNTAI_API_KEY``.
            website_id: Default Asyntai website for every tool. When it is not
                set, Asyntai uses the account's primary website.
            all_tools: Also register the tools that change the knowledge base or
                read leads and conversations.
            base_url: Asyntai API root. Trailing slashes are stripped.
            timeout: Per-request timeout in seconds.
            proxy: Proxy URL for the outgoing HTTP connection.
            verify: Whether to verify the server's TLS certificate.
            middleware: Middleware applied to every tool in the toolkit.

        Raises:
            ValueError: If no API key is passed and ``ASYNTAI_API_KEY`` is not set.
        """
        key = api_key or os.environ.get(_API_KEY_ENV_VAR)
        if not key:
            raise ValueError(f"api_key is required: pass it or set {_API_KEY_ENV_VAR}")

        self._api_key = key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._proxy = proxy
        self._verify = verify

        tools = [self.ask(website_id=website_id), self.list_websites()]
        if all_tools:
            tools += [
                self.add_text(website_id=website_id),
                self.add_url(website_id=website_id),
                self.list_knowledge(website_id=website_id),
                self.get_leads(website_id=website_id),
                self.get_conversation(),
            ]

        super().__init__(*tools, name="asyntai_toolkit", middleware=middleware)

    def ask(
        self,
        *,
        website_id: int | Variable | None = None,
        name: str = "asyntai_ask_website",
        description: str = (
            "Ask a question to the Asyntai knowledge base of a website. The answer comes from the "
            "website's own pages, documents and knowledge base."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        """Build the tool that asks the knowledge base.

        Args:
            website_id: Website to ask when the model does not name one.
            name: Tool name registered with the agent.
            description: Tool description shown to the model.
            middleware: Middleware applied to this tool.

        Returns:
            A ``FunctionTool`` that calls Asyntai's ``/chat/`` endpoint.
        """
        api_key = self._api_key
        client_kwargs = self._client_kwargs()

        @tool(name=name, description=description, middleware=middleware)
        async def asyntai_ask_website(
            question: Annotated[str, Field(description="The question to ask, in natural language.")],
            ctx: Context,
            session_id: Annotated[
                str | None,
                Field(description="Conversation id from an earlier answer, to ask a follow-up question."),
            ] = None,
            website: Annotated[int | None, Field(description="Id of the Asyntai website to ask.")] = None,
        ) -> ToolResult:
            """Ask the Asyntai knowledge base of a website."""
            raw = await _request(
                client_kwargs,
                api_key,
                "POST",
                "/chat/",
                body={
                    "message": question,
                    "session_id": session_id,
                    "website_id": _website(website, website_id, ctx),
                },
            )
            return ToolResult(AsyntaiAnswer(answer=_text(raw, "response"), session_id=_text(raw, "session_id")))

        return asyntai_ask_website

    def list_websites(
        self,
        *,
        name: str = "asyntai_list_websites",
        description: str = "List the Asyntai websites (chatbots) on the account, with their ids.",
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        """Build the tool that lists the account's websites.

        Args:
            name: Tool name registered with the agent.
            description: Tool description shown to the model.
            middleware: Middleware applied to this tool.

        Returns:
            A ``FunctionTool`` that calls Asyntai's ``/websites/`` endpoint.
        """
        api_key = self._api_key
        client_kwargs = self._client_kwargs()

        @tool(name=name, description=description, middleware=middleware)
        async def asyntai_list_websites() -> ToolResult:
            """List the Asyntai websites on the account."""
            raw = await _request(client_kwargs, api_key, "GET", "/websites/")
            return ToolResult(AsyntaiWebsites(websites=_items(raw, "websites")))

        return asyntai_list_websites

    def add_text(
        self,
        *,
        website_id: int | Variable | None = None,
        name: str = "asyntai_add_text",
        description: str = "Add a text to the Asyntai knowledge base, so the chatbot can answer from it.",
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        """Build the tool that adds a text to the knowledge base.

        Args:
            website_id: Website to add to when the model does not name one.
            name: Tool name registered with the agent.
            description: Tool description shown to the model.
            middleware: Middleware applied to this tool.

        Returns:
            A ``FunctionTool`` that calls Asyntai's ``/knowledge/text/`` endpoint.
        """
        api_key = self._api_key
        client_kwargs = self._client_kwargs()

        @tool(name=name, description=description, middleware=middleware)
        async def asyntai_add_text(
            title: Annotated[str, Field(description="A short title for the text.")],
            content: Annotated[str, Field(description="The text to add.")],
            ctx: Context,
            website: Annotated[int | None, Field(description="Id of the Asyntai website to add it to.")] = None,
        ) -> ToolResult:
            """Add a text to the Asyntai knowledge base."""
            raw = await _request(
                client_kwargs,
                api_key,
                "POST",
                "/knowledge/text/",
                body={"title": title, "content": content, "website_id": _website(website, website_id, ctx)},
            )
            return ToolResult(_knowledge_added(raw))

        return asyntai_add_text

    def add_url(
        self,
        *,
        website_id: int | Variable | None = None,
        name: str = "asyntai_add_url",
        description: str = "Add a web page to the Asyntai knowledge base. Asyntai reads the page and stores its content.",
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        """Build the tool that adds a web page to the knowledge base.

        Args:
            website_id: Website to add to when the model does not name one.
            name: Tool name registered with the agent.
            description: Tool description shown to the model.
            middleware: Middleware applied to this tool.

        Returns:
            A ``FunctionTool`` that calls Asyntai's ``/knowledge/url/`` endpoint.
        """
        api_key = self._api_key
        client_kwargs = self._client_kwargs()

        @tool(name=name, description=description, middleware=middleware)
        async def asyntai_add_url(
            url: Annotated[str, Field(description="The address of the page to add.")],
            ctx: Context,
            website: Annotated[int | None, Field(description="Id of the Asyntai website to add it to.")] = None,
        ) -> ToolResult:
            """Add a web page to the Asyntai knowledge base."""
            raw = await _request(
                client_kwargs,
                api_key,
                "POST",
                "/knowledge/url/",
                body={"url": url, "website_id": _website(website, website_id, ctx)},
            )
            return ToolResult(_knowledge_added(raw))

        return asyntai_add_url

    def list_knowledge(
        self,
        *,
        website_id: int | Variable | None = None,
        name: str = "asyntai_list_knowledge",
        description: str = "List the entries in the Asyntai knowledge base.",
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        """Build the tool that lists the knowledge base entries.

        Args:
            website_id: Website to read when the model does not name one.
            name: Tool name registered with the agent.
            description: Tool description shown to the model.
            middleware: Middleware applied to this tool.

        Returns:
            A ``FunctionTool`` that calls Asyntai's ``/knowledge/`` endpoint.
        """
        api_key = self._api_key
        client_kwargs = self._client_kwargs()

        @tool(name=name, description=description, middleware=middleware)
        async def asyntai_list_knowledge(
            ctx: Context,
            limit: Annotated[int, Field(description="Maximum number of entries to return.")] = 50,
            website: Annotated[int | None, Field(description="Id of the Asyntai website to read.")] = None,
        ) -> ToolResult:
            """List the entries in the Asyntai knowledge base."""
            raw = await _request(
                client_kwargs,
                api_key,
                "GET",
                "/knowledge/",
                params={"limit": limit, "website_id": _website(website, website_id, ctx)},
            )
            return ToolResult(AsyntaiKnowledgeEntries(entries=_items(raw, "entries")))

        return asyntai_list_knowledge

    def get_leads(
        self,
        *,
        website_id: int | Variable | None = None,
        name: str = "asyntai_get_leads",
        description: str = "Read the leads that the Asyntai chatbot captured (visitors who left contact details).",
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        """Build the tool that reads the captured leads.

        Args:
            website_id: Website to read when the model does not name one. When
                it is not set, Asyntai returns the leads of all websites.
            name: Tool name registered with the agent.
            description: Tool description shown to the model.
            middleware: Middleware applied to this tool.

        Returns:
            A ``FunctionTool`` that calls Asyntai's ``/leads/`` endpoint.
        """
        api_key = self._api_key
        client_kwargs = self._client_kwargs()

        @tool(name=name, description=description, middleware=middleware)
        async def asyntai_get_leads(
            ctx: Context,
            limit: Annotated[int, Field(description="Maximum number of leads to return (at most 100).")] = 50,
            email: Annotated[str | None, Field(description="Only return the lead with this email address.")] = None,
            since: Annotated[
                str | None,
                Field(description="Only return leads captured after this date or time (ISO 8601, e.g. 2026-01-31)."),
            ] = None,
            website: Annotated[int | None, Field(description="Id of the Asyntai website to read.")] = None,
        ) -> ToolResult:
            """Read the leads that the Asyntai chatbot captured."""
            raw = await _request(
                client_kwargs,
                api_key,
                "GET",
                "/leads/",
                params={
                    "limit": limit,
                    "email": email,
                    "since": since,
                    "website_id": _website(website, website_id, ctx),
                },
            )
            return ToolResult(AsyntaiLeads(leads=_items(raw, "leads")))

        return asyntai_get_leads

    def get_conversation(
        self,
        *,
        name: str = "asyntai_get_conversation",
        description: str = "Read the messages of one Asyntai chat conversation.",
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        """Build the tool that reads one conversation.

        Args:
            name: Tool name registered with the agent.
            description: Tool description shown to the model.
            middleware: Middleware applied to this tool.

        Returns:
            A ``FunctionTool`` that calls Asyntai's ``/conversations/`` endpoint.
        """
        api_key = self._api_key
        client_kwargs = self._client_kwargs()

        @tool(name=name, description=description, middleware=middleware)
        async def asyntai_get_conversation(
            session_id: Annotated[str, Field(description="The id of the conversation, for example from a lead.")],
            limit: Annotated[int, Field(description="Maximum number of messages to return.")] = 50,
        ) -> ToolResult:
            """Read the messages of one Asyntai chat conversation."""
            raw = await _request(
                client_kwargs,
                api_key,
                "GET",
                "/conversations/",
                params={"session_id": session_id, "limit": limit},
            )
            return ToolResult(AsyntaiConversation(session_id=session_id, messages=_items(raw, "messages")))

        return asyntai_get_conversation

    def _client_kwargs(self) -> dict[str, Any]:
        """Snapshot the connection settings as ``httpx.AsyncClient`` keyword arguments."""
        return {
            "base_url": self._base_url,
            "timeout": self._timeout,
            "proxy": self._proxy,
            "verify": self._verify,
        }


def _website(website: int | None, default: int | Variable | None, ctx: Context) -> int | None:
    """Return the website the model named, else the tool's default."""
    if website is not None:
        return website
    resolved: int | None = resolve_variable(default, ctx, param_name="website_id")
    return resolved


def _knowledge_added(raw: dict[str, Any]) -> AsyntaiKnowledgeAdded:
    """Build the result of an add-to-knowledge call."""
    chunks = raw.get("chunks_created")
    return AsyntaiKnowledgeAdded(
        entry_id=_text(raw, "id"),
        title=_text(raw, "title"),
        url=_text(raw, "url"),
        chunks_created=chunks if isinstance(chunks, int) else 0,
    )
