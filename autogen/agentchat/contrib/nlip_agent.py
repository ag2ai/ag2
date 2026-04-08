# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from typing import Any

import httpx

from ...events.agent_events import TerminationEvent
from ...import_utils import optional_import_block, require_optional_import
from ...io.base import IOStream
from ...oai.client import OpenAIWrapper
from ..conversable_agent import ConversableAgent
from ..remote import RemoteAgentError, RemoteAgentNotFoundError, RequestMessage, ResponseMessage
from ..remote.agent_service import AgentService

with optional_import_block() as result:
    from nlip_sdk.nlip import NLIP_Factory, NLIP_Message
    from nlip_server.server import NLIP_Application, NLIP_Session, setup_server

if not result.is_successful:
    NLIP_Session = object
    NLIP_Application = object

logger = logging.getLogger(__name__)


class NlipClientError(RemoteAgentError):
    """Base exception for NLIP client errors."""

    pass


class NlipConnectionError(NlipClientError):
    """Raised when connection to NLIP server fails."""

    pass


class NlipTimeoutError(NlipClientError):
    """Raised when NLIP request times out."""

    pass


class NlipAgentNotFoundError(NlipClientError, RemoteAgentNotFoundError):
    """Raised when NLIP agent is not found at the specified URL."""

    pass


@require_optional_import(["nlip_sdk", "nlip_server"], "nlip")
def request_message_to_nlip(request: RequestMessage) -> "NLIP_Message":
    """Convert AG2 RequestMessage to NLIP_Message.

    NLIP sessions are stateless — each request is independent.  When an
    ``NlipRemoteAgent`` participates in a GroupChat, the ``messages`` list
    it receives is the full GroupChat history which may contain internal
    AG2/LLM artefacts (``tool_calls``, ``role: "tool"``, etc.) that are
    meaningless to a remote NLIP server.

    The **last user-directed message** with actual text content becomes the
    top-level NLIP content (the current query).  The full chat history is
    attached as a JSON submessage labeled ``ag2_chat_history`` so the
    receiving agent can optionally use it for context.

    The ``context`` and ``client_tools`` fields are also attached as JSON
    submessages labeled ``ag2_context`` and ``ag2_client_tools`` respectively.

    Args:
        request: AG2 RequestMessage containing messages

    Returns:
        NLIP_Message with the latest query as content and full history
        as a JSON submessage
    """
    # Walk backwards to find the last message with real text content,
    # skipping tool-call artefacts and empty/None content.
    latest_text = ""
    for msg in reversed(request.messages):
        content = msg.get("content") or ""
        # Skip tool-related messages and messages whose content is literal "None"
        if msg.get("role") == "tool" or msg.get("tool_calls") or content in ("", "None"):
            continue
        latest_text = content
        break

    nlip_msg = NLIP_Factory.create_text(latest_text, language="english")

    # Attach the full chat history as a JSON submessage.
    # Sanitize: remap "tool" role → "assistant", drop tool_calls field,
    # and skip messages with empty/None content.
    if request.messages:
        sanitized: list[dict[str, str]] = []
        for msg in request.messages:
            content = msg.get("content") or ""
            if content in ("", "None"):
                continue
            role = msg.get("role", "user")
            if role == "tool":
                role = "assistant"
            sanitized.append({"role": role, "content": content})
        if sanitized:
            nlip_msg.add_json({"messages": sanitized}, label="ag2_chat_history")

    if request.context:
        nlip_msg.add_json(request.context, label="ag2_context")

    if request.client_tools:
        nlip_msg.add_json(request.client_tools, label="ag2_client_tools")

    return nlip_msg


@require_optional_import(["nlip_sdk", "nlip_server"], "nlip")
def request_message_from_nlip(nlip_msg: "NLIP_Message") -> RequestMessage:
    """Convert an incoming NLIP_Message (from a client) to an AG2 RequestMessage.

    Supports two wire formats produced by :func:`request_message_to_nlip`:

    * **Structured** – a JSON submessage labeled ``ag2_chat_history``
      containing the full chat history (sent by :class:`NlipRemoteAgent`).
      ``ag2_context`` and ``ag2_client_tools`` are also packed.
    * **Plain text** – a bare natural-language query (sent by curl or any
      non-AG2 NLIP client).

    Args:
        nlip_msg: NLIP_Message received from client

    Returns:
        RequestMessage whose ``messages`` list is ready to feed to an agent
    """
    # Prefer structured history from ag2_chat_history submessage
    if nlip_msg.submessages:
        history_submsg = nlip_msg.find_labeled_submessage("ag2_chat_history")
        context_submsg = nlip_msg.find_labeled_submessage("ag2_context")
        client_tools_submsg = nlip_msg.find_labeled_submessage("ag2_client_tools")

        history = []
        if history_submsg is not None and isinstance(history_submsg.content, dict):
            history = history_submsg.content.get("messages", [])

        context = {}
        if context_submsg is not None and isinstance(context_submsg.content, dict):
            context = context_submsg.content

        client_tools = {}
        if client_tools_submsg is not None and isinstance(client_tools_submsg.content, dict):
            client_tools = client_tools_submsg.content

        return RequestMessage(messages=history, context=context, client_tools=client_tools)

    # Fallback: treat top-level text as a single user message
    text = nlip_msg.extract_text() or ""
    if not text.strip():
        return RequestMessage(messages=[])

    return RequestMessage(messages=[{"role": "user", "content": text}])


@require_optional_import(["nlip_sdk", "nlip_server"], "nlip")
def response_message_to_nlip(response: ResponseMessage) -> "NLIP_Message":
    """Convert AG2 ResponseMessage to NLIP_Message.

    The response content is encoded as plain text (no ``role: `` prefix) so
    that any NLIP-conformant client – not just AG2 – can consume it directly.
    When there are multiple messages (e.g. tool-call steps followed by a final
    reply) only the last assistant message is used as the top-level content;
    earlier messages are silently discarded because NLIP sessions are stateless.

    Args:
        response: AG2 ResponseMessage

    Returns:
        NLIP_Message with plain text content and an optional INPUT_REQUIRED
        error sub-message when human input is needed
    """
    # Prefer the last assistant message; fall back to the last message of any role.
    main_text = ""
    for msg in reversed(response.messages):
        content = msg.get("content") or ""
        if content:
            main_text = content
            break

    nlip_msg = NLIP_Factory.create_text(main_text, language="english")

    if response.context:
        nlip_msg.add_json(response.context, label="ag2_context")

    if response.input_required:
        error_msg = NLIP_Factory.create_error_code("INPUT_REQUIRED")
        error_msg.content = response.input_required
        if nlip_msg.submessages is None:
            nlip_msg.submessages = []
        nlip_msg.submessages.append(error_msg)

    return nlip_msg


@require_optional_import(["nlip_sdk", "nlip_server"], "nlip")
def response_message_from_nlip(nlip_msg: "NLIP_Message") -> ResponseMessage:
    """Parse a NLIP response message received from a remote server.

    The server's top-level text content is treated as the assistant's
    reply.  No role-prefix parsing is performed — the content is taken
    as-is.

    Args:
        nlip_msg: NLIP_Message from server

    Returns:
        ResponseMessage with messages and optional input_required
    """
    text = nlip_msg.extract_text() or ""

    messages: list[dict[str, Any]] = []
    if text.strip():
        messages.append({"role": "assistant", "content": text})

    # Check for INPUT_REQUIRED error and ag2_context in submessages
    input_required = None
    context: dict[str, Any] | None = None
    if nlip_msg.submessages:
        context_submsg = nlip_msg.find_labeled_submessage("ag2_context")
        if context_submsg is not None and isinstance(context_submsg.content, dict):
            context = context_submsg.content

        for submsg in nlip_msg.submessages:
            if submsg.format == "error" and submsg.content and "INPUT_REQUIRED" in str(submsg.content):
                input_required = submsg.content
                break

    return ResponseMessage(messages=messages, context=context, input_required=input_required)


@require_optional_import(["nlip_sdk", "nlip_server"], "nlip")
class AG2NlipSession(NLIP_Session):
    """NLIP Session that executes an AG2 ConversableAgent."""

    def __init__(self, agent: ConversableAgent):
        super().__init__()
        self.agent = agent
        self.agent_service = AgentService(agent)

    async def start(self):
        """Initialize session (optional for stateless sessions)."""
        await super().start()
        logger.info(f"Started NLIP session for agent: {self.agent.name}")

    async def execute(self, msg: "NLIP_Message") -> "NLIP_Message":
        """Execute the agent and return NLIP response.

        Args:
            msg: NLIP_Message from client

        Returns:
            NLIP_Message response
        """
        logger.info(f"Executing agent {self.agent.name} with NLIP message")

        # Convert NLIP → RequestMessage
        request = request_message_from_nlip(msg)

        # Execute agent via AgentService
        # Note: AgentService yields ServiceResponse objects
        final_service_response = None
        async for service_response in self.agent_service(request):
            if service_response.input_required:
                # Return input request — propagate any context updates
                # the agent may have made before asking for input.
                return response_message_to_nlip(
                    ResponseMessage(
                        messages=[{"role": "assistant", "content": "Input required"}],
                        context=service_response.context,
                        input_required=service_response.input_required,
                    )
                )
            final_service_response = service_response

        # Convert ServiceResponse → ResponseMessage → NLIP
        if final_service_response is None:
            response_msg = ResponseMessage(messages=[{"role": "assistant", "content": "No response generated"}])
        elif final_service_response.message is None:
            # Agent finished but produced no final message — still propagate
            # any context updates it may have made along the way.
            response_msg = ResponseMessage(
                messages=[{"role": "assistant", "content": "No response generated"}],
                context=final_service_response.context,
            )
        else:
            # ServiceResponse has a single 'message' (dict), not 'messages' (list)
            response_msg = ResponseMessage(
                messages=[final_service_response.message],
                context=final_service_response.context,
            )

        return response_message_to_nlip(response_msg)

    async def correlated_execute(self, msg: "NLIP_Message") -> dict[str, Any]:  # type: ignore[override]
        response: NLIP_Message = await super().correlated_execute(msg)
        return response.model_dump(exclude_none=True)

    async def stop(self):
        """Clean up resources."""
        logger.info(f"Stopping NLIP session for agent: {self.agent.name}")
        await super().stop()


@require_optional_import(["nlip_sdk", "nlip_server"], "nlip")
class AG2NlipApplication(NLIP_Application):
    """NLIP Application that serves AG2 agents.

    This class is both a :class:`nlip_server.server.NLIP_Application` (so
    ``nlip-server`` can call :meth:`create_session` / :meth:`startup` /
    :meth:`shutdown` on it) **and** an ASGI callable. The latter lets users
    hand the instance directly to any ASGI server without an extra wrapper::

        import uvicorn
        from autogen import ConversableAgent
        from autogen.agentchat.contrib.nlip_agent import AG2NlipApplication

        agent = ConversableAgent(name="assistant", ...)
        nlip_app = AG2NlipApplication(agent)
        uvicorn.run(nlip_app, host="0.0.0.0", port=8000)

    Internally the instance eagerly builds the FastAPI application via
    :func:`nlip_server.server.setup_server` and delegates all ASGI traffic
    to it. Building eagerly (rather than lazily on first request) is
    required so that ASGI lifespan startup/shutdown events are wired up
    before any request arrives.
    """

    def __init__(self, agent: ConversableAgent):
        super().__init__()
        self.agent = agent
        self._asgi_app = setup_server(self)

    @property
    def asgi_app(self):
        return self._asgi_app

    async def __call__(self, scope, receive, send) -> None:
        """ASGI entrypoint — delegates to the wrapped FastAPI application."""
        await self._asgi_app(scope, receive, send)

    async def startup(self):
        """Application startup hook."""
        logger.info(f"Starting NLIP application for agent: {self.agent.name}")

    async def shutdown(self):
        """Application shutdown hook."""
        logger.info(f"Shutting down NLIP application for agent: {self.agent.name}")

    def create_session(self) -> "NLIP_Session":
        """Create a new session for each request.

        Returns:
            AG2NlipSession instance
        """
        return AG2NlipSession(self.agent)


@require_optional_import(["nlip_sdk", "nlip_server"], "nlip")
class NlipRemoteAgent(ConversableAgent):
    """Remote agent client for NLIP endpoints.

    This class allows you to connect to a remote NLIP endpoint and use it
    as a ConversableAgent in AG2 workflows.

    Example:
        >>> remote_agent = NlipRemoteAgent(url="http://remote-server:8000", name="remote_assistant")
        >>> await remote_agent.a_generate_remote_reply(messages=[{"role": "user", "content": "Hello"}])
    """

    def __init__(
        self,
        url: str,
        name: str,
        *,
        silent: bool | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        """Initialize NlipRemoteAgent.

        Args:
            url: The base URL of the NLIP server (e.g., "http://localhost:8000")
            name: A unique identifier for this agent instance
            silent: Whether to print messages (default: None)
            timeout: Request timeout in seconds (default: 60.0)
            max_retries: Maximum number of retry attempts (default: 3)
        """
        self.url = url.rstrip("/")  # Remove trailing slash
        self._timeout = timeout
        self._max_retries = max_retries

        super().__init__(name, silent=silent)

        self.__llm_config: dict[str, Any] = {}

        # Replace OAI reply functions with remote reply
        self.replace_reply_func(
            ConversableAgent.generate_oai_reply,
            NlipRemoteAgent.generate_remote_reply,
        )
        self.replace_reply_func(
            ConversableAgent.a_generate_oai_reply,
            NlipRemoteAgent.a_generate_remote_reply,
        )

    def generate_remote_reply(
        self,
        messages: list[dict[str, Any]] | None = None,
        sender: ConversableAgent | None = None,
        config: OpenAIWrapper | None = None,
    ) -> tuple[bool, dict[str, Any] | None]:
        """Synchronous version not supported.

        NLIP communication is inherently asynchronous. Use a_generate_remote_reply instead.

        Raises:
            NotImplementedError: Always raised
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} only supports async communication. Use a_generate_remote_reply instead."
        )

    async def a_generate_remote_reply(
        self,
        messages: list[dict[str, Any]] | None = None,
        sender: ConversableAgent | None = None,
        config: OpenAIWrapper | None = None,
    ) -> tuple[bool, dict[str, Any] | None]:
        """Generate reply by calling remote NLIP endpoint.

        Args:
            messages: List of messages in OpenAI format
            sender: The agent that sent the message
            config: OpenAI wrapper configuration (for tool extraction)

        Returns:
            Tuple of (final, reply) where final is True if this is the final reply
        """
        if messages is None:
            messages = self._oai_messages[sender]

        # Build request (NLIP standard: only messages)
        request = RequestMessage(
            messages=messages,
            context=self.context_variables.data,
            client_tools=self.__llm_config.get("tools", []),
        )

        # Convert to NLIP
        nlip_request = request_message_to_nlip(request)

        # Send HTTP request with retries
        for attempt in range(self._max_retries):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(
                        f"{self.url}/nlip/",
                        json=nlip_request.model_dump(exclude_none=True),
                    )
                    response.raise_for_status()

                # Parse response
                nlip_response = NLIP_Message.model_validate(response.json())
                response_msg = response_message_from_nlip(nlip_response)

                # Handle input_required
                if response_msg.input_required:
                    user_input = await self.a_get_human_input(
                        prompt=f"Input for `{self.name}`\n{response_msg.input_required}"
                    )

                    if user_input == "exit":
                        IOStream.get_default().send(
                            TerminationEvent(
                                source=self.name,
                                message="User requested exit",
                            )
                        )
                        return True, None

                    # Append user input and retry
                    messages.append({"content": user_input, "role": "user"})
                    return await self.a_generate_remote_reply(messages, sender, config)

                if response_msg.context:
                    self.context_variables.update(response_msg.context)
                    if sender:
                        sender.context_variables.update(response_msg.context)

                # Return final message
                if response_msg.messages:
                    return True, response_msg.messages[-1]
                else:
                    return True, {"role": "assistant", "content": ""}

            except httpx.TimeoutException as e:
                if attempt == self._max_retries - 1:
                    raise NlipTimeoutError(f"Request to {self.url} timed out after {self._max_retries} attempts") from e
                logger.warning(f"Request timeout (attempt {attempt + 1}/{self._max_retries}), retrying...")
                await asyncio.sleep(1.0)

            except httpx.ConnectError as e:
                if attempt == self._max_retries - 1:
                    raise NlipConnectionError(f"Failed to connect to {self.url}") from e
                logger.warning(f"Connection failed (attempt {attempt + 1}/{self._max_retries}), retrying...")
                await asyncio.sleep(1.0)

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    raise NlipAgentNotFoundError(f"NLIP agent not found at {self.url}") from e
                raise NlipClientError(f"HTTP error {e.response.status_code}: {e.response.text}") from e

            except Exception as e:
                raise NlipClientError(f"Unexpected error communicating with NLIP server: {e}") from e

        # Should never reach here
        raise NlipClientError("Failed after maximum retries")

    def update_tool_signature(
        self,
        tool_sig: str | dict[str, Any],
        is_remove: bool,
        silent_override: bool = False,
    ) -> None:
        """This method required to support Handoffs."""
        self.__llm_config = self._update_tool_config(
            self.__llm_config,
            tool_sig=tool_sig,
            is_remove=is_remove,
            silent_override=silent_override,
        )
