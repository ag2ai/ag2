import asyncio
import traceback
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from managers.cancellation_token import CancellationToken

# Type aliases for input handling
InputFuncType = Callable[[str, str], Awaitable[str]]
InputRequestType = str  # "text_input", "file_upload", etc.


class WebSocketManager:
    """
    Simplified WebSocket manager for AG2AI GroupChat workflows

    Handles WebSocket connections, message streaming, and user input requests
    without any external dependencies.
    """

    def __init__(self):
        self._connections: dict[str, WebSocket] = {}
        self._closed_connections: set[str] = set()
        self._input_responses: dict[str, asyncio.Queue[str]] = {}
        self._active_tasks: dict[str, asyncio.Task] = {}
        self._cancellation_tokens: dict[int, CancellationToken] = {}
        self._stop_flags: dict[str, bool] = {}

    async def connect(self, websocket: WebSocket, session_id: str) -> bool:
        """
        Accept a new WebSocket connection

        Args:
            websocket: FastAPI WebSocket instance
            session_id: Unique identifier for this session

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            await websocket.accept()
            self._connections[session_id] = websocket
            self._closed_connections.discard(session_id)
            self._input_responses[session_id] = asyncio.Queue()
            self._stop_flags[session_id] = False

            await self._send_message(
                session_id,
                {
                    "type": "system",
                    "status": "connected",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            return True
        except Exception:
            return False

    async def start_chat_stream(
        self,
        session_id: str,
        initial_message: str = "",
        chat=None,
    ) -> None:
        """
        Start a chat stream over the WebSocket connection.

        Args:
            session_id (str): The unique ID of the chat session.
            initial_message (dict): The initial message sent by the user.
            chat (AgentChat): The chat handler instance.
        """
        if session_id not in self._connections or session_id in self._closed_connections:
            raise ValueError(f"No active connection for session {session_id}")

        self._stop_flags[session_id] = False
        cancellation_token = CancellationToken()
        self._cancellation_tokens[session_id] = cancellation_token

        try:
            # Send initial message if provided
            if initial_message:
                await self._send_message(
                    session_id,
                    {
                        "type": "message",
                        "data": {
                            "source": "user",
                            "content": initial_message,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                    },
                )

            chat._set_cancellation_token(self._cancellation_tokens.get(session_id, None))
            task = asyncio.create_task(
                chat.manager.a_initiate_chat(chat.manager, message=initial_message.get("task"), clear_history=False)
            )
            self._active_tasks[session_id] = task

            # Wait for completion or cancellation
            await task

            if not self._stop_flags.get(session_id, True) and session_id not in self._closed_connections:
                await self._send_message(
                    session_id,
                    {
                        "type": "completion",
                        "status": "complete",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )
            else:
                await self._send_message(
                    session_id,
                    {
                        "type": "completion",
                        "status": "cancelled",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )

        except Exception as e:
            traceback.print_exc()
            await self._handle_stream_error(session_id, e)
        finally:
            self._cancellation_tokens.pop(session_id, None)
            self._active_tasks.pop(session_id, None)
            self._stop_flags.pop(session_id, None)

    def _create_input_func(self, session_id: str, timeout: int = 600) -> InputFuncType:
        """
        Create an input function for requesting user input during chat

        Args:
            session_id: Session identifier
            timeout: Timeout in seconds for input response

        Returns:
            Input function that can be used by chat handlers
        """

        async def input_handler(
            prompt: str = "",
            input_type: InputRequestType = "text_input",
        ) -> str:
            try:
                # Send input request to client
                await self._send_message(
                    session_id,
                    {
                        "type": "input_request",
                        "input_type": input_type,
                        "prompt": prompt,
                        "data": {"source": "system", "content": prompt},
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )

                # Wait for response with timeout
                if session_id in self._input_responses:
                    try:

                        async def poll_for_response():
                            while True:
                                # Check if session was closed/stopped
                                if session_id in self._closed_connections or self._stop_flags.get(session_id, False):
                                    raise ValueError("Session was closed or stopped")

                                # Try to get response with short timeout
                                try:
                                    response = await asyncio.wait_for(
                                        self._input_responses[session_id].get(),
                                        timeout=min(timeout, 5),
                                    )
                                    return response
                                except asyncio.TimeoutError:
                                    continue  # Keep checking for closed status

                        response = await asyncio.wait_for(poll_for_response(), timeout=timeout)
                        return response

                    except asyncio.TimeoutError:
                        await self.stop_session(session_id, "Input timeout")
                        raise
                else:
                    raise ValueError(f"No input queue for session {session_id}")

            except Exception:
                raise

        return input_handler

    async def send_message(self, session_id: str, message: dict[str, Any]) -> None:
        """
        Send a message to the client

        Args:
            session_id: Session identifier
            message: Message dictionary to send
        """
        await self._send_message(session_id, message)

    async def send_chat_message(self, session_id: str, content: str, source: str = "assistant") -> None:
        """
        Send a formatted chat message to the client

        Args:
            session_id: Session identifier
            content: Message content
            source: Message source (e.g., "assistant", "user", agent name)
        """
        await self._send_message(
            session_id,
            {
                "type": "message",
                "data": {
                    "source": source,
                    "content": content,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            },
        )

    async def send_agent_message(self, session_id: str, agent_name: str, content: str) -> None:
        """
        Send a message from a specific agent

        Args:
            session_id: Session identifier
            agent_name: Name of the agent sending the message
            content: Message content
        """
        await self.send_chat_message(session_id, content, source=agent_name)

    async def handle_input_response(self, session_id: str, response: str) -> None:
        """
        Handle input response from client

        Args:
            session_id: Session identifier
            response: User's input response
        """
        if session_id in self._input_responses:
            await self._input_responses[session_id].put(response)
        else:
            print(f"Received input response for inactive session {session_id}")

    async def stop_session(self, session_id: str, reason: str = "Session stopped") -> None:
        """
        Stop an active session

        Args:
            session_id: Session identifier
            reason: Reason for stopping
        """
        if session_id in self._cancellation_tokens:
            try:
                # Set stop flag
                self._stop_flags[session_id] = True
                # Cancel the task if it exists
                if session_id in self._active_tasks:
                    task = self._active_tasks[session_id]
                    if not task.done():
                        task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    raise e

                # Finally cancel the token
                self._cancellation_tokens[session_id].cancel()
                # Send completion message if connection is active
                if session_id in self._connections and session_id not in self._closed_connections:
                    await self._send_message(
                        session_id,
                        {
                            "type": "completion",
                            "status": "stopped",
                            "reason": reason,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                    )

            except Exception as e:
                print(e)
                raise

    async def disconnect(self, session_id: str) -> None:
        """
        Clean up connection and associated resources

        Args:
            session_id: Session identifier
        """

        # Mark as closed before cleanup
        self._closed_connections.add(session_id)

        # Stop the session
        await self.stop_session(session_id, "Connection closed")

        # Clean up resources
        self._connections.pop(session_id, None)
        self._cancellation_tokens.pop(session_id, None)
        self._input_responses.pop(session_id, None)
        self._active_tasks.pop(session_id, None)
        self._stop_flags.pop(session_id, None)

    async def _send_message(self, session_id: str, message: dict[str, Any]) -> None:
        """
        Internal method to send a message through WebSocket

        Args:
            session_id: Session identifier
            message: Message dictionary to send
        """
        if session_id in self._closed_connections:
            raise Exception(f"Session closed for {session_id}")

        try:
            if session_id in self._connections:
                websocket = self._connections[session_id]
                await websocket.send_json(message)
        except WebSocketDisconnect:
            await self.disconnect(session_id)
        except Exception:
            await self.disconnect(session_id)

    async def _handle_stream_error(self, session_id: str, error: Exception) -> None:
        """
        Handle stream errors

        Args:
            session_id: Session identifier
            error: Exception that occurred
        """
        if session_id not in self._closed_connections:
            await self._send_message(
                session_id,
                {
                    "type": "error",
                    "message": str(error),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

    async def cleanup(self) -> None:
        """Clean up all active connections and resources"""

        try:
            # Stop all sessions
            for session_id in list(self._active_tasks.keys()):
                self._stop_flags[session_id] = True
                if session_id in self._cancellation_tokens:
                    self._cancellation_tokens[session_id].cancel()

            # Disconnect all websockets with timeout
            async def disconnect_all():
                for session_id in list(self.active_connections):
                    try:
                        await asyncio.wait_for(self.disconnect(session_id), timeout=2)
                    except asyncio.TimeoutError:
                        raise
                    except Exception:
                        raise

            await asyncio.wait_for(disconnect_all(), timeout=10)

        except asyncio.TimeoutError:
            raise
        except Exception:
            raise
        finally:
            # Clear all internal state
            self._connections.clear()
            self._cancellation_tokens.clear()
            self._closed_connections.clear()
            self._input_responses.clear()
            self._active_tasks.clear()
            self._stop_flags.clear()

    @property
    def active_connections(self) -> set[str]:
        """Get set of active session IDs"""
        return set(self._connections.keys()) - self._closed_connections

    @property
    def active_sessions(self) -> set[str]:
        """Get set of sessions with active tasks"""
        return set(self._active_tasks.keys())

    def is_connected(self, session_id: str) -> bool:
        """Check if a session is actively connected"""
        return session_id in self._connections and session_id not in self._closed_connections

    def is_session_stopped(self, session_id: str) -> bool:
        """Check if a session has been stopped"""
        return self._stop_flags.get(session_id, False)

    def get_cancellation_token(self, session_id: str) -> CancellationToken | None:
        """Get the cancellation token for a session"""
        return self._cancellation_tokens.get(session_id)
