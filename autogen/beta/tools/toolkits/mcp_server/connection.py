# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
import json
import uuid
from typing import Any

import httpx

from autogen.beta.tools.toolkits.mcp_server.types import (
    JsonRpcNotification,
    JsonRpcRequest,
    JsonRpcResponse,
    MCPServerConfig,
    RawMCPTool,
)

PROTOCOL_VERSION = "2025-11-25"


class MCPConnection:
    """
    Single MCP server connection handler.
    Initialization is lazy: the handshake is performed automatically on the
    first request and the session is reused for all subsequent calls.
    """

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._session_id: str | None = None
        self._initialize_result: JsonRpcResponse | None = None
        self._protocol_version: str | None = None

    @staticmethod
    def _parse_response(response: httpx.Response) -> JsonRpcResponse:
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            return MCPConnection._parse_sse_response(response.text)
        return response.json()

    @staticmethod
    def _parse_sse_response(text: str) -> JsonRpcResponse:
        data_lines: list[str] = []
        result = None
        for line in text.splitlines():
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip(" "))
            elif line == "" and data_lines:
                result = json.loads("\n".join(data_lines))
                data_lines = []
        if data_lines:
            result = json.loads("\n".join(data_lines))
        if result is None:
            raise ValueError("No data field found in SSE response")
        return result

    async def _ensure_initialized(self) -> None:
        """Initialize the session on first use; no-op on subsequent calls."""
        if self._initialize_result is not None:
            return

        init_payload = {
            "id": str(uuid.uuid4()),
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "autogen", "version": "beta"},
            },
        }

        headers = {
            "Accept": "application/json, text/event-stream",
            **(self.config.headers or {}),
        }

        async with httpx.AsyncClient(
            timeout=self.config.connection_timeout, headers=headers, follow_redirects=True
        ) as client:
            response = await client.post(self.config.server_url, json=init_payload)
            response.raise_for_status()

            if session_id := response.headers.get("Mcp-Session-Id"):
                self._session_id = session_id

            result = self._parse_response(response)
            self._protocol_version = result.get("result", {}).get("protocolVersion", PROTOCOL_VERSION)
            self._initialize_result = result

            notification: JsonRpcNotification = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
            }
            notify_headers = {**headers}
            if self._session_id:
                notify_headers["Mcp-Session-Id"] = self._session_id
            await client.post(self.config.server_url, json=notification, headers=notify_headers)

    async def _make_request(self, data: JsonRpcRequest) -> JsonRpcResponse:
        await self._ensure_initialized()

        headers = {
            "Accept": "application/json, text/event-stream",
            **(self.config.headers or {}),
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        if self._protocol_version:
            headers["MCP-Protocol-Version"] = self._protocol_version

        async with httpx.AsyncClient(
            timeout=self.config.connection_timeout, headers=headers, follow_redirects=True
        ) as client:
            response = await client.post(self.config.server_url, json=data)
            response.raise_for_status()
            return self._parse_response(response)

    async def get_tools(self) -> list[RawMCPTool]:
        result = await self._make_request({
            "id": str(uuid.uuid4()),
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
        })
        return result.get("result", {}).get("tools", [])

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> JsonRpcResponse:
        return await self._make_request({
            "id": str(uuid.uuid4()),
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        })
