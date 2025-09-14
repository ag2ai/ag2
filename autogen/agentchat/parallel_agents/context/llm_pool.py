# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import threading
import time
from typing import Any


class LLMConnection:
    """Represents an LLM connection with context"""

    def __init__(self, connection_id: str, config: dict[str, Any] = None):
        self.connection_id = connection_id
        self.config = config or {}
        self.current_context: dict[str, Any] = {}
        self.last_used = time.time()
        self.is_busy = False
        self.lock = threading.Lock()

    def set_context(self, context: dict[str, Any]):
        """Set the current context for this connection"""
        with self.lock:
            self.current_context = context.copy()
            self.last_used = time.time()

    def get_context(self) -> dict[str, Any]:
        """Get the current context"""
        with self.lock:
            return self.current_context.copy()

    def execute_prompt(self, prompt: str) -> str:
        """Execute prompt with current context (mock implementation)"""
        with self.lock:
            self.is_busy = True
            self.last_used = time.time()

            # Mock LLM execution - in real implementation, this would call actual LLM
            context_info = ", ".join([f"{k}={v}" for k, v in self.current_context.items()])
            result = (
                f"LLM Response (Connection {self.connection_id}): Processed '{prompt}' with context [{context_info}]"
            )

            # Simulate processing time
            time.sleep(0.1)

            self.is_busy = False
            return result


class LLMPool:
    """Manages pool of LLM connections for a specific CPU core with real-time context switching"""

    def __init__(self, core_id: int, pool_size: int = 2):
        self.core_id = core_id
        self.pool_size = pool_size
        self.connections: dict[str, LLMConnection] = {}
        self.available_connections: list = []
        self.pool_lock = threading.RLock()

        # Initialize connection pool
        self._initialize_pool()

        print(f"LLMPool initialized for core {core_id} with {pool_size} connections")

    def _initialize_pool(self):
        """Initialize LLM connections for this core"""
        for i in range(self.pool_size):
            connection_id = f"core_{self.core_id}_conn_{i}"
            connection = LLMConnection(connection_id)

            self.connections[connection_id] = connection
            self.available_connections.append(connection_id)

    def switch_context(self, connection: LLMConnection, new_context: dict[str, Any]) -> LLMConnection:
        """Update connection's context and return it for real-time context switching"""
        switch_start = time.time()

        # Perform context switch
        connection.set_context(new_context)

        switch_time = time.time() - switch_start
        print(f"Context switched for {connection.connection_id} in {switch_time:.3f}s")

        return connection

    def execute_with_context(self, agent_id: str, prompt: str, context: dict[str, Any]) -> str:
        """Switch context, execute LLM call, return result"""
        # Get available connection
        connection = self._get_available_connection()

        if connection is None:
            return f"Error: No available LLM connections for agent {agent_id} on core {self.core_id}"

        try:
            # Switch to new context
            self.switch_context(connection, context)

            # Execute with context
            result = connection.execute_prompt(prompt)

            print(f"LLM execution completed for agent {agent_id} on core {self.core_id}")
            return result

        finally:
            # Return connection to pool
            self._return_connection(connection)

    def _get_available_connection(self) -> LLMConnection | None:
        """Get an available connection from the pool"""
        with self.pool_lock:
            # Find available connection
            for conn_id in self.available_connections:
                connection = self.connections[conn_id]
                if not connection.is_busy:
                    self.available_connections.remove(conn_id)
                    return connection

            # If no available connections, wait for one
            print(f"No available connections in pool for core {self.core_id}, waiting...")

            # Simple wait strategy - in production, use proper scheduling
            for _ in range(10):  # Wait up to 1 second
                time.sleep(0.1)
                for conn_id in list(self.connections.keys()):
                    connection = self.connections[conn_id]
                    if not connection.is_busy and conn_id not in self.available_connections:
                        return connection

            return None

    def _return_connection(self, connection: LLMConnection):
        """Return connection to the available pool"""
        with self.pool_lock:
            if connection.connection_id not in self.available_connections:
                self.available_connections.append(connection.connection_id)

    def get_pool_stats(self) -> dict[str, Any]:
        """Get statistics about the LLM pool"""
        with self.pool_lock:
            busy_connections = sum(1 for conn in self.connections.values() if conn.is_busy)

            return {
                "core_id": self.core_id,
                "total_connections": len(self.connections),
                "available_connections": len(self.available_connections),
                "busy_connections": busy_connections,
                "utilization": busy_connections / len(self.connections) if self.connections else 0,
            }

    def cleanup(self):
        """Clean up LLM pool resources"""
        with self.pool_lock:
            # Wait for busy connections to finish
            for connection in self.connections.values():
                while connection.is_busy:
                    time.sleep(0.1)

            self.connections.clear()
            self.available_connections.clear()

        print(f"LLMPool for core {self.core_id} cleaned up")
