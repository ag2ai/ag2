# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import queue
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any


class ThreadMessenger:
    """Handles same-core agent communication using threading"""

    def __init__(self, max_workers: int = 4):
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.agent_queues: dict[str, queue.Queue] = {}
        self.queue_lock = threading.RLock()

    def send_async_message(self, from_agent: str, to_agent: str, message: dict) -> Future:
        """Use ThreadPoolExecutor.submit() for async same-core communication"""

        def _deliver_message():
            try:
                # Get or create queue for target agent
                target_queue = self.register_agent_queue(to_agent)

                # Add sender info to message
                delivery_message = {
                    "from_agent": from_agent,
                    "to_agent": to_agent,
                    "payload": message,
                    "delivery_type": "same_core_thread",
                }

                # Put message in agent's queue
                target_queue.put(delivery_message)

                print(f"Thread message delivered: {from_agent} -> {to_agent}")
                return True

            except Exception as e:
                print(f"Failed to deliver thread message {from_agent} -> {to_agent}: {e}")
                return False

        # Submit to thread pool for async execution
        future = self.thread_pool.submit(_deliver_message)
        return future

    def register_agent_queue(self, agent_id: str) -> queue.Queue:
        """Create queue.Queue() for agent"""
        with self.queue_lock:
            if agent_id not in self.agent_queues:
                self.agent_queues[agent_id] = queue.Queue()
                print(f"Created thread queue for agent: {agent_id}")

            return self.agent_queues[agent_id]

    def get_message(self, agent_id: str, timeout: float = 1.0) -> Any:
        """Get message from agent's queue"""
        if agent_id not in self.agent_queues:
            return None

        try:
            return self.agent_queues[agent_id].get(timeout=timeout)
        except queue.Empty:
            return None

    def has_messages(self, agent_id: str) -> bool:
        """Check if agent has pending messages"""
        if agent_id not in self.agent_queues:
            return False

        return not self.agent_queues[agent_id].empty()

    def shutdown(self):
        """Shutdown thread pool"""
        print("Shutting down thread messenger...")
        self.thread_pool.shutdown(wait=True)
        print("Thread messenger shutdown complete")
