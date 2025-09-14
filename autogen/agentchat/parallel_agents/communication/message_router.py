# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp

from .serialization import MessageSerializer


class MessageRouter:
    """Routes messages between agents based on their core locations"""

    def __init__(self):
        self.agent_to_core_mapping: dict[str, int] = {}
        self.cross_core_queues: dict[int, mp.Queue] = {}
        self.serializer = MessageSerializer()
        self.thread_messenger = None  # Will be set externally

    def register_agent_location(self, agent_id: str, core_id: int):
        """Store agent_id -> core_id mapping"""
        self.agent_to_core_mapping[agent_id] = core_id

        # Create cross-core queue if not exists
        if core_id not in self.cross_core_queues:
            self.cross_core_queues[core_id] = mp.Queue()

    def route_message(self, from_agent: str, to_agent: str, message: dict):
        """Check if same core, route appropriately"""
        from_core = self.agent_to_core_mapping.get(from_agent)
        to_core = self.agent_to_core_mapping.get(to_agent)

        if from_core is None or to_core is None:
            print(f"Warning: Agent location not registered for {from_agent} or {to_agent}")
            return False

        # Add routing metadata
        routing_message = {
            "from_agent": from_agent,
            "to_agent": to_agent,
            "from_core": from_core,
            "to_core": to_core,
            "payload": message,
            "message_type": "agent_communication",
        }

        if from_core == to_core:
            # Same core - use threading (concurrent.futures)
            print(f"Routing same-core message: {from_agent} -> {to_agent} (core {from_core})")
            return self._send_same_core_message(routing_message)
        else:
            # Cross core - use multiprocessing
            print(f"Routing cross-core message: {from_agent} (core {from_core}) -> {to_agent} (core {to_core})")
            return self.send_cross_core_message(to_core, routing_message)

    def send_cross_core_message(self, target_core: int, message: dict):
        """Use multiprocessing.Queue for cross-core communication"""
        try:
            if target_core not in self.cross_core_queues:
                self.cross_core_queues[target_core] = mp.Queue()

            # Serialize and send message
            serialized_message = self.serializer.serialize(message)
            self.cross_core_queues[target_core].put(serialized_message)

            print(f"Cross-core message sent to core {target_core}")
            return True

        except Exception as e:
            print(f"Failed to send cross-core message to core {target_core}: {e}")
            return False

    def _send_same_core_message(self, message: dict):
        """Send message within same core using threading"""
        try:
            if self.thread_messenger is not None:
                return self.thread_messenger.send_async_message(message["from_agent"], message["to_agent"], message)
            else:
                print(f"Same-core message queued: {message['from_agent']} -> {message['to_agent']}")
                return True

        except Exception as e:
            print(f"Failed to send same-core message: {e}")
            return False

    def receive_cross_core_message(self, core_id: int, timeout: float = 1.0):
        """Receive message from cross-core queue"""
        try:
            if core_id not in self.cross_core_queues:
                return None

            serialized_message = self.cross_core_queues[core_id].get(timeout=timeout)
            message = self.serializer.deserialize(serialized_message)

            print(f"Cross-core message received on core {core_id}")
            return message

        except mp.queues.Empty:
            return None
        except Exception as e:
            print(f"Failed to receive cross-core message on core {core_id}: {e}")
            return None

    def set_thread_messenger(self, thread_messenger):
        """Set the thread messenger for same-core communication"""
        self.thread_messenger = thread_messenger
