from typing import Dict, Optional
from dataclasses import dataclass
from autogen.agentchat import ConversableAgent

@dataclass
class BroadcastMessage:
    """Message to be broadcast to all agents."""
    content: Dict
    request_halt: bool = False

@dataclass 
class ResetMessage:
    """Message to reset agent state."""
    pass

@dataclass
class OrchestrationEvent:
    """Event for logging orchestration."""
    source: str
    message: str
