---
title: Bindu: A2A Protocol and Decentralized Identity for Agents
---

[Bindu](https://docs.getbindu.com/) is an open-source framework and Agent-to-Agent (A2A) protocol that equips AI agents with Decentralized Identifiers (DIDs), cryptographically verifiable communication, and a robust skills system. This page focuses on integrating **Bindu** with AG2 to create secure, interoperable, and identity-aware multi-agent swarms.

At a high level, Bindu bridges the trust gap in decentralized AI networks by ensuring that when AG2 agents communicate, negotiate, or execute financial payloads across different boundaries, their identities and messages are mathematically proven. For more information, see the [Bindu documentation](https://docs.getbindu.com).

|                                              |                                                                   |
| -------------------------------------------- | ----------------------------------------------------------------- |
| **Decentralized Identifiers (DIDs)**         | Give agents a self-sovereign, cryptographically verifiable identity |
| **A2A Protocol Compliance**                  | Standardized, secure agent-to-agent communication across networks |
| **Skills System**                            | Equip agents with tools, storage, and advanced capabilities       |
| **X402 Payments**                            | Native support for secure agent-to-agent micro-transactions       |
| **Zero-Friction Integration**                | Upgrade existing AG2 agents to Bindu nodes with a single function |

## Installation

Bindu works seamlessly with existing AG2 setups.

1. **Install the Bindu SDK:**

```bash
pip install bindu
```

2. **Configure your environment:**

Ensure your LLM provider keys are set for AG2.

```bash
OPENROUTER_API_KEY=<YOUR_OPENROUTER_API_KEY>
```

## Features

- **Cryptographic Passports**: Automatically generate W3C-compliant DIDs and Ed25519 key pairs for every agent
- **Message Signing**: Guarantee the authenticity, integrity, and non-repudiation of agent payloads
- **Interoperability**: Connect AG2 agents with agents built in other frameworks using the language-agnostic A2A protocol
- **Secure Swarm Routing**: Cryptographically verify the identity of worker nodes before passing them sensitive context data

## Common Use Cases

- Cross-organization agent collaboration
- Agent-to-agent financial workflows and trading
- Auditable automation in regulated systems
- Secure multi-agent research teams

## AG2 with Bindu Example

This example demonstrates how to use `bindufy` to upgrade a standard AG2 `ConversableAgent` with a secure Bindu identity (DID).

```python
import logging
import os
from autogen import ConversableAgent, LLMConfig
from bindu.penguin import bindufy
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# 1. Configure the LLM Provider
# Using OpenRouter to access Claude-3 in this example, but any AG2-supported LLM works.
llm_config = LLMConfig({
    "api_type": "openai",
    "model": "anthropic/claude-3-haiku",
    "api_key": os.getenv("OPENROUTER_API_KEY", ""),
    "base_url": "https://openrouter.ai/api/v1",
})

# 2. Define the Bindu Microservice Metadata
# This configuration dictates how the agent is identified and routed on the Bindu network.
config = {
    "author": "ag2.developer@example.com",
    "name": "ag2-networked-assistant",
    "description": "An AG2 ConversableAgent exposed via the Bindu A2A protocol.",
    "deployment": {
        "url": "http://localhost:3773",
        "expose": False, # Set to True to expose the agent to the public internet via tunneling
    },
    "skills": []
}

# 3. Create the A2A Bridge Handler
# This function is triggered every time a Bindu node sends a message to this agent's DID.
def handler(messages: list[dict[str, str]]):
    # Parse the incoming JSON-RPC payload to extract the user's latest prompt
    if not messages:
        return [{"role": "assistant", "content": "No input provided."}]

    user_input = messages[-1].get("content", "")
    if not user_input:
        return [{"role": "assistant", "content": "Empty message."}]

    # Initialize a fresh agent per request to maintain a stateless API environment
    assistant = ConversableAgent(
        name="assistant",
        system_message="You are a helpful AG2 assistant. Keep answers concise.",
        llm_config=llm_config,
        human_input_mode="NEVER" # Disable CLI prompts for automated API execution
    )

    # Trigger the AG2 workflow
    response = assistant.run(
        message=user_input,
        max_turns=1 # Restrict to a single request/response cycle for standard API behavior
    )

    # Process the event stream to completion
    response.process()

    # Extract the final reply from the AG2 chat history and return it to the Bindu network
    if hasattr(response, "chat_history") and response.chat_history:
        reply = response.chat_history[-1].get("content", "")
        return [{"role": "assistant", "content": reply}]

    return [{"role": "assistant", "content": "Task completed."}]

if __name__ == "__main__":
    # Bindufy the agent: generates a W3C DID and starts the A2A JSON-RPC server
    bindufy(config, handler)
```

Access the complete architecture and advanced examples from the [Bindu GitHub repository](https://github.com/GetBindu/Bindu).

This example showcases:

1. **Microservice Configuration:** Defining a Bindu networked assistant with an open port.
2. **The AG2 Bridge:** Creating a stateless `handler` that translates incoming A2A payloads into an AG2 `ConversableAgent` chat sequence.
3. **Cryptographic Exposure:** Using `bindufy` to automatically generate a DID and expose the AG2 logic as a mathematically verifiable network endpoint.
