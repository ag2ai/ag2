#!/usr/bin/env python3
"""Example 10: A2A version fallback — v0.8 client vs v0.9 server.

Demonstrates that when a client requests A2UI v0.8 but the server only
supports v0.9, the response falls back to text-only. No A2UI DataParts
are returned.

Tests three scenarios:
  1. Client requests v0.9 → gets TextPart + A2UI DataPart
  2. Client requests v0.8 → gets text-only (graceful fallback)
  3. Client requests no extension → gets text-only

Requires: A Gemini API key + ag2[a2a] extra.

Usage:
    export GEMINI_API_KEY="..."
    python 10_a2a_version_fallback.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Set GEMINI_API_KEY to run this example.")
    sys.exit(1)

from httpx import ASGITransport

from autogen.a2a import A2aAgentServer, CardSettings, HttpxClientFactory
from autogen.a2a.client import A2aRemoteAgent
from autogen.agents.experimental.a2ui import A2UI_DEFAULT_DELIMITER, A2UIAgent
from autogen.agents.experimental.a2ui.a2a_helpers import A2UI_EXTENSION_URI

llm_config = {"api_type": "google", "model": "gemini-3.1-flash-lite-preview", "api_key": api_key}

CATALOG_PATH = Path(__file__).parent / "social_catalog.json"

# ─── Create A2UI agent + A2A server ───

agent = A2UIAgent(
    name="preview_agent",
    system_message=(
        "You are a product card designer. Given a product brief, create a simple "
        "A2UI card preview. Keep it minimal — just a Card with a Text headline "
        "and a Button. Always include a short text summary before the A2UI JSON."
    ),
    llm_config=llm_config,
    include_schema_in_prompt=False,
)

server = A2aAgentServer(
    agent=agent,
    url="http://localhost:9000",
    agent_card=CardSettings(name="Preview Agent", description="Test agent for version fallback"),
)

# Build ASGI app for in-process testing (no real HTTP needed)
a2a_app = server.build()

print("=" * 70)
print("  A2A Version Fallback Test")
print("=" * 70)

# Show server card
card = server.card
exts = card.capabilities.extensions or []
print("\nServer agent card extensions:")
for ext in exts:
    params = ext.params or {}
    print(f"  URI: {ext.uri}")
    print(f"  Params: {json.dumps(params, indent=4)}")

V08_URI = "https://a2ui.org/a2a-extension/a2ui/v0.8"
V09_URI = A2UI_EXTENSION_URI


async def test_scenario(name: str, requested_extensions: list[str]) -> None:
    """Send a message to the A2A server with specific extension requests."""
    print(f"\n{'─' * 70}")
    print(f"  Scenario: {name}")
    print(f"  Requested extensions: {requested_extensions or '(none)'}")
    print(f"{'─' * 70}")

    # Create client with ASGI transport (in-process, no real HTTP)
    client_factory = HttpxClientFactory(
        transport=ASGITransport(a2a_app),
    )

    remote = A2aRemoteAgent(
        url="http://test-server",
        name="test_client",
        client=client_factory,
    )

    # Override the extension request to simulate different client capabilities
    remote._get_requested_extensions = lambda: requested_extensions or None  # type: ignore[assignment]

    # Send a simple product brief
    message = "Create a product card for H2Oh water bottle, $39."

    try:
        _, reply = await remote.a_generate_remote_reply(
            messages=[{"role": "user", "content": message}],
            sender=None,
        )

        if reply is None:
            print("  Result: No reply received")
            return

        reply_content = reply.get("content", "") if isinstance(reply, dict) else str(reply)
        reply_messages = reply.get("messages", []) if isinstance(reply, dict) else []

        # Check what we got
        has_text = bool(reply_content)
        has_a2ui_datapart = bool(reply_messages and any(isinstance(m, dict) and "version" in m for m in reply_messages))
        has_a2ui_delimiter = A2UI_DEFAULT_DELIMITER in reply_content if isinstance(reply_content, str) else False

        print(f"  Reply type: {type(reply).__name__}")
        print(f"  Has text content: {has_text}")
        print(f"  Has A2UI DataPart: {has_a2ui_datapart}")
        print(f"  Contains A2UI delimiter in text: {has_a2ui_delimiter}")
        if has_text:
            print(f"  Content preview: {str(reply_content)[:150]}...")
        if has_a2ui_datapart:
            print(f"  A2UI operations: {len(reply_messages)}")
            for op in reply_messages:
                op_type = next((k for k in op if k != "version"), "?")
                print(f"    - {op_type}")

        if has_a2ui_datapart:
            print("  ✓  A2UI DataPart received (structured A2UI via extension)")
        elif has_a2ui_delimiter:
            print("  ⚠️  A2UI in raw text (no DataPart — extension not activated, fallback to text)")
        else:
            print("  ✓  Text-only response (no A2UI)")

    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")


async def main() -> None:
    print("\n" + "=" * 70)
    print("  Running scenarios...")
    print("=" * 70)

    # Scenario 1: Client requests v0.9 (our version) → should get A2UI
    await test_scenario("Client requests v0.9 (supported)", [V09_URI])

    # Scenario 2: Client requests v0.8 (old version) → should get text-only
    await test_scenario("Client requests v0.8 (unsupported)", [V08_URI])

    # Scenario 3: Client requests no extension → should get text-only
    await test_scenario("Client requests no A2UI extension", [])

    print(f"\n{'=' * 70}")
    print("  Done.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    asyncio.run(main())
