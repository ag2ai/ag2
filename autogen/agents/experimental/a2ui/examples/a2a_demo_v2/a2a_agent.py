# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A2A server hosting the social previewer A2UIAgent.

The A2UIAgent is auto-detected by A2aAgentServer which:
- Uses A2UIAgentExecutor (splits responses into TextPart + A2UI DataPart)
- Declares A2UI v0.9 extension + supportedCatalogIds in the agent card
- Handles extension negotiation (v0.9 clients get DataParts, others get text)

Run this first, then run frontend.py in a separate terminal.

Usage:
    export GEMINI_API_KEY="..."
    python a2a_agent.py
    # A2A server runs at http://localhost:9000
    # Agent card at http://localhost:9000/.well-known/agent.json
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv

from autogen.a2a import A2aAgentServer, CardSettings
from autogen.agents.experimental.a2ui import A2UIAction, A2UIAgent

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Set GEMINI_API_KEY to run this demo.")
    sys.exit(1)

llm_config = {"api_type": "google", "model": "gemini-3.1-flash-lite-preview", "api_key": api_key}

CATALOG_PATH = Path(__file__).parent.parent / "social_catalog.json"

CUSTOM_RULES = """**CUSTOM COMPONENT RULES:**
- For 'LinkedInPost', populate ALL fields: authorName, authorHeadline, authorAvatarUrl, body, hashtags, mediaChild, likes, comments, reposts.
- For 'XPost', populate ALL fields: authorName, authorHandle, authorAvatarUrl, verified, body, mediaChild, replies, reposts, likes, views, bookmarks.
- Engagement metrics are integers, not strings.
- Create Image components separately and reference by ID in mediaChild.
"""


# ─── Tools ───


def schedule_posts(
    time: Annotated[str, "The time to schedule posts for"] = "not specified",
) -> str:
    """Schedule the marketing posts (email, LinkedIn, X) for publishing at the given time."""
    print(f"  [schedule_posts] Scheduling all posts for {time}")
    return (
        f"All three posts have been scheduled for {time} tomorrow!\n\n"
        f"- Email: queued for {time}\n"
        f"- LinkedIn: scheduled for {time}\n"
        f"- X/Twitter: scheduled for {time}"
    )


# ─── A2UI Agent ───

social_previewer = A2UIAgent(
    name="social_previewer",
    system_message=(
        "You are a marketing content designer. You work in a multi-step flow:\n\n"
        "## STEP 1: Generate previews (initial request or rewrite)\n"
        "When you receive a product brief OR a 'rewrite_previews' action, create "
        "THREE preview cards in a single A2UI response:\n\n"
        "1. **Email preview**: Use basic catalog components in a Card:\n"
        "   To:, Subject:, Divider, Image, headline (h2), body, Divider, CTA Button, "
        "Divider, footer (caption)\n\n"
        "2. **LinkedIn preview**: LinkedInPost with ALL fields populated\n\n"
        "3. **X/Twitter preview**: XPost with ALL fields populated, body under 280 chars\n\n"
        "Use a Column as root with section headers (Text with variant 'h2') between previews.\n"
        "IMPORTANT: Do NOT use markdown syntax (##, **, etc.) in Text component values. "
        "Use the 'variant' field for styling instead (h1, h2, h3, caption, body).\n"
        "At the bottom, add a Row with Approve and Rewrite buttons.\n\n"
        "IMPORTANT: Always use surfaceId 'marketing' for all surfaces.\n\n"
        "## STEP 2: Approved — show scheduling options on a NEW surface\n"
        "When you receive an 'approve_previews' action, create a NEW surface with "
        "surfaceId 'scheduling' (do NOT touch the 'marketing' surface). Use the same catalogId.\n"
        "Then updateComponents on 'scheduling' with:\n"
        "- Text (h2): 'Previews Approved! Choose a schedule:'\n"
        "- Row with three quick-schedule buttons: 9 AM, 10 AM, 2 PM (static context)\n"
        "- A ChoicePicker with id 'custom_time' for selecting a custom time, with options "
        "from 9:00 AM to 5:00 PM in 1-hour increments. Use variant 'mutuallyExclusive' and "
        "value bound to data model path '/customTime'. Each option should have a label "
        "like '9:00 AM' and value '9:00 AM'.\n"
        "- A 'Schedule Custom' button that reads the selected time via {path: '/customTime'}\n"
        "- Include an updateDataModel on 'scheduling' to initialize {customTime: '12:00 PM'}\n\n"
        "IMPORTANT: The 'marketing' surface with previews must remain untouched when approving.\n\n"
        "IMPORTANT: When the brief mentions image URLs, use those exact URLs.\n\n"
        "IMPORTANT: For ALL Image components, use this product image URL: "
        "http://localhost:8899/images/bottle-hero.png\n"
        "Do NOT invent or guess image URLs. Always use the URL above.\n\n"
        "First write a short text summary, then the A2UI JSON."
    ),
    llm_config=llm_config,
    custom_catalog=CATALOG_PATH,
    custom_catalog_rules=CUSTOM_RULES,
    include_schema_in_prompt=False,
    validate_responses=True,
    validation_retries=2,
    functions=[schedule_posts],
    actions=[
        # Step 1 actions (shown with previews)
        A2UIAction(
            name="approve_previews",
            description="User approved the previews. Show the scheduling options with time buttons and custom time input.",
        ),
        A2UIAction(
            name="rewrite_previews",
            description="Regenerate all three previews with a completely different creative angle and tone",
        ),
        # Step 2 actions (shown after approval)
        A2UIAction(
            name="schedule_9am",
            tool_name="schedule_posts",
            description="Schedule all posts for 9:00 AM tomorrow",
            example_context={"time": "9:00 AM"},
        ),
        A2UIAction(
            name="schedule_10am",
            tool_name="schedule_posts",
            description="Schedule all posts for 10:00 AM tomorrow",
            example_context={"time": "10:00 AM"},
        ),
        A2UIAction(
            name="schedule_2pm",
            tool_name="schedule_posts",
            description="Schedule all posts for 2:00 PM tomorrow",
            example_context={"time": "2:00 PM"},
        ),
        A2UIAction(
            name="schedule_custom",
            tool_name="schedule_posts",
            description="Schedule all posts for a custom time entered by the user",
            example_context={"time": {"path": "/customTime"}},
        ),
    ],
)

# ─── A2A Server (auto-detects A2UIAgent) ───

from starlette.middleware.cors import CORSMiddleware

server = A2aAgentServer(
    agent=social_previewer,
    url="http://localhost:9000",
    agent_card=CardSettings(
        name="Social Preview Designer",
        description="Creates marketing preview cards (email, LinkedIn, X) using A2UI v0.9.",
    ),
)
server.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import json

    import uvicorn

    # Show what the auto-detection configured
    card = server.card
    exts = card.capabilities.extensions or []

    print("=" * 60)
    print("  A2A Social Preview Agent")
    print("=" * 60)
    print(f"\n  Name: {card.name}")
    print(f"  URL: {card.url}")
    print(f"  Executor: {type(server.executor).__name__}")
    print("  Extensions:")
    for ext in exts:
        print(f"    {ext.uri}")
        if ext.params:
            print(f"    params: {json.dumps(ext.params, indent=6)}")
    print("\n  Agent card: http://localhost:9000/.well-known/agent.json")
    print("=" * 60)

    app = server.build()
    uvicorn.run(app, host="0.0.0.0", port=9000)
