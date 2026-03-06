"""Planet-Satellite Architecture: Multi-Perspective Code Review

Demonstrates spawning multiple task satellites to analyse code from
different perspectives (security, performance, maintainability),
then synthesising a unified review.

Architecture:
    Planet (gpt-5.4)  -- reads the code, decides review perspectives,
                         spawns specialist satellites, writes final review.
    Satellites (gpt-5.2) -- each analyses from one perspective.
    TokenMonitor       -- tracks costs.

Usage:
    export OPENAI_API_KEY=sk-...
    python playground/planet_satellite_code_review.py
"""

import asyncio
import os

from autogen.beta.config.openai import OpenAIConfig
from autogen.beta.satellites import (
    PlanetAgent,
    SatelliteFlag,
    TaskSatelliteRequest,
    TaskSatelliteResult,
    TokenMonitor,
)
from autogen.beta.stream import MemoryStream

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

planet_config = OpenAIConfig(
    model="gpt-5.4",
    api_key=os.environ.get("OPENAI_API_KEY"),
)

satellite_config = OpenAIConfig(
    model="gpt-5.2",
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# ---------------------------------------------------------------------------
# Sample code to review
# ---------------------------------------------------------------------------

CODE_SAMPLE = '''\
import sqlite3
import os

def get_user(user_id):
    """Fetch user from database."""
    conn = sqlite3.connect(os.environ["DB_PATH"])
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")
    row = cursor.fetchone()
    conn.close()
    if row:
        return {"id": row[0], "name": row[1], "email": row[2]}
    return None

def process_users(ids):
    """Process multiple users sequentially."""
    results = []
    for uid in ids:
        user = get_user(uid)
        if user:
            results.append(user)
    return results

def export_csv(users, path):
    """Export users to CSV."""
    with open(path, "w") as f:
        f.write("id,name,email\\n")
        for u in users:
            f.write(f"{u['id']},{u['name']},{u['email']}\\n")
'''

# ---------------------------------------------------------------------------
# Planet agent
# ---------------------------------------------------------------------------

PLANET_PROMPT = f"""\
You are a senior code reviewer. You have the following code to review:

```python
{CODE_SAMPLE}
```

Your process:
1. Use `spawn_tasks` to delegate three specialist reviews in parallel:
   - "Security review: analyse the Python code for SQL injection, \
path traversal, secrets handling, and other security vulnerabilities. \
Provide specific line references and fixes."
   - "Performance review: analyse the Python code for N+1 queries, \
connection management, memory usage, and scalability issues. \
Suggest concrete optimisations."
   - "Maintainability review: analyse the Python code for error handling, \
type safety, testability, and code organisation. \
Suggest improvements following Python best practices."
2. Synthesise the satellite findings into a unified code review with:
   - A severity-ranked list of findings (Critical / Major / Minor)
   - Concrete fix suggestions for each finding
   - An overall assessment (1-2 sentences)
"""

planet = PlanetAgent(
    "Code Review Lead",
    prompt=PLANET_PROMPT,
    config=planet_config,
    satellite_config=satellite_config,
    satellite_prompt=(
        "You are a specialist code reviewer. Analyse the code provided "
        "in the task description from your specific perspective. "
        "Be concrete: reference specific lines, explain why each issue "
        "matters, and provide a corrected code snippet where appropriate. "
        "Keep your review to 200-400 words."
    ),
    satellites=[
        TokenMonitor(warn_threshold=15_000, alert_threshold=40_000),
    ],
)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

async def main() -> None:
    print(f"{'=' * 70}")
    print("Multi-Perspective Code Review")
    print(f"{'=' * 70}\n")
    print("Code under review:")
    print(CODE_SAMPLE)
    print(f"{'=' * 70}\n")

    stream = MemoryStream()

    async def _log(event: object) -> None:
        if isinstance(event, TaskSatelliteRequest):
            label = event.task[:50].replace("\n", " ")
            print(f"  [spawn] {event.satellite_name}: {label}...")
        elif isinstance(event, TaskSatelliteResult):
            print(f"  [done]  {event.satellite_name}: {len(event.result)} chars")
        elif isinstance(event, SatelliteFlag):
            print(f"  [flag]  [{event.severity}] {event.message}")

    stream.subscribe(_log)

    conversation = await planet.ask("Review the code above.", stream=stream)

    print(f"\n{'=' * 70}")
    print("REVIEW REPORT")
    print(f"{'=' * 70}\n")

    if conversation.message and conversation.message.message:
        print(conversation.message.message.content)

    for sat in planet._satellites:
        if isinstance(sat, TokenMonitor):
            print(f"\n[Token usage: {sat.total_tokens:,} total tokens]")


if __name__ == "__main__":
    asyncio.run(main())
