"""Planet-Satellite Architecture: Coding Agent

Demonstrates a planet agent that plans and delegates a small coding project.
The planet acts as architect — it plans the structure, delegates file
implementations to task satellites, writes the results to disk, and verifies
them by running Python.

All tools are real: files are read/written to the specified output directory,
and Python code is executed in a subprocess.

Architecture:
    Planet (main model)     -- plans project, delegates, writes files, runs tests
    Satellites (lighter)    -- each implements one module/file independently
    TokenMonitor            -- tracks cumulative token usage
    LoopDetector            -- flags repetitive tool-call patterns

Usage:
    export OPENAI_API_KEY=sk-...
    python playground/planet_satellite_coding.py [output_dir]

    Default output directory: ./playground_output/coding_agent/
"""

import asyncio
import os
import subprocess
import sys
from pathlib import Path

from autogen.beta.config.openai import OpenAIConfig
from autogen.beta.events import ModelMessageChunk
from autogen.beta.satellites import (
    LoopDetector,
    PlanetAgent,
    SatelliteCompleted,
    SatelliteFlag,
    SatelliteStarted,
    TaskSatelliteRequest,
    TaskSatelliteResult,
    TokenMonitor,
)
from autogen.beta.stream import MemoryStream
from autogen.beta.tools import tool

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(
    sys.argv[1] if len(sys.argv) > 1 else "./playground_output/coding_agent"
).resolve()

# ---------------------------------------------------------------------------
# Tools — all real, scoped to OUTPUT_DIR
# ---------------------------------------------------------------------------


@tool
def list_files(path: str = ".") -> str:
    """List files and directories in the project.

    Args:
        path: Relative path within the project directory (default: root).
    """
    target = OUTPUT_DIR / path
    if not target.exists():
        return f"Directory does not exist: {path}"
    entries = sorted(target.iterdir())
    lines = []
    for e in entries:
        prefix = "d " if e.is_dir() else "f "
        lines.append(f"{prefix}{e.relative_to(OUTPUT_DIR)}")
    return "\n".join(lines) if lines else "(empty directory)"


@tool
def read_file(path: str) -> str:
    """Read a file from the project directory.

    Args:
        path: Relative path to the file within the project.
    """
    target = OUTPUT_DIR / path
    if not target.exists():
        return f"File not found: {path}"
    return target.read_text()


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file in the project directory. Creates parent directories as needed.

    Args:
        path: Relative path within the project (e.g., "src/main.py").
        content: The full file content to write.
    """
    target = OUTPUT_DIR / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"Written {len(content)} bytes to {path}"


@tool
def run_python(command: str) -> str:
    """Execute a Python command or script in the project directory.

    Args:
        command: Python code to execute (passed to `python -c`).
    """
    result = subprocess.run(
        [sys.executable, "-c", command],
        cwd=str(OUTPUT_DIR),
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = ""
    if result.stdout:
        output += f"STDOUT:\n{result.stdout}\n"
    if result.stderr:
        output += f"STDERR:\n{result.stderr}\n"
    output += f"Exit code: {result.returncode}"
    return output.strip()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

planet_config = OpenAIConfig(
    model="gpt-4o",
    api_key=os.environ.get("OPENAI_API_KEY"),
    streaming=True,
)

satellite_config = OpenAIConfig(
    model="gpt-4o-mini",
    api_key=os.environ.get("OPENAI_API_KEY"),
    streaming=True,
)

# ---------------------------------------------------------------------------
# Planet agent
# ---------------------------------------------------------------------------

PLANET_PROMPT = f"""\
You are a lead software engineer. You plan, delegate, and assemble coding projects.

PROJECT DIRECTORY: {OUTPUT_DIR}
You have tools to read/write files and run Python in this directory.

Your workflow:
1. Understand the user's request and plan the project structure (which files/modules).
2. Use `spawn_tasks` to delegate each file's implementation to a task satellite.
   Give each satellite a clear spec: filename, purpose, API (function signatures),
   and how it connects to other modules. Pass all tasks with parallel=true.
3. When satellite results come back, use `write_file` to save each implementation.
4. Use `run_python` to verify the code works (import check, basic test).
5. If something fails, fix it yourself — do NOT re-delegate.

Keep your own output focused on planning and integration. Let satellites write code.
"""

planet = PlanetAgent(
    "Lead Engineer",
    prompt=PLANET_PROMPT,
    config=planet_config,
    satellite_config=satellite_config,
    satellite_prompt=(
        "You are a software engineer implementing a single module. "
        "Write clean, well-structured Python code. Include docstrings. "
        "Return ONLY the complete file content — no markdown fences, "
        "no explanation outside the code. The code must be importable."
    ),
    satellites=[
        TokenMonitor(warn_threshold=30_000, alert_threshold=80_000),
        LoopDetector(window_size=8, repeat_threshold=3),
    ],
    tools=[list_files, read_file, write_file, run_python],
)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

async def main() -> None:
    task = (
        "Build a Python CLI expense tracker. Requirements:\n"
        "- expense.py: Expense dataclass with date, amount, category, description\n"
        "- storage.py: JSON file-based storage (save/load list of expenses)\n"
        "- analytics.py: Functions for total by category, monthly summary, top expenses\n"
        "- main.py: CLI interface using argparse with commands: add, list, summary\n"
        "Make it work end-to-end. After writing files, run a quick smoke test."
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 70}")
    print(f"Coding Agent — Output: {OUTPUT_DIR}")
    print(f"  Planet model:    {planet_config.model}")
    print(f"  Satellite model: {satellite_config.model}")
    print(f"{'=' * 70}\n")

    stream = MemoryStream()
    _speaker = ""

    async def _on_event(event: object) -> None:
        nonlocal _speaker
        if isinstance(event, SatelliteStarted):
            print(
                f"  \033[35m[Natural Satellite: {event.name}]\033[0m attached",
                flush=True,
            )
        elif isinstance(event, SatelliteCompleted):
            print(
                f"  \033[35m[Natural Satellite: {event.name}]\033[0m detached",
                flush=True,
            )
        elif isinstance(event, TaskSatelliteRequest):
            _speaker = ""
            print(
                f"\n  \033[32m[spawn]\033[0m {event.satellite_name} "
                f"({satellite_config.model}): {event.task[:70]}...",
                flush=True,
            )
        elif isinstance(event, TaskSatelliteResult):
            _speaker = ""
            print(
                f"\n  \033[32m[done]\033[0m  {event.satellite_name}: "
                f"{len(event.result)} chars",
                flush=True,
            )
        elif isinstance(event, SatelliteFlag):
            print(
                f"\n  \033[33m[flag]\033[0m  [{event.severity}] {event.message}",
                flush=True,
            )
        elif isinstance(event, ModelMessageChunk):
            if _speaker != "planet":
                _speaker = "planet"
                print(f"\n\033[1;36m  [Planet: Lead Engineer ({planet_config.model})] >\033[0m\n", flush=True)
            sys.stdout.write(event.content)
            sys.stdout.flush()

    stream.subscribe(_on_event)

    print("Planning project...\n", flush=True)
    conversation = await planet.ask(task, stream=stream)
    print()

    # Show created files
    print(f"\n{'=' * 70}")
    print("Files created:")
    for f in sorted(OUTPUT_DIR.rglob("*.py")):
        print(f"  {f.relative_to(OUTPUT_DIR)}")

    for sat in planet._satellites:
        if isinstance(sat, TokenMonitor):
            print(f"\n[Token usage: {sat.total_tokens:,} total tokens]")


if __name__ == "__main__":
    asyncio.run(main())
