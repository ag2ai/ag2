"""Planet-Satellite Architecture: Data Analysis Pipeline

Demonstrates a planet agent that orchestrates multi-angle analysis of a real
dataset. The planet explores the data using tools, then delegates specialist
analyses to satellites (statistical summary, trend analysis, segment comparison,
anomaly detection). Results are synthesised into an executive report.

Tools are real — pandas operations on a bundled CSV dataset.

Architecture:
    Planet (main model)     -- explores data, plans analyses, synthesises report
    Satellites (lighter)    -- each analyses from one specialist perspective
    TokenMonitor            -- tracks cumulative token usage

Usage:
    export OPENAI_API_KEY=sk-...
    python playground/planet_satellite_data_analysis.py
"""

import asyncio
import os
import sys
from pathlib import Path

from autogen.beta.config.openai import OpenAIConfig
from autogen.beta.events import ModelMessageChunk
from autogen.beta.satellites import (
    PlanetAgent,
    SatelliteFlag,
    TaskSatelliteRequest,
    TaskSatelliteResult,
    TokenMonitor,
)
from autogen.beta.stream import MemoryStream
from autogen.beta.tools import tool

# ---------------------------------------------------------------------------
# Data path
# ---------------------------------------------------------------------------

DATA_PATH = Path(__file__).parent / "data" / "sales.csv"

# ---------------------------------------------------------------------------
# Tools — real pandas operations
# ---------------------------------------------------------------------------


@tool
def inspect_data() -> str:
    """Load the sales dataset and return its shape, columns, dtypes, and first 10 rows."""
    import pandas as pd

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    parts = [
        f"Shape: {df.shape[0]} rows x {df.shape[1]} columns",
        f"\nColumns & types:\n{df.dtypes.to_string()}",
        f"\nFirst 10 rows:\n{df.head(10).to_string()}",
        f"\nBasic stats:\n{df.describe().to_string()}",
    ]
    return "\n".join(parts)


@tool
def run_analysis(code: str) -> str:
    """Execute pandas analysis code on the sales dataset.

    The variable `df` is pre-loaded as a pandas DataFrame with columns:
    date, product, region, category, units_sold, unit_price, cost_per_unit,
    customer_segment. The `date` column is parsed as datetime.

    Your code should print() its results — the printed output is returned.

    Args:
        code: Python code to execute. `df` and `pd` are available.
    """
    import io
    from contextlib import redirect_stdout

    import pandas as pd

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    # Add computed columns that analyses commonly need
    df["revenue"] = df["units_sold"] * df["unit_price"]
    df["cost"] = df["units_sold"] * df["cost_per_unit"]
    df["profit"] = df["revenue"] - df["cost"]
    df["month"] = df["date"].dt.to_period("M")

    buf = io.StringIO()
    with redirect_stdout(buf):
        exec(code, {"df": df, "pd": pd})
    output = buf.getvalue()
    return output if output else "(no output printed)"


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

PLANET_PROMPT = """\
You are a senior data analyst. You explore datasets, plan analyses, and
produce executive-level insights.

Your workflow:
1. Use `inspect_data` to understand the dataset structure and contents.
2. Use `run_analysis` to run a few exploratory queries (e.g., revenue by
   product, monthly trends) to build your understanding.
3. Based on what you find, use `spawn_tasks` to delegate 4 specialist analyses
   in parallel:
   - Statistical summary: key metrics, distributions, top/bottom performers
   - Trend analysis: month-over-month growth, seasonal patterns, trajectory
   - Segment comparison: performance by region, customer segment, product category
   - Anomaly detection: unusual data points, outliers, inconsistencies
   IMPORTANT: Include the relevant data summaries from your exploration in each
   task description so satellites have the context they need.
4. Synthesise satellite findings into a concise executive report with:
   - Key findings (bullet points)
   - Recommended actions
   - Areas needing further investigation
"""

planet = PlanetAgent(
    "Data Analyst Lead",
    prompt=PLANET_PROMPT,
    config=planet_config,
    satellite_config=satellite_config,
    satellite_prompt=(
        "You are a specialist data analyst. Analyse the provided data from "
        "your specific perspective. Be quantitative — include specific numbers, "
        "percentages, and comparisons. Structure your findings clearly with "
        "headers. Keep to 300-500 words."
    ),
    satellites=[
        TokenMonitor(warn_threshold=25_000, alert_threshold=60_000),
    ],
    tools=[inspect_data, run_analysis],
)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

async def main() -> None:
    print(f"{'=' * 70}")
    print(f"Data Analysis Pipeline — Dataset: {DATA_PATH.name}")
    print(f"  Planet model:    {planet_config.model}")
    print(f"  Satellite model: {satellite_config.model}")
    print(f"{'=' * 70}\n")

    stream = MemoryStream()
    _speaker = ""

    async def _on_event(event: object) -> None:
        nonlocal _speaker
        if isinstance(event, TaskSatelliteRequest):
            _speaker = ""
            label = event.task[:70].replace("\n", " ")
            print(
                f"\n  \033[32m[spawn]\033[0m {event.satellite_name} "
                f"({satellite_config.model}): {label}...",
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
                print(f"\n\033[1;36m  [Planet: Data Analyst Lead ({planet_config.model})] >\033[0m\n", flush=True)
            sys.stdout.write(event.content)
            sys.stdout.flush()

    stream.subscribe(_on_event)

    print("Exploring data...\n", flush=True)
    conversation = await planet.ask(
        "Analyse the sales dataset and produce an executive insights report.",
        stream=stream,
    )
    print()

    for sat in planet._satellites:
        if isinstance(sat, TokenMonitor):
            print(f"\n[Token usage: {sat.total_tokens:,} total tokens]")


if __name__ == "__main__":
    asyncio.run(main())
