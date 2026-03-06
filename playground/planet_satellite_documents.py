"""Planet-Satellite Architecture: Multi-Document Contract Analyser

Demonstrates real file reading tools for processing multiple documents.
The planet reads a set of contracts, then delegates parallel extraction
tasks: key terms, financial obligations, risk clauses, and cross-document
consistency checks.

A custom "completeness" satellite monitors that all documents are referenced
in the analysis — catching cases where a document is accidentally skipped.

Architecture:
    Planet (main model)     -- reads contracts, plans extraction, synthesises
    Satellites (lighter)    -- each extracts from one analytical dimension
    TokenMonitor            -- tracks token usage
    CompletenessMonitor     -- flags if any document is not covered

Usage:
    export OPENAI_API_KEY=sk-...
    python playground/planet_satellite_documents.py
"""

import asyncio
import os
import sys
from pathlib import Path

from autogen.beta.annotations import Context
from autogen.beta.config.openai import OpenAIConfig
from autogen.beta.events import BaseEvent, ModelMessage, ModelMessageChunk
from autogen.beta.satellites import (
    BaseSatellite,
    OnEvent,
    PlanetAgent,
    SatelliteFlag,
    Severity,
    TaskSatelliteRequest,
    TaskSatelliteResult,
    TokenMonitor,
)
from autogen.beta.stream import MemoryStream
from autogen.beta.tools import tool

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------

CONTRACTS_DIR = Path(__file__).parent / "data" / "contracts"

# ---------------------------------------------------------------------------
# Tools — real file operations
# ---------------------------------------------------------------------------


@tool
def list_documents() -> str:
    """List all available contract documents."""
    files = sorted(CONTRACTS_DIR.glob("*.txt"))
    if not files:
        return "No documents found."
    lines = []
    for f in files:
        size = f.stat().st_size
        lines.append(f"  {f.name} ({size:,} bytes)")
    return f"Available contracts ({len(files)} files):\n" + "\n".join(lines)


@tool
def read_document(filename: str) -> str:
    """Read a contract document by filename.

    Args:
        filename: The document filename (e.g., 'saas_subscription.txt').
    """
    path = CONTRACTS_DIR / filename
    if not path.exists():
        available = [f.name for f in CONTRACTS_DIR.glob("*.txt")]
        return f"File not found: {filename}. Available: {available}"
    return f"=== {filename} ===\n\n{path.read_text()}"


# ---------------------------------------------------------------------------
# Custom satellite: completeness monitor
# ---------------------------------------------------------------------------

EXPECTED_DOCS = {"saas_subscription", "data_processing", "consulting_services"}


class CompletenessMonitor(BaseSatellite):
    """Monitors that all documents are referenced in the analysis.

    After seeing enough model output, flags if any expected document
    hasn't been mentioned — catching accidental omissions.
    """

    def __init__(self) -> None:
        super().__init__("completeness-monitor", trigger=OnEvent(ModelMessage))
        self._all_text = ""
        self._flagged = False

    async def process(
        self, events: list[BaseEvent], ctx: Context
    ) -> SatelliteFlag | None:
        for event in events:
            if isinstance(event, ModelMessage):
                self._all_text += event.content.lower()

        # Only check after substantial output has been generated
        if len(self._all_text) > 2000 and not self._flagged:
            missing = []
            for doc in EXPECTED_DOCS:
                # Check for various forms of the document name
                name_variants = [doc, doc.replace("_", " "), doc.replace("_", "-")]
                if not any(v in self._all_text for v in name_variants):
                    missing.append(doc)

            if missing:
                self._flagged = True
                return SatelliteFlag(
                    source=self.name,
                    severity=Severity.WARNING,
                    message=(
                        f"Documents not yet referenced in analysis: "
                        f"{', '.join(missing)}. Ensure all contracts are "
                        f"covered in the final report."
                    ),
                )
        return None


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
You are a senior legal analyst reviewing a set of related contracts between
two parties. Your job is to extract key information and flag risks.

Your workflow:
1. Use `list_documents` to see what contracts are available.
2. Use `read_document` to read ALL contracts (read them in parallel if possible,
   or one at a time — but read all of them before proceeding).
3. Use `spawn_tasks` to delegate FOUR parallel extraction tasks. IMPORTANT:
   Include the FULL text of ALL contracts in each task description so each
   satellite has complete context.

   The four extraction tasks:
   a) "KEY TERMS EXTRACTION: Extract all key commercial terms from these
       contracts: pricing, payment terms, term length, renewal conditions,
       termination provisions, and notice periods. Present in a comparison
       table format."

   b) "FINANCIAL OBLIGATIONS: Identify all financial obligations, caps,
       penalties, and fee structures across all contracts. Calculate total
       annual cost. Flag any escalation clauses or hidden costs."

   c) "RISK ANALYSIS: Identify liability caps, indemnification obligations,
       limitation of liability carve-outs, data breach responsibilities,
       and insurance requirements. Rate each risk as High/Medium/Low."

   d) "CROSS-DOCUMENT CONSISTENCY: Check for inconsistencies, conflicts,
       or gaps between the three contracts. Are dates aligned? Do data
       handling obligations in the DPA match what the SaaS agreement says?
       Does the consulting agreement's IP clause conflict with the SaaS
       license? Flag any issues."

4. Synthesise into an executive contract review:
   - Summary of the relationship and total commitment
   - Key terms comparison table
   - Top risks (ranked by severity)
   - Inconsistencies found
   - Recommended negotiation points
"""

planet = PlanetAgent(
    "Contract Reviewer",
    prompt=PLANET_PROMPT,
    config=planet_config,
    satellite_config=satellite_config,
    satellite_prompt=(
        "You are a legal analyst specialising in contract review. Analyse "
        "the provided contracts from your specific perspective. Be precise — "
        "quote specific clauses and section numbers. Use tables where "
        "appropriate. Keep to 400-600 words."
    ),
    satellites=[
        TokenMonitor(warn_threshold=30_000, alert_threshold=70_000),
        CompletenessMonitor(),
    ],
    tools=[list_documents, read_document],
)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

async def main() -> None:
    print(f"{'=' * 70}")
    print("Multi-Document Contract Analysis")
    print(f"{'=' * 70}\n")

    # Show available documents
    for f in sorted(CONTRACTS_DIR.glob("*.txt")):
        print(f"  {f.name}")
    print()

    stream = MemoryStream()
    _speaker = ""

    async def _on_event(event: object) -> None:
        nonlocal _speaker
        if isinstance(event, TaskSatelliteRequest):
            _speaker = ""
            label = event.task[:70].replace("\n", " ")
            print(
                f"\n  \033[32m[spawn]\033[0m {event.satellite_name}: {label}...",
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
                print(f"\n\033[1;36m  [Planet: Contract Reviewer] >\033[0m\n", flush=True)
            sys.stdout.write(event.content)
            sys.stdout.flush()

    stream.subscribe(_on_event)

    print("Reviewing contracts...\n", flush=True)
    conversation = await planet.ask(
        "Review the complete set of contracts between Acme Corp and GlobalTech Inc. "
        "Produce an executive summary with key terms, risks, and recommendations.",
        stream=stream,
    )
    print()

    for sat in planet._satellites:
        if isinstance(sat, TokenMonitor):
            print(f"\n[Token usage: {sat.total_tokens:,} total tokens]")


if __name__ == "__main__":
    asyncio.run(main())
