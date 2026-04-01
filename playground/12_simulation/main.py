#!/usr/bin/env python3
"""Decision Simulation — scale-out network with many autonomous personas.

Category 12: Simulation (Many-Actor Network)

A decision is broadcast to a network of persona actors — each representing
a different organizational stakeholder (CFO, CTO, CMO, Legal, HR, Ops,
Customer Advocate, Competitor Analyst, Board Member). Each persona
independently analyzes the decision's impact from their domain perspective.
An analyst actor then synthesizes all perspectives into a unified
risk/opportunity report.

This demonstrates that the network naturally scales to any number of actors
— the coordinator discovers and delegates to all of them dynamically.

Usage:
    python playground/12_simulation/main.py
    python playground/12_simulation/main.py --scenario 2
    python playground/12_simulation/main.py --scenario 3
    python playground/12_simulation/main.py --model gemini-3-flash-preview
"""

from __future__ import annotations

import asyncio
import random
import sys
import time
from datetime import datetime

from autogen.beta import Actor, LoopDetector, TokenMonitor, tool
from autogen.beta.config.gemini import GeminiConfig
from autogen.beta.network import (
    DelegationRejected,
    DelegationRequest,
    DelegationResult,
    Network,
    TelemetryPlugin,
)

# =====================================================================
# ANSI helpers
# =====================================================================

_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"
_CYAN = "\033[96m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_MAGENTA = "\033[95m"
_BLUE = "\033[94m"
_WHITE = "\033[97m"


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _wrap(text: str, indent: int = 4, width: int = 72) -> str:
    words = text.split()
    lines: list[str] = []
    line = " " * indent
    for w in words:
        if len(line) + len(w) + 1 > width:
            lines.append(line)
            line = " " * indent + w
        else:
            line += (" " + w) if len(line) > indent else w
    if len(line) > indent:
        lines.append(line)
    return "\n".join(lines)


# =====================================================================
# Tools — Coordinator
# =====================================================================


@tool
async def structure_decision(
    decision: str,
    context: str,
    time_horizon: str,
) -> str:
    """Structure a business decision for simulation analysis.

    Args:
        decision: The decision being evaluated.
        context: Business context and background.
        time_horizon: Expected timeline (e.g. "6 months", "2 years").
    """
    return (
        f"SIMULATION BRIEF\n"
        f"{'=' * 50}\n"
        f"  Decision: {decision}\n"
        f"  Context: {context}\n"
        f"  Time Horizon: {time_horizon}\n"
        f"  Simulation ID: SIM-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000, 9999)}\n"
        f"  Status: READY FOR DISTRIBUTION\n"
        f"  Requested Perspectives: ALL REGISTERED PERSONAS"
    )


# =====================================================================
# Tools — Persona Actors (one domain-specific tool each)
# =====================================================================


@tool
async def run_financial_model(
    decision: str,
    revenue_impact_pct: float,
    cost_impact_pct: float,
    payback_months: int,
) -> str:
    """Run financial projections for a business decision.

    Args:
        decision: The decision being analyzed.
        revenue_impact_pct: Estimated revenue impact as percentage (-100 to +100).
        cost_impact_pct: Estimated cost impact as percentage (-100 to +100).
        payback_months: Expected months to break even.
    """
    base_revenue = 50_000_000
    projected = base_revenue * (1 + revenue_impact_pct / 100)
    cost_delta = base_revenue * 0.6 * (cost_impact_pct / 100)
    return (
        f"FINANCIAL ANALYSIS\n"
        f"{'=' * 50}\n"
        f"  Current ARR: ${base_revenue:,.0f}\n"
        f"  Projected ARR: ${projected:,.0f} ({revenue_impact_pct:+.1f}%)\n"
        f"  Cost Delta: ${cost_delta:,.0f} ({cost_impact_pct:+.1f}%)\n"
        f"  Payback Period: {payback_months} months\n"
        f"  NPV (3yr): ${(projected - base_revenue) * 2.5 - abs(cost_delta) * 1.5:,.0f}\n"
        f"  Risk Rating: {'HIGH' if abs(revenue_impact_pct) > 15 else 'MODERATE' if abs(revenue_impact_pct) > 5 else 'LOW'}"
    )


@tool
async def analyze_brand_impact(
    decision: str,
    brand_alignment_score: int,
    market_perception_risk: str,
) -> str:
    """Analyze brand and marketing impact of a decision.

    Args:
        decision: The decision being analyzed.
        brand_alignment_score: How well this aligns with brand (1-10).
        market_perception_risk: Risk level for market perception (low/medium/high).
    """
    return (
        f"MARKETING & BRAND ANALYSIS\n"
        f"{'=' * 50}\n"
        f"  Brand Alignment: {brand_alignment_score}/10\n"
        f"  Market Perception Risk: {market_perception_risk.upper()}\n"
        f"  Recommended Messaging: {'Proactive announcement' if brand_alignment_score >= 7 else 'Careful positioning needed'}\n"
        f"  PR Preparation: {'Standard' if market_perception_risk == 'low' else 'Enhanced crisis readiness'}\n"
        f"  Competitor Narrative Risk: {'LOW' if brand_alignment_score >= 8 else 'MODERATE'}"
    )


@tool
async def assess_technical_feasibility(
    decision: str,
    engineering_months: int,
    infrastructure_changes: str,
    risk_factors: str,
) -> str:
    """Assess technical feasibility and engineering requirements.

    Args:
        decision: The decision being analyzed.
        engineering_months: Estimated engineering effort in person-months.
        infrastructure_changes: Required infrastructure changes.
        risk_factors: Key technical risk factors.
    """
    return (
        f"TECHNICAL FEASIBILITY\n"
        f"{'=' * 50}\n"
        f"  Engineering Effort: {engineering_months} person-months\n"
        f"  Team Size Needed: {max(2, engineering_months // 3)} engineers\n"
        f"  Infrastructure: {infrastructure_changes}\n"
        f"  Risk Factors: {risk_factors}\n"
        f"  Technical Debt Impact: {'Significant' if engineering_months > 12 else 'Manageable'}\n"
        f"  Feasibility Rating: {'FEASIBLE' if engineering_months <= 18 else 'CHALLENGING'}"
    )


@tool
async def assess_workforce_impact(
    decision: str,
    headcount_change: int,
    reskilling_needed: bool,
    morale_risk: str,
) -> str:
    """Assess workforce and talent implications.

    Args:
        decision: The decision being analyzed.
        headcount_change: Net headcount change (positive = hiring, negative = reduction).
        reskilling_needed: Whether existing staff need reskilling.
        morale_risk: Expected impact on team morale (low/medium/high).
    """
    return (
        f"WORKFORCE ANALYSIS\n"
        f"{'=' * 50}\n"
        f"  Headcount Change: {headcount_change:+d}\n"
        f"  {'Hiring' if headcount_change > 0 else 'Reduction'} Timeline: {abs(headcount_change) * 2} weeks\n"
        f"  Reskilling: {'Required' if reskilling_needed else 'Not required'}\n"
        f"  Morale Risk: {morale_risk.upper()}\n"
        f"  Retention Risk: {'ELEVATED' if morale_risk == 'high' or headcount_change < -5 else 'NORMAL'}\n"
        f"  Culture Impact: {'Significant restructuring' if abs(headcount_change) > 10 else 'Incremental adjustment'}"
    )


@tool
async def check_regulatory_compliance(
    decision: str,
    jurisdictions: str,
    data_privacy_impact: bool,
    contract_implications: str,
) -> str:
    """Check legal and regulatory compliance implications.

    Args:
        decision: The decision being analyzed.
        jurisdictions: Affected legal jurisdictions.
        data_privacy_impact: Whether decision affects data privacy posture.
        contract_implications: Impact on existing contracts/agreements.
    """
    return (
        f"LEGAL & REGULATORY ANALYSIS\n"
        f"{'=' * 50}\n"
        f"  Jurisdictions: {jurisdictions}\n"
        f"  Data Privacy Impact: {'YES — GDPR/CCPA review required' if data_privacy_impact else 'Minimal'}\n"
        f"  Contract Implications: {contract_implications}\n"
        f"  Regulatory Filing: {'Required' if data_privacy_impact else 'Not required'}\n"
        f"  Legal Review Timeline: {random.randint(2, 6)} weeks\n"
        f"  Risk of Litigation: {'MODERATE' if data_privacy_impact else 'LOW'}"
    )


@tool
async def predict_customer_impact(
    decision: str,
    affected_segment: str,
    satisfaction_change: int,
    churn_risk_pct: float,
) -> str:
    """Predict impact on customer satisfaction and retention.

    Args:
        decision: The decision being analyzed.
        affected_segment: Primary customer segment affected.
        satisfaction_change: Expected NPS change (-100 to +100).
        churn_risk_pct: Estimated additional churn risk as percentage.
    """
    current_nps = 42
    return (
        f"CUSTOMER IMPACT ANALYSIS\n"
        f"{'=' * 50}\n"
        f"  Affected Segment: {affected_segment}\n"
        f"  Current NPS: {current_nps}\n"
        f"  Projected NPS: {current_nps + satisfaction_change}\n"
        f"  Churn Risk: {churn_risk_pct:.1f}% additional\n"
        f"  Revenue at Risk: ${50_000_000 * churn_risk_pct / 100:,.0f}\n"
        f"  Mitigation: {'Proactive outreach critical' if churn_risk_pct > 3 else 'Standard communication'}"
    )


@tool
async def analyze_supply_chain_impact(
    decision: str,
    vendor_changes: str,
    lead_time_weeks: int,
    capacity_impact_pct: float,
) -> str:
    """Analyze operational and supply chain implications.

    Args:
        decision: The decision being analyzed.
        vendor_changes: Required changes to vendor relationships.
        lead_time_weeks: Additional lead time needed.
        capacity_impact_pct: Impact on operational capacity as percentage.
    """
    return (
        f"OPERATIONS ANALYSIS\n"
        f"{'=' * 50}\n"
        f"  Vendor Changes: {vendor_changes}\n"
        f"  Additional Lead Time: {lead_time_weeks} weeks\n"
        f"  Capacity Impact: {capacity_impact_pct:+.1f}%\n"
        f"  Process Changes: {'Major overhaul' if abs(capacity_impact_pct) > 20 else 'Incremental updates'}\n"
        f"  SLA Risk: {'ELEVATED' if lead_time_weeks > 8 else 'MANAGEABLE'}\n"
        f"  Rollback Complexity: {'HIGH' if abs(capacity_impact_pct) > 15 else 'LOW'}"
    )


@tool
async def predict_competitor_response(
    decision: str,
    primary_competitors: str,
    likely_response: str,
    market_share_impact_pct: float,
) -> str:
    """Predict how competitors will respond to this decision.

    Args:
        decision: The decision being analyzed.
        primary_competitors: Key competitors who will react.
        likely_response: Expected competitive response.
        market_share_impact_pct: Expected market share change as percentage.
    """
    return (
        f"COMPETITIVE ANALYSIS\n"
        f"{'=' * 50}\n"
        f"  Primary Competitors: {primary_competitors}\n"
        f"  Likely Response: {likely_response}\n"
        f"  Market Share Impact: {market_share_impact_pct:+.1f}%\n"
        f"  First-Mover Advantage: {'YES' if market_share_impact_pct > 0 else 'NO'}\n"
        f"  Response Timeline: {random.randint(1, 6)} months\n"
        f"  Competitive Moat: {'Strengthened' if market_share_impact_pct > 2 else 'Unchanged'}"
    )


@tool
async def evaluate_strategic_alignment(
    decision: str,
    vision_alignment_score: int,
    investor_sentiment: str,
    long_term_value: str,
) -> str:
    """Evaluate from a board/strategic oversight perspective.

    Args:
        decision: The decision being analyzed.
        vision_alignment_score: Alignment with company vision (1-10).
        investor_sentiment: Expected investor reaction (positive/neutral/negative).
        long_term_value: Assessment of long-term strategic value.
    """
    return (
        f"STRATEGIC / BOARD PERSPECTIVE\n"
        f"{'=' * 50}\n"
        f"  Vision Alignment: {vision_alignment_score}/10\n"
        f"  Investor Sentiment: {investor_sentiment.upper()}\n"
        f"  Long-term Value: {long_term_value}\n"
        f"  Fiduciary Risk: {'REVIEW NEEDED' if investor_sentiment == 'negative' else 'ACCEPTABLE'}\n"
        f"  Board Approval: {'Likely' if vision_alignment_score >= 7 else 'Requires discussion'}\n"
        f"  Governance Impact: {'Material' if investor_sentiment == 'negative' else 'Routine'}"
    )


# =====================================================================
# Tools — Analyst
# =====================================================================


@tool
async def synthesize_perspectives(
    total_perspectives: int,
    support_count: int,
    oppose_count: int,
    neutral_count: int,
    key_risks: str,
    key_opportunities: str,
) -> str:
    """Synthesize all simulation perspectives into a unified view.

    Args:
        total_perspectives: Total number of perspectives analyzed.
        support_count: Number of perspectives supporting the decision.
        oppose_count: Number of perspectives opposing.
        neutral_count: Number of neutral perspectives.
        key_risks: Top 3 risks across all perspectives.
        key_opportunities: Top 3 opportunities across all perspectives.
    """
    consensus = (
        "STRONG SUPPORT"
        if support_count > total_perspectives * 0.7
        else "MODERATE SUPPORT"
        if support_count > total_perspectives * 0.5
        else "DIVIDED"
        if support_count >= oppose_count
        else "OPPOSITION MAJORITY"
    )
    return (
        f"SIMULATION SYNTHESIS\n"
        f"{'=' * 50}\n"
        f"  Perspectives Analyzed: {total_perspectives}\n"
        f"  Support / Oppose / Neutral: {support_count} / {oppose_count} / {neutral_count}\n"
        f"  Consensus: {consensus}\n"
        f"  Confidence Level: {min(95, 60 + total_perspectives * 3)}%\n"
        f"\n  KEY RISKS:\n    {key_risks}\n"
        f"\n  KEY OPPORTUNITIES:\n    {key_opportunities}\n"
        f"\n  Simulation Quality: {'HIGH' if total_perspectives >= 7 else 'MODERATE'} "
        f"({total_perspectives} independent analyses)"
    )


@tool
async def generate_recommendation(
    decision: str,
    recommendation: str,
    conditions: str,
    immediate_actions: str,
) -> str:
    """Generate final recommendation based on simulation results.

    Args:
        decision: The original decision evaluated.
        recommendation: Final recommendation (proceed/proceed-with-conditions/defer/reject).
        conditions: Key conditions or prerequisites for proceeding.
        immediate_actions: Recommended immediate next steps.
    """
    rec_upper = recommendation.upper().replace("-", " ")
    return (
        f"SIMULATION RECOMMENDATION\n"
        f"{'=' * 50}\n"
        f"  Decision: {decision}\n"
        f"  Recommendation: {rec_upper}\n"
        f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"\n  CONDITIONS:\n    {conditions}\n"
        f"\n  IMMEDIATE ACTIONS:\n    {immediate_actions}\n"
        f"\n  Note: This recommendation is based on simulated analysis.\n"
        f"  Real-world validation recommended before execution."
    )


# =====================================================================
# Persona definitions
# =====================================================================

PERSONAS = [
    {
        "name": "cfo",
        "title": "Chief Financial Officer",
        "prompt": (
            "You are the CFO in a decision simulation.\n\n"
            "Analyze the proposed decision STRICTLY from a financial perspective. "
            "Consider revenue impact, cost structure, cash flow, payback period, "
            "and financial risk. Use your run_financial_model tool to produce "
            "a quantitative analysis.\n\n"
            "Be specific with numbers. State whether you SUPPORT, OPPOSE, or are "
            "NEUTRAL on the decision, and why."
        ),
        "tools": [run_financial_model],
        "capabilities": ["simulation", "finance"],
    },
    {
        "name": "cmo",
        "title": "Chief Marketing Officer",
        "prompt": (
            "You are the CMO in a decision simulation.\n\n"
            "Analyze the proposed decision from a marketing and brand perspective. "
            "Consider brand alignment, market perception, competitive positioning, "
            "and messaging strategy. Use your analyze_brand_impact tool.\n\n"
            "State whether you SUPPORT, OPPOSE, or are NEUTRAL on the decision, "
            "and explain your marketing rationale."
        ),
        "tools": [analyze_brand_impact],
        "capabilities": ["simulation", "marketing"],
    },
    {
        "name": "cto",
        "title": "Chief Technology Officer",
        "prompt": (
            "You are the CTO in a decision simulation.\n\n"
            "Analyze the proposed decision from a technical feasibility standpoint. "
            "Consider engineering effort, infrastructure needs, technical debt, "
            "and risk factors. Use your assess_technical_feasibility tool.\n\n"
            "State whether you SUPPORT, OPPOSE, or are NEUTRAL on the decision, "
            "and explain the technical implications."
        ),
        "tools": [assess_technical_feasibility],
        "capabilities": ["simulation", "technology"],
    },
    {
        "name": "hr-director",
        "title": "HR Director",
        "prompt": (
            "You are the HR Director in a decision simulation.\n\n"
            "Analyze the proposed decision from a workforce and talent perspective. "
            "Consider headcount changes, reskilling needs, morale impact, retention "
            "risk, and culture effects. Use your assess_workforce_impact tool.\n\n"
            "State whether you SUPPORT, OPPOSE, or are NEUTRAL on the decision, "
            "and explain workforce implications."
        ),
        "tools": [assess_workforce_impact],
        "capabilities": ["simulation", "human-resources"],
    },
    {
        "name": "legal-counsel",
        "title": "General Counsel",
        "prompt": (
            "You are the General Counsel in a decision simulation.\n\n"
            "Analyze the proposed decision from a legal and regulatory perspective. "
            "Consider compliance requirements, data privacy, contractual obligations, "
            "and litigation risk. Use your check_regulatory_compliance tool.\n\n"
            "State whether you SUPPORT, OPPOSE, or are NEUTRAL on the decision, "
            "and flag any legal risks or requirements."
        ),
        "tools": [check_regulatory_compliance],
        "capabilities": ["simulation", "legal"],
    },
    {
        "name": "customer-advocate",
        "title": "VP of Customer Success",
        "prompt": (
            "You are the VP of Customer Success in a decision simulation.\n\n"
            "Analyze the proposed decision from the customer's perspective. "
            "Consider customer satisfaction, churn risk, NPS impact, and "
            "segment-specific effects. Use your predict_customer_impact tool.\n\n"
            "State whether you SUPPORT, OPPOSE, or are NEUTRAL on the decision, "
            "and explain the customer impact."
        ),
        "tools": [predict_customer_impact],
        "capabilities": ["simulation", "customer-success"],
    },
    {
        "name": "operations-lead",
        "title": "VP of Operations",
        "prompt": (
            "You are the VP of Operations in a decision simulation.\n\n"
            "Analyze the proposed decision from an operational standpoint. "
            "Consider supply chain, vendor relationships, capacity, SLAs, "
            "and rollback complexity. Use your analyze_supply_chain_impact tool.\n\n"
            "State whether you SUPPORT, OPPOSE, or are NEUTRAL on the decision, "
            "and explain operational implications."
        ),
        "tools": [analyze_supply_chain_impact],
        "capabilities": ["simulation", "operations"],
    },
    {
        "name": "competitor-analyst",
        "title": "Competitive Intelligence Lead",
        "prompt": (
            "You are the Competitive Intelligence Lead in a decision simulation.\n\n"
            "Analyze how competitors will respond to this decision. Consider "
            "market share dynamics, first-mover advantage, and competitive moat. "
            "Use your predict_competitor_response tool.\n\n"
            "State whether you SUPPORT, OPPOSE, or are NEUTRAL on the decision, "
            "and explain the competitive landscape implications."
        ),
        "tools": [predict_competitor_response],
        "capabilities": ["simulation", "competitive-intelligence"],
    },
    {
        "name": "board-member",
        "title": "Independent Board Director",
        "prompt": (
            "You are an Independent Board Director in a decision simulation.\n\n"
            "Evaluate the proposed decision from a strategic governance perspective. "
            "Consider alignment with company vision, investor sentiment, long-term "
            "value creation, and fiduciary responsibility. Use your "
            "evaluate_strategic_alignment tool.\n\n"
            "State whether you SUPPORT, OPPOSE, or are NEUTRAL on the decision, "
            "and explain your strategic rationale."
        ),
        "tools": [evaluate_strategic_alignment],
        "capabilities": ["simulation", "strategy"],
    },
]


# =====================================================================
# Scenarios
# =====================================================================

SCENARIOS = {
    1: (
        "SaaS Price Increase",
        "We are considering raising our SaaS subscription prices by 25% across "
        "all tiers. Our product is a B2B analytics platform with 2,000 enterprise "
        "customers, $50M ARR, and 90% gross margins. Competitors have not raised "
        "prices recently. We haven't adjusted pricing in 3 years. Customer "
        "satisfaction (NPS) is 42. Engineering costs have risen 30% due to AI "
        "infrastructure investment.",
    ),
    2: (
        "European Market Expansion",
        "We are evaluating expanding into the European market by opening a new "
        "data center in Frankfurt and hiring a 15-person EU sales team. Current "
        "revenue is 95% North America. We have 200 inbound EU leads per quarter "
        "that we currently cannot serve due to data residency requirements (GDPR). "
        "Estimated investment: $8M over 18 months. Three competitors already "
        "operate in EU.",
    ),
    3: (
        "Legacy Product Sunset",
        "We are considering discontinuing our legacy on-premise product line to "
        "go fully cloud-native. The on-prem product represents 20% of revenue "
        "($10M) but consumes 40% of engineering resources. 150 enterprise "
        "customers depend on it, including 3 Fortune 500 accounts. Migration "
        "path exists but requires 6-12 months per customer. Cloud product has "
        "85% feature parity.",
    ),
}


# =====================================================================
# Main
# =====================================================================


async def main() -> None:
    # -- Parse CLI args -----------------------------------------------
    model = "gemini-3.1-pro-preview"
    scenario_num = 1
    args = sys.argv[1:]
    i = 0
    custom_msg = None
    while i < len(args):
        if args[i] == "--scenario" and i + 1 < len(args):
            scenario_num = int(args[i + 1])
            i += 2
        elif args[i] == "--model" and i + 1 < len(args):
            model = args[i + 1]
            i += 2
        elif not args[i].startswith("--"):
            custom_msg = args[i]
            i += 1
        else:
            i += 1

    if custom_msg:
        title, message = "Custom Decision", custom_msg
    else:
        title, message = SCENARIOS[scenario_num]

    # -- Create persona actors ----------------------------------------
    persona_actors: list[Actor] = []
    for p in PERSONAS:
        actor = Actor(
            p["name"],
            prompt=p["prompt"],
            config=GeminiConfig(model=model, temperature=0.3),
            tools=p["tools"],
            observers=[TokenMonitor(warn_threshold=10_000, alert_threshold=30_000)],
        )
        persona_actors.append(actor)

    # -- Create coordinator -------------------------------------------
    coordinator = Actor(
        "coordinator",
        prompt=(
            "You are a Simulation Coordinator.\n\n"
            "Your job is to take a business decision and gather independent analyses "
            "from every available simulation persona in the network.\n\n"
            "Steps:\n"
            "1. Use structure_decision to formalize the decision\n"
            "2. Use discover_agents with capability='simulation' to find all personas\n"
            "3. Use delegate_to to send the decision to EVERY persona found — send "
            "each the full decision context and ask for their domain-specific analysis. "
            "Include a request that they state SUPPORT/OPPOSE/NEUTRAL.\n"
            "4. After ALL personas have responded, compile their analyses\n"
            "5. Use delegate_to to send ALL collected analyses to the 'analyst' agent, "
            "asking for a unified synthesis and recommendation\n"
            "6. Return the analyst's final report\n\n"
            "IMPORTANT: You MUST delegate to ALL simulation personas, not just some. "
            "The value of the simulation depends on comprehensive perspective coverage."
        ),
        config=GeminiConfig(model=model, temperature=0.2),
        tools=[structure_decision],
        observers=[
            TokenMonitor(warn_threshold=15_000, alert_threshold=50_000),
            LoopDetector(repeat_threshold=3),
        ],
    )

    # -- Create analyst -----------------------------------------------
    analyst = Actor(
        "analyst",
        prompt=(
            "You are a Simulation Analyst.\n\n"
            "You receive the compiled analyses from all simulation personas. "
            "Your job is to synthesize them into a clear, actionable report.\n\n"
            "Steps:\n"
            "1. Count how many perspectives support, oppose, or are neutral\n"
            "2. Use synthesize_perspectives to create the synthesis\n"
            "3. Use generate_recommendation with your final recommendation\n"
            "4. Return a comprehensive report covering: consensus, key risks, "
            "key opportunities, conditions for proceeding, and next steps\n\n"
            "Be balanced and objective. Weigh all perspectives fairly."
        ),
        config=GeminiConfig(model=model, temperature=0.2),
        tools=[synthesize_perspectives, generate_recommendation],
        observers=[
            TokenMonitor(warn_threshold=15_000, alert_threshold=50_000),
            LoopDetector(repeat_threshold=3),
        ],
    )

    # -- Create network -----------------------------------------------
    telemetry = TelemetryPlugin()

    network = Network(
        plugins=[telemetry],
        max_delegation_depth=4,
    )

    # Register coordinator
    await network.register(
        coordinator,
        capabilities=["coordination"],
        description="Distributes decisions to simulation personas and collects results",
    )

    # Register all persona actors
    for actor, p in zip(persona_actors, PERSONAS):
        await network.register(
            actor,
            capabilities=p["capabilities"],
            description=f"Simulation persona: {p['title']} — analyzes from {p['capabilities'][-1]} perspective",
        )

    # Register analyst
    await network.register(
        analyst,
        capabilities=["analysis", "synthesis"],
        description="Synthesizes all simulation perspectives into a unified recommendation",
    )

    # -- Subscribe to hub stream for live logging ---------------------
    delegation_count = 0

    async def _on_hub_event(event: object) -> None:
        nonlocal delegation_count
        ts = _ts()
        if isinstance(event, DelegationRequest):
            delegation_count += 1
            target = event.target
            is_persona = any(p["name"] == target for p in PERSONAS)
            color = _BLUE if is_persona else _GREEN if target == "analyst" else _MAGENTA
            label = f"[{delegation_count:2d}]"
            print(
                f"  {_DIM}{ts}{_RESET}  "
                f"{color}{_BOLD}DELEGATE{_RESET} {_DIM}{label}{_RESET}  "
                f"{color}{event.source} -> {event.target}{_RESET}"
            )
            preview = event.task[:100].replace("\n", " ")
            print(
                f"  {_DIM}{' ' * 12}{_RESET}            {_DIM}{preview}{'...' if len(event.task) > 100 else ''}{_RESET}"
            )
        elif isinstance(event, DelegationResult):
            target = event.target
            is_persona = any(p["name"] == target for p in PERSONAS)
            color = _BLUE if is_persona else _GREEN if target == "analyst" else _MAGENTA
            stance = ""
            body = (event.result or "").upper()
            if "SUPPORT" in body and "OPPOSE" not in body:
                stance = f" {_GREEN}[SUPPORT]{_RESET}"
            elif "OPPOSE" in body:
                stance = f" {_RED}[OPPOSE]{_RESET}"
            elif "NEUTRAL" in body:
                stance = f" {_YELLOW}[NEUTRAL]{_RESET}"
            print(
                f"  {_DIM}{ts}{_RESET}  "
                f"{color}{_BOLD}COMPLETE{_RESET}       "
                f"{color}{event.target} -> {event.source}{_RESET}{stance}"
            )
        elif isinstance(event, DelegationRejected):
            print(
                f"  {_DIM}{ts}{_RESET}  "
                f"{_RED}{_BOLD}REJECTED{_RESET}       "
                f"{_RED}{event.source} -> {event.target}: {event.reason}{_RESET}"
            )

    network.hub.stream.subscribe(_on_hub_event)

    # -- Print header -------------------------------------------------
    total_actors = 1 + len(PERSONAS) + 1  # coordinator + personas + analyst
    print()
    print(f"  {_BOLD}{'=' * 60}{_RESET}")
    print(f"  {_BOLD}DECISION SIMULATION{_RESET}  {_DIM}(12_simulation){_RESET}")
    print(f"  {_BOLD}{'=' * 60}{_RESET}")
    print()
    print(f"  {_CYAN}Category:{_RESET}  Many-Actor Network (Scale-Out Simulation)")
    print(f"  {_CYAN}Scenario:{_RESET}  {title}")
    print(f"  {_CYAN}Model:{_RESET}     {model}")
    print(f"  {_CYAN}Actors:{_RESET}    {total_actors} total")
    print(f"  {_DIM}             coordinator -> 9 personas -> analyst{_RESET}")
    print(f"  {_CYAN}Personas:{_RESET}  {', '.join(p['name'] for p in PERSONAS)}")
    print()
    print(f"  {_BOLD}Decision:{_RESET}")
    print(_wrap(message, indent=4, width=72))
    print()
    print(f"  {_DIM}{'─' * 60}{_RESET}")
    print(
        f"  {_BOLD}Log legend:{_RESET}  "
        f"{_BLUE}persona{_RESET}  "
        f"{_GREEN}analyst{_RESET}  "
        f"{_MAGENTA}coordinator{_RESET}  "
        f"{_GREEN}SUPPORT{_RESET}/{_RED}OPPOSE{_RESET}/{_YELLOW}NEUTRAL{_RESET}"
    )
    print(f"  {_DIM}{'─' * 60}{_RESET}")
    print()

    # -- Run ----------------------------------------------------------
    t0 = time.monotonic()
    reply = await network.ask(coordinator, message)
    elapsed = time.monotonic() - t0

    # -- Print result -------------------------------------------------
    print()
    print(f"  {_BOLD}{'=' * 60}{_RESET}")
    print(f"  {_GREEN}{_BOLD}SIMULATION COMPLETE{_RESET}  {_DIM}({elapsed:.1f}s){_RESET}")
    print(f"  {_BOLD}{'=' * 60}{_RESET}")
    print()
    for line in (reply.body or "(no content)").split("\n"):
        print(f"  {line}")
    print()

    # -- Print telemetry ----------------------------------------------
    m = telemetry.metrics
    print(f"  {_DIM}{'─' * 60}{_RESET}")
    print(f"  {_BOLD}Telemetry:{_RESET}")
    print(f"    Total delegations:  {m.total_delegations}")
    print(f"    Total completions:  {m.total_completions}")
    print(f"    Total actors:       {total_actors}")
    if m.by_target:
        print("    By target:")
        for target, count in sorted(m.by_target.items()):
            print(f"      {target}: {count}")
    print()

    await network.hub.close()


if __name__ == "__main__":
    asyncio.run(main())
