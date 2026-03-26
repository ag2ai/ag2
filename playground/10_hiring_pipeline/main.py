#!/usr/bin/env python3
"""Hiring Pipeline — Fanout + Conditional topology demo.

Category 10: Advanced Topology (Fanout + Conditional Routing)

When a candidate applies, a recruiter delegates to three parallel screeners
via Fanout topology — technical, culture-fit, and background check — all
running simultaneously. A Conditional topology then routes candidates based
on their screening results: strong matches fast-track to the hiring manager,
borderline candidates go through additional review, and clear rejects get
a decline.

This is the first example to demonstrate:
- Fanout topology (parallel side-effect plugins)
- Conditional topology (branching based on envelope predicates)
- Nested topologies (Conditional wrapping Pipelines containing Fanout)

Usage:
    python playground/10_hiring_pipeline/main.py
    python playground/10_hiring_pipeline/main.py --scenario 2
    python playground/10_hiring_pipeline/main.py --scenario 3
    python playground/10_hiring_pipeline/main.py --model gemini-3-flash-preview
"""

from __future__ import annotations

import asyncio
import random
import sys
import time
from datetime import datetime

from autogen.beta.config.gemini import GeminiConfig
from autogen.beta.network import (
    Actor,
    BasePlugin,
    Conditional,
    DelegationRejected,
    DelegationRequest,
    DelegationResult,
    Envelope,
    Fanout,
    HubContext,
    LoopDetector,
    Network,
    Pipeline,
    TelemetryPlugin,
    TokenMonitor,
)
from autogen.beta.tools.final import tool

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
_BG_GREEN = "\033[42m"
_BG_YELLOW = "\033[43m"
_BG_RED = "\033[41m"
_BLACK = "\033[30m"


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
# Screening Plugins — the Fanout participants
# =====================================================================


class TechnicalScreener(BasePlugin):
    """Routing plugin that evaluates technical qualifications.

    Runs as part of a Fanout — receives a copy of the envelope,
    logs its assessment, and passes through (Fanout is side-effect only).
    """

    def __init__(self) -> None:
        super().__init__()
        self.last_result: str = "pending"

    async def process(self, envelope: Envelope, ctx: HubContext) -> Envelope | None:
        # Simulate screening delay
        await asyncio.sleep(0.1)
        task_text = str(getattr(envelope.event, "task", ""))
        # Simple heuristic for demo: check for technical keywords
        tech_keywords = [
            "python",
            "engineering",
            "senior",
            "architect",
            "lead",
            "10 years",
            "machine learning",
            "distributed",
            "systems",
            "phd",
            "masters",
        ]
        matches = sum(1 for kw in tech_keywords if kw.lower() in task_text.lower())
        if matches >= 5:
            self.last_result = "strong"
        elif matches >= 1:
            self.last_result = "adequate"
        else:
            self.last_result = "weak"

        color = _GREEN if self.last_result == "strong" else _YELLOW if self.last_result == "adequate" else _RED
        print(
            f"  {_DIM}{_ts()}{_RESET}  "
            f"{_BLUE}{_BOLD}SCREEN{_RESET}  "
            f"Technical: {color}{self.last_result.upper()}{_RESET} "
            f"{_DIM}({matches} keyword matches){_RESET}"
        )
        return envelope


class CultureScreener(BasePlugin):
    """Routing plugin that evaluates culture fit signals."""

    def __init__(self) -> None:
        super().__init__()
        self.last_result: str = "pending"

    async def process(self, envelope: Envelope, ctx: HubContext) -> Envelope | None:
        await asyncio.sleep(0.1)
        task_text = str(getattr(envelope.event, "task", ""))
        culture_keywords = [
            "team",
            "collaborative",
            "mentor",
            "open source",
            "community",
            "leadership",
            "communication",
            "diverse",
            "growth",
            "passion",
        ]
        matches = sum(1 for kw in culture_keywords if kw.lower() in task_text.lower())
        if matches >= 2:
            self.last_result = "strong"
        elif matches >= 1:
            self.last_result = "adequate"
        else:
            self.last_result = "neutral"

        color = _GREEN if self.last_result == "strong" else _YELLOW if self.last_result == "adequate" else _CYAN
        print(
            f"  {_DIM}{_ts()}{_RESET}  "
            f"{_BLUE}{_BOLD}SCREEN{_RESET}  "
            f"Culture:   {color}{self.last_result.upper()}{_RESET} "
            f"{_DIM}({matches} signal matches){_RESET}"
        )
        return envelope


class BackgroundChecker(BasePlugin):
    """Routing plugin that runs background verification."""

    def __init__(self) -> None:
        super().__init__()
        self.last_result: str = "pending"

    async def process(self, envelope: Envelope, ctx: HubContext) -> Envelope | None:
        await asyncio.sleep(0.15)  # Slightly slower — simulates external check
        # Random background check result for demo
        outcome = random.choices(["clear", "clear", "clear", "flag"], weights=[3, 3, 3, 1])[0]
        self.last_result = outcome

        color = _GREEN if outcome == "clear" else _RED
        print(f"  {_DIM}{_ts()}{_RESET}  {_BLUE}{_BOLD}SCREEN{_RESET}  Background: {color}{outcome.upper()}{_RESET}")
        return envelope


# =====================================================================
# Route Logger — logs which path the Conditional chose
# =====================================================================


class FastTrackLogger(BasePlugin):
    """Logs when a candidate is fast-tracked."""

    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    async def process(self, envelope: Envelope, ctx: HubContext) -> Envelope | None:
        self.count += 1
        print(
            f"  {_DIM}{_ts()}{_RESET}  "
            f"{_BG_GREEN}{_BLACK}{_BOLD} FAST-TRACK {_RESET}  "
            f"{_GREEN}Candidate routed to hiring manager (strong match){_RESET}"
        )
        return envelope


class StandardReviewLogger(BasePlugin):
    """Logs when a candidate goes through standard review."""

    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    async def process(self, envelope: Envelope, ctx: HubContext) -> Envelope | None:
        self.count += 1
        print(
            f"  {_DIM}{_ts()}{_RESET}  "
            f"{_BG_YELLOW}{_BLACK}{_BOLD} STANDARD {_RESET}  "
            f"{_YELLOW}Candidate routed through additional review{_RESET}"
        )
        return envelope


# =====================================================================
# Tools — Recruiter
# =====================================================================


@tool
async def create_candidate_profile(
    name: str,
    role_applied: str,
    experience_years: int,
    skills: str,
    education: str,
) -> str:
    """Create a candidate profile from the application.

    Args:
        name: Candidate's full name.
        role_applied: Position they applied for.
        experience_years: Total years of relevant experience.
        skills: Comma-separated list of key skills.
        education: Highest education level and field.
    """
    cid = f"CAN-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000, 9999)}"
    return (
        f"Candidate Profile Created\n"
        f"{'=' * 40}\n"
        f"  ID: {cid}\n"
        f"  Name: {name}\n"
        f"  Position: {role_applied}\n"
        f"  Experience: {experience_years} years\n"
        f"  Skills: {skills}\n"
        f"  Education: {education}\n"
        f"  Status: SCREENING\n"
        f"  Applied: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )


@tool
async def check_role_requirements(role: str) -> str:
    """Check the requirements for an open role.

    Args:
        role: The job title to check requirements for.
    """
    reqs = {
        "senior engineer": {
            "min_years": 5,
            "required": "Python, distributed systems",
            "preferred": "ML, cloud platforms",
            "team_size": 8,
        },
        "engineering manager": {
            "min_years": 8,
            "required": "People management, technical background",
            "preferred": "Scaling teams, cross-functional leadership",
            "team_size": 15,
        },
    }
    r = reqs.get(
        role.lower(),
        {
            "min_years": 3,
            "required": "Relevant domain experience",
            "preferred": "Strong communication",
            "team_size": 6,
        },
    )
    return (
        f"Role Requirements — {role}\n"
        f"{'=' * 40}\n"
        f"  Minimum Experience: {r['min_years']} years\n"
        f"  Required Skills: {r['required']}\n"
        f"  Preferred: {r['preferred']}\n"
        f"  Team Size: {r['team_size']} people\n"
        f"  Open Positions: {random.randint(1, 3)}"
    )


recruiter_tools = [create_candidate_profile, check_role_requirements]


# =====================================================================
# Tools — Hiring Manager
# =====================================================================


@tool
async def evaluate_candidate(
    candidate_name: str,
    strengths: str,
    concerns: str,
    role_fit_score: int,
) -> str:
    """Evaluate a candidate for final hiring decision.

    Args:
        candidate_name: Name of the candidate.
        strengths: Key strengths identified during screening.
        concerns: Any concerns or gaps identified.
        role_fit_score: Score from 1-10 for role fit.
    """
    recommendation = (
        "STRONG HIRE"
        if role_fit_score >= 8
        else "HIRE"
        if role_fit_score >= 6
        else "BORDERLINE"
        if role_fit_score >= 4
        else "NO HIRE"
    )
    return (
        f"Candidate Evaluation — {candidate_name}\n"
        f"{'=' * 40}\n"
        f"  Role Fit Score: {role_fit_score}/10\n"
        f"  Strengths: {strengths}\n"
        f"  Concerns: {concerns}\n"
        f"  Recommendation: {recommendation}\n"
        f"  Compensation Band: {'Senior' if role_fit_score >= 7 else 'Mid-level'}\n"
        f"  Start Date Availability: {'Immediate' if random.random() > 0.5 else '2 weeks notice'}"
    )


@tool
async def make_offer(
    candidate_name: str,
    role: str,
    decision: str,
    notes: str = "",
) -> str:
    """Make the final hiring decision and generate offer or rejection.

    Args:
        candidate_name: Name of the candidate.
        role: Position title.
        decision: Hiring decision (hire/reject/hold).
        notes: Additional notes for the decision.
    """
    decision_upper = decision.upper()
    lines = [
        "HIRING DECISION",
        "=" * 40,
        f"  Candidate: {candidate_name}",
        f"  Position: {role}",
        f"  Decision: {decision_upper}",
        f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    ]
    if notes:
        lines.append(f"  Notes: {notes}")
    if decision_upper == "HIRE":
        salary = f"${random.randint(150, 250) * 1000:,}"
        lines.extend([
            "\n  Offer Details:",
            f"    Base Salary: {salary}",
            f"    Equity: {random.randint(5000, 20000):,} RSUs (4-year vest)",
            f"    Sign-on Bonus: ${random.randint(10, 30) * 1000:,}",
            "    Benefits: Standard package (health, 401k match, PTO)",
            "\n  Next Steps: Send formal offer letter, schedule onboarding",
        ])
    elif decision_upper == "HOLD":
        lines.append("\n  Next Steps: Schedule additional interview round, re-evaluate in 1 week")
    else:
        lines.append("\n  Next Steps: Send personalized rejection with feedback, add to talent pool")
    return "\n".join(lines)


hiring_manager_tools = [evaluate_candidate, make_offer]


# =====================================================================
# Scenarios
# =====================================================================

SCENARIOS = {
    1: (
        "Strong Senior Engineer",
        "New application received: Sarah Chen applying for Senior Engineer. "
        "10 years Python experience, distributed systems expertise at Google "
        "and Stripe, masters in CS from Stanford. Active open source "
        "contributor, led a team of 6, passionate about mentoring junior "
        "engineers. Known for collaborative leadership style.",
    ),
    2: (
        "Junior with Potential",
        "New application received: Alex Rivera applying for Senior Engineer. "
        "3 years experience, mostly frontend JavaScript. Bootcamp graduate, "
        "self-taught Python. No distributed systems background. Enthusiastic "
        "about growth, good communication skills, but lacks the technical "
        "depth for a senior role.",
    ),
    3: (
        "Engineering Manager",
        "New application received: Dr. James Park applying for Engineering Manager. "
        "15 years in tech, PhD in distributed computing, 5 years managing teams "
        "of 20+ engineers at Meta. Led cross-functional platform migration "
        "serving 100M users. Passionate about diverse team building and "
        "engineering culture. Active in tech community leadership.",
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
        title, message = "Custom Application", custom_msg
    else:
        title, message = SCENARIOS[scenario_num]

    # -- Create screening plugins (shared state for Conditional) ------
    tech_screener = TechnicalScreener()
    culture_screener = CultureScreener()
    background_checker = BackgroundChecker()
    fast_track_logger = FastTrackLogger()
    standard_review_logger = StandardReviewLogger()

    # -- Build topology -----------------------------------------------
    # The topology tree:
    #
    #   Pipeline(
    #     Fanout(                       ← parallel screening
    #       TechnicalScreener,
    #       CultureScreener,
    #       BackgroundChecker,
    #     ),
    #     Conditional(                  ← route based on results
    #       predicate=is_strong_match,
    #       if_true=Pipeline(FastTrackLogger),
    #       if_false=Pipeline(StandardReviewLogger),
    #     ),
    #   )

    def is_strong_match(envelope: Envelope) -> bool:
        """Check if the candidate passed all screens strongly."""
        return (
            tech_screener.last_result == "strong"
            and culture_screener.last_result in ("strong", "adequate")
            and background_checker.last_result == "clear"
        )

    topology = Pipeline(
        Fanout(tech_screener, culture_screener, background_checker),
        Conditional(
            predicate=is_strong_match,
            if_true=Pipeline(fast_track_logger),
            if_false=Pipeline(standard_review_logger),
        ),
    )

    # -- Create actors ------------------------------------------------
    recruiter = Actor(
        "recruiter",
        prompt=(
            "You are a Recruiter.\n\n"
            "Your job is to process incoming applications, create candidate "
            "profiles, check role requirements, and hand off to the hiring "
            "manager for evaluation.\n\n"
            "Steps:\n"
            "1. Use check_role_requirements to understand what the role needs\n"
            "2. Use create_candidate_profile with the candidate's details\n"
            "3. Use discover_agents to find a hiring manager\n"
            "4. Use delegate_to to send the complete candidate profile, role "
            "requirements, and your initial assessment to the hiring manager\n"
            "5. Return the final hiring decision\n\n"
            "Include ALL relevant details when delegating — the hiring manager "
            "needs the full picture."
        ),
        config=GeminiConfig(model=model, temperature=0.3),
        tools=recruiter_tools,
        observers=[TokenMonitor(warn_threshold=10_000, alert_threshold=30_000)],
    )

    hiring_manager = Actor(
        "hiring-manager",
        prompt=(
            "You are a Hiring Manager.\n\n"
            "You receive screened candidates from the recruiter. Your job is "
            "to evaluate them and make the final hiring decision.\n\n"
            "Steps:\n"
            "1. Review the candidate profile and screening results\n"
            "2. Use evaluate_candidate with your assessment (role_fit_score "
            "1-10, where 8+ is strong hire, 6-7 hire, 4-5 borderline, "
            "below 4 no hire)\n"
            "3. Use make_offer with your final decision "
            "(hire/reject/hold)\n"
            "4. Return a comprehensive summary of the decision\n\n"
            "Be fair but rigorous. Consider both technical skills and culture fit."
        ),
        config=GeminiConfig(model=model, temperature=0.2),
        tools=hiring_manager_tools,
        observers=[
            TokenMonitor(warn_threshold=10_000, alert_threshold=30_000),
            LoopDetector(repeat_threshold=3),
        ],
    )

    # -- Create network -----------------------------------------------
    telemetry = TelemetryPlugin()

    network = Network(
        topology=topology,
        plugins=[telemetry],
        max_delegation_depth=4,
    )

    await network.register(
        recruiter,
        capabilities=["recruiting", "screening", "intake"],
        description="Processes applications, creates profiles, coordinates screening",
    )
    await network.register(
        hiring_manager,
        capabilities=["hiring", "evaluation", "offers"],
        description="Evaluates candidates, makes hiring decisions, extends offers",
    )

    # -- Subscribe to hub stream for live logging ---------------------
    async def _on_hub_event(event: object) -> None:
        ts = _ts()
        if isinstance(event, DelegationRequest):
            print(
                f"  {_DIM}{ts}{_RESET}  "
                f"{_MAGENTA}{_BOLD}HUB{_RESET}  "
                f"{_MAGENTA}{event.source} -> {event.target}{_RESET}"
            )
            preview = event.task[:120].replace("\n", " ")
            print(f"  {_DIM}{' ' * 12}{_RESET}       {_DIM}{preview}{'...' if len(event.task) > 120 else ''}{_RESET}")
        elif isinstance(event, DelegationResult):
            print(
                f"  {_DIM}{ts}{_RESET}  "
                f"{_GREEN}{_BOLD}HUB{_RESET}  "
                f"{_GREEN}{event.target} completed -> {event.source}{_RESET}"
            )
        elif isinstance(event, DelegationRejected):
            print(
                f"  {_DIM}{ts}{_RESET}  "
                f"{_RED}{_BOLD}HUB REJECTED{_RESET}  "
                f"{_RED}{event.source} -> {event.target}: {event.reason}{_RESET}"
            )

    network.hub.stream.subscribe(_on_hub_event)

    # -- Print header -------------------------------------------------
    print()
    print(f"  {_BOLD}{'=' * 60}{_RESET}")
    print(f"  {_BOLD}HIRING PIPELINE{_RESET}  {_DIM}(10_hiring_pipeline){_RESET}")
    print(f"  {_BOLD}{'=' * 60}{_RESET}")
    print()
    print(f"  {_CYAN}Category:{_RESET}  Advanced Topology (Fanout + Conditional)")
    print(f"  {_CYAN}Scenario:{_RESET}  {title}")
    print(f"  {_CYAN}Model:{_RESET}     {model}")
    print(f"  {_CYAN}Agents:{_RESET}    recruiter -> hiring-manager")
    print(f"  {_CYAN}Topology:{_RESET}  Pipeline(")
    print(f"  {_DIM}             Fanout(TechScreen, CultureScreen, BackgroundCheck),{_RESET}")
    print(f"  {_DIM}             Conditional(strong? -> FastTrack | StandardReview){_RESET}")
    print(f"  {_DIM}           ){_RESET}")
    print()
    print(f"  {_BOLD}Application:{_RESET}")
    print(_wrap(message, indent=4, width=72))
    print()
    print(f"  {_DIM}{'─' * 60}{_RESET}")
    print(
        f"  {_BOLD}Log legend:{_RESET}  "
        f"{_BLUE}SCREEN{_RESET}=parallel screening  "
        f"{_MAGENTA}HUB{_RESET}=delegation  "
        f"{_GREEN}HUB{_RESET}=completed"
    )
    print(f"  {_DIM}{'─' * 60}{_RESET}")
    print()

    # -- Run ----------------------------------------------------------
    t0 = time.monotonic()
    reply = await network.ask(recruiter, message)
    elapsed = time.monotonic() - t0

    # -- Print result -------------------------------------------------
    print()
    print(f"  {_BOLD}{'=' * 60}{_RESET}")
    print(f"  {_GREEN}{_BOLD}PIPELINE COMPLETE{_RESET}  {_DIM}({elapsed:.1f}s){_RESET}")
    print(f"  {_BOLD}{'=' * 60}{_RESET}")
    print()
    for line in (reply.body or "(no content)").split("\n"):
        print(f"  {line}")
    print()

    # -- Print screening summary --------------------------------------
    print(f"  {_DIM}{'─' * 60}{_RESET}")
    print(f"  {_BOLD}Screening Summary:{_RESET}")
    print(f"    Technical:   {tech_screener.last_result.upper()}")
    print(f"    Culture:     {culture_screener.last_result.upper()}")
    print(f"    Background:  {background_checker.last_result.upper()}")
    print(f"    Route:       {'Fast-track' if fast_track_logger.count > 0 else 'Standard review'}")
    print()

    # -- Print telemetry ----------------------------------------------
    m = telemetry.metrics
    print(f"  {_BOLD}Telemetry:{_RESET}")
    print(f"    Total delegations:  {m.total_delegations}")
    print(f"    Total completions:  {m.total_completions}")
    if m.by_target:
        print(f"    By target:  {dict(m.by_target)}")
    print()

    await network.hub.close()


if __name__ == "__main__":
    asyncio.run(main())
