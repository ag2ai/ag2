"""AI-Powered GTM Sales Pipeline — multi-agent sales automation.

Category 9: GTM / Sales Pipeline (Design Partner Demo)

Five agents coordinate the full sales cycle: SDR prospecting, AE deal
management, SE technical validation, Marketing lead intelligence, and CS
customer expansion. A PipelineGuard plugin enforces valid routing between
agents, and Scheduler drives autonomous pipeline reviews.

Usage:
    python playground/09_gtm_sales/main.py                    # new territory
    python playground/09_gtm_sales/main.py --scenario 2       # deal acceleration
    python playground/09_gtm_sales/main.py --scenario 3       # autonomous pipeline
    python playground/09_gtm_sales/main.py --scenario 4       # customer expansion
    python playground/09_gtm_sales/main.py --model gemini-3-flash-preview
"""

from __future__ import annotations

import argparse
import asyncio
import random
import time
from datetime import datetime, timedelta

from autogen.beta.annotations import Context
from autogen.beta.config.gemini import GeminiConfig
from autogen.beta.network import (
    Actor,
    BasePlugin,
    DelegationRejected,
    DelegationRequest,
    DelegationResult,
    Envelope,
    HubContext,
    IntervalWatch,
    LoopDetector,
    Network,
    Pipeline,
    SchedulerTriggerFired,
    Signal,
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


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


# =====================================================================
# Simulated CRM Data
# =====================================================================

CRM_COMPANIES = {
    "CloudForge Technologies": {
        "industry": "Technology",
        "employees": 850,
        "revenue": "$120M",
        "region": "West Coast",
        "tech_stack": ["AWS", "Kubernetes", "Python", "React"],
        "funding": "Series C, $45M (2025)",
        "news": "Expanding AI platform team, hiring 50 engineers",
        "contacts": [
            {"name": "Sarah Chen", "title": "VP of Engineering", "email": "schen@cloudforge.io"},
            {"name": "Marcus Webb", "title": "CTO", "email": "mwebb@cloudforge.io"},
        ],
    },
    "Meridian Health Systems": {
        "industry": "Healthcare",
        "employees": 2200,
        "revenue": "$340M",
        "region": "West Coast",
        "tech_stack": ["Azure", "Java", ".NET", "Epic EHR"],
        "funding": "Public (MHSC), market cap $1.2B",
        "news": "Digital transformation initiative announced Q1 2026",
        "contacts": [
            {"name": "Dr. Lisa Patel", "title": "Chief Digital Officer", "email": "lpatel@meridianhealth.com"},
            {"name": "James Rodriguez", "title": "Director of IT", "email": "jrodriguez@meridianhealth.com"},
        ],
    },
    "NovaPay Financial": {
        "industry": "Fintech",
        "employees": 430,
        "revenue": "$68M",
        "region": "East Coast",
        "tech_stack": ["GCP", "Go", "PostgreSQL", "React"],
        "funding": "Series B, $30M (2025)",
        "news": "Launching enterprise payment processing Q2 2026",
        "contacts": [
            {"name": "Priya Sharma", "title": "Head of Product", "email": "psharma@novapay.com"},
            {"name": "David Kim", "title": "VP of Engineering", "email": "dkim@novapay.com"},
        ],
    },
    "RetailEdge Inc.": {
        "industry": "Retail Tech",
        "employees": 1500,
        "revenue": "$210M",
        "region": "Midwest",
        "tech_stack": ["AWS", "Java", "Snowflake", "Tableau"],
        "funding": "Private equity backed, $200M valuation",
        "news": "Acquired inventory analytics startup, integrating AI capabilities",
        "contacts": [
            {"name": "Tom Baker", "title": "SVP of Technology", "email": "tbaker@retailedge.com"},
            {"name": "Michelle Torres", "title": "Director of Innovation", "email": "mtorres@retailedge.com"},
        ],
    },
    "Apex Manufacturing": {
        "industry": "Manufacturing",
        "employees": 3800,
        "revenue": "$580M",
        "region": "Southeast",
        "tech_stack": ["On-prem", "SAP", "Siemens MES", "SCADA"],
        "funding": "Public (APXM), market cap $2.1B",
        "news": "Industry 4.0 initiative, modernizing factory floor with AI/ML",
        "contacts": [
            {"name": "Robert Hayes", "title": "VP of Digital Manufacturing", "email": "rhayes@apexmfg.com"},
            {"name": "Angela Wu", "title": "Chief Technology Officer", "email": "awu@apexmfg.com"},
        ],
    },
    "Synapse Analytics": {
        "industry": "Technology",
        "employees": 320,
        "revenue": "$42M",
        "region": "West Coast",
        "tech_stack": ["GCP", "Python", "BigQuery", "dbt"],
        "funding": "Series A, $18M (2024)",
        "news": "Growing 80% YoY, expanding into enterprise data platforms",
        "contacts": [
            {"name": "Alex Rivera", "title": "CEO", "email": "arivera@synapseanalytics.io"},
            {"name": "Nina Chang", "title": "VP of Engineering", "email": "nchang@synapseanalytics.io"},
        ],
    },
    "Guardian Insurance Group": {
        "industry": "Insurance",
        "employees": 5200,
        "revenue": "$1.8B",
        "region": "Northeast",
        "tech_stack": ["Azure", ".NET", "SQL Server", "Guidewire"],
        "funding": "Public (GDIG), market cap $4.5B",
        "news": "Claims automation project greenlit, $20M budget allocated",
        "contacts": [
            {"name": "Patricia Morrison", "title": "CIO", "email": "pmorrison@guardianins.com"},
            {"name": "Kevin Zhang", "title": "Director of AI/ML", "email": "kzhang@guardianins.com"},
        ],
    },
    "Velocity Logistics": {
        "industry": "Logistics",
        "employees": 1100,
        "revenue": "$175M",
        "region": "West Coast",
        "tech_stack": ["AWS", "Python", "MongoDB", "Kafka"],
        "funding": "Series D, $80M (2025)",
        "news": "Building autonomous routing optimization platform",
        "contacts": [
            {"name": "Carlos Mendez", "title": "VP of Product", "email": "cmendez@velocitylog.com"},
            {"name": "Jennifer Park", "title": "Head of Engineering", "email": "jpark@velocitylog.com"},
        ],
    },
}

# Simulated existing pipeline
CRM_PIPELINE = [
    {
        "id": "OPP-2026-0147",
        "company": "CloudForge Technologies",
        "contact": "Sarah Chen",
        "deal_size": 120_000,
        "stage": "Demo",
        "next_step": "Technical deep-dive scheduled",
        "days_in_stage": 8,
        "close_date": "2026-05-15",
    },
    {
        "id": "OPP-2026-0152",
        "company": "Meridian Health Systems",
        "contact": "Dr. Lisa Patel",
        "deal_size": 250_000,
        "stage": "Proposal",
        "next_step": "Awaiting procurement review",
        "days_in_stage": 12,
        "close_date": "2026-06-01",
    },
    {
        "id": "OPP-2026-0158",
        "company": "NovaPay Financial",
        "contact": "Priya Sharma",
        "deal_size": 180_000,
        "stage": "Negotiation",
        "next_step": "Legal redline in progress",
        "days_in_stage": 5,
        "close_date": "2026-04-30",
    },
    {
        "id": "OPP-2026-0161",
        "company": "RetailEdge Inc.",
        "contact": "Tom Baker",
        "deal_size": 95_000,
        "stage": "Discovery",
        "next_step": "Initial requirements call",
        "days_in_stage": 3,
        "close_date": "2026-07-15",
    },
]

# Simulated existing customers
CRM_CUSTOMERS = [
    {
        "company": "Synapse Analytics",
        "contract_value": 85_000,
        "renewal_date": "2026-09-01",
        "health_score": 92,
        "nps": 9,
        "adoption_rate": 87,
        "support_tickets_30d": 2,
        "expansion_signals": ["Added 3 new teams", "API usage up 140%"],
    },
    {
        "company": "Guardian Insurance Group",
        "contract_value": 320_000,
        "renewal_date": "2026-12-01",
        "health_score": 78,
        "nps": 7,
        "adoption_rate": 65,
        "support_tickets_30d": 8,
        "expansion_signals": ["Claims team interested in advanced module"],
    },
    {
        "company": "Velocity Logistics",
        "contract_value": 145_000,
        "renewal_date": "2026-08-15",
        "health_score": 88,
        "nps": 8,
        "adoption_rate": 82,
        "support_tickets_30d": 3,
        "expansion_signals": ["Expanding to EU operations", "Requested multi-region support"],
    },
]


# =====================================================================
# PipelineGuard — routing plugin (soft enforcement)
# =====================================================================


class PipelineGuard(BasePlugin):
    """Validates sales pipeline routing rules.

    Soft enforcement: logs warnings for unexpected routes but allows them
    through. This lets the framework guide agents without rigidly blocking
    creative problem-solving.

    Allowed routes:
        SDR -> AE, Marketing
        AE  -> SE, CS, SDR
        SE  -> AE
        Marketing -> SDR
        CS  -> AE
    """

    ALLOWED_ROUTES: dict[str, list[str]] = {
        "sdr": ["ae", "marketing"],
        "ae": ["se", "cs", "sdr"],
        "se": ["ae"],
        "marketing": ["sdr"],
        "cs": ["ae"],
    }

    def __init__(self) -> None:
        super().__init__()
        self.total_routed = 0
        self.total_warnings = 0
        self.route_log: list[tuple[str, str, bool]] = []  # (source, target, allowed)

    async def process(self, envelope: Envelope, ctx: HubContext) -> Envelope | None:
        source = envelope.sender or ""
        target = envelope.recipient or ""
        allowed_targets = self.ALLOWED_ROUTES.get(source, [])
        is_allowed = target in allowed_targets or source == "scheduler"

        self.total_routed += 1
        self.route_log.append((source, target, is_allowed))

        if not is_allowed and source:
            self.total_warnings += 1
            print(
                f"  {_DIM}{_ts()}{_RESET}  "
                f"{_YELLOW}{_BOLD}PIPELINE GUARD{_RESET} "
                f"{_YELLOW}Unexpected route: {source} -> {target} "
                f"(allowed: {', '.join(allowed_targets) or 'none'}){_RESET}"
            )

        return envelope  # Soft enforcement — always allow


# =====================================================================
# DealTracker — system plugin (observes hub stream)
# =====================================================================


class DealTracker(BasePlugin):
    """Tracks deal-related delegations for pipeline visibility."""

    def __init__(self) -> None:
        super().__init__()
        self.delegation_log: list[dict] = []
        self._hub = None
        self._sub_ids: list = []

    def install(self, hub) -> None:  # type: ignore[override]
        from autogen.beta.events.conditions import TypeCondition

        self._hub = hub
        self._sub_ids.append(
            hub.stream.subscribe(
                self._on_request,
                condition=TypeCondition(DelegationRequest),
            )
        )
        self._sub_ids.append(
            hub.stream.subscribe(
                self._on_result,
                condition=TypeCondition(DelegationResult),
            )
        )

    def uninstall(self) -> None:
        if self._hub:
            for sub_id in self._sub_ids:
                self._hub.stream.unsubscribe(sub_id)
        self._sub_ids.clear()

    async def _on_request(self, event: DelegationRequest, ctx: Context) -> None:  # type: ignore[override]
        self.delegation_log.append(
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "type": "request",
                "source": event.source,
                "target": event.target,
                "task_preview": event.task[:120],
            }
        )

    async def _on_result(self, event: DelegationResult, ctx: Context) -> None:  # type: ignore[override]
        self.delegation_log.append(
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "type": "result",
                "source": event.source,
                "target": event.target,
                "result_preview": event.result[:120],
            }
        )


# =====================================================================
# Tools — SDR (Sales Development Rep)
# =====================================================================


@tool
async def search_leads(industry: str = "", company_size: str = "", region: str = "") -> str:
    """Search CRM for leads matching criteria.

    Args:
        industry: Industry filter (e.g. "Healthcare", "Technology").
        company_size: Size filter (e.g. "500+", "1000+").
        region: Region filter (e.g. "West Coast", "East Coast").
    """
    results = []
    min_employees = 0
    if company_size:
        min_employees = int(company_size.replace("+", "").replace(",", "").strip())

    for name, data in CRM_COMPANIES.items():
        if industry and industry.lower() not in data["industry"].lower():
            continue
        if min_employees and data["employees"] < min_employees:
            continue
        if region and region.lower() not in data["region"].lower():
            continue
        contact = data["contacts"][0]
        score = random.randint(55, 95)
        results.append(
            f"  {name} | {data['industry']} | {data['employees']:,} employees | {data['revenue']} revenue\n"
            f"    Contact: {contact['name']}, {contact['title']} ({contact['email']})\n"
            f"    Region: {data['region']} | Lead Score: {score}/100\n"
            f"    Tech Stack: {', '.join(data['tech_stack'])}"
        )

    if not results:
        return "No leads found matching criteria. Try broadening your search filters."
    return f"Found {len(results)} leads:\n\n" + "\n\n".join(results)


@tool
async def enrich_lead(company_name: str) -> str:
    """Enrich lead data with technographics, funding, and recent news.

    Args:
        company_name: Company name to enrich.
    """
    data = CRM_COMPANIES.get(company_name)
    if not data:
        # Fuzzy match
        for name, d in CRM_COMPANIES.items():
            if company_name.lower() in name.lower():
                data = d
                company_name = name
                break
    if not data:
        return f"Company '{company_name}' not found in CRM. Check spelling."

    return (
        f"=== {company_name} — Enriched Profile ===\n"
        f"Industry: {data['industry']}\n"
        f"Employees: {data['employees']:,} | Revenue: {data['revenue']}\n"
        f"Region: {data['region']}\n"
        f"Tech Stack: {', '.join(data['tech_stack'])}\n"
        f"Funding: {data['funding']}\n"
        f"Recent News: {data['news']}\n"
        f"Contacts:\n" + "\n".join(f"  - {c['name']}, {c['title']} ({c['email']})" for c in data["contacts"])
    )


@tool
async def qualify_lead(
    company_name: str,
    budget: str = "unknown",
    authority: str = "unknown",
    need: str = "unknown",
    timeline: str = "unknown",
) -> str:
    """Run BANT qualification on a lead.

    Args:
        company_name: Company to qualify.
        budget: Budget information (e.g. "$100K+", "allocated", "unknown").
        authority: Decision-maker access level.
        need: Business need / pain point.
        timeline: Purchase timeline.
    """
    scores = {
        "budget": random.randint(5, 10) if budget != "unknown" else random.randint(2, 5),
        "authority": random.randint(5, 10) if authority != "unknown" else random.randint(2, 5),
        "need": random.randint(6, 10) if need != "unknown" else random.randint(3, 6),
        "timeline": random.randint(5, 10) if timeline != "unknown" else random.randint(2, 5),
    }
    total = sum(scores.values())
    rating = "HOT" if total >= 30 else "WARM" if total >= 22 else "COLD"

    return (
        f"=== BANT Qualification: {company_name} ===\n"
        f"Budget:    {scores['budget']}/10 — {budget}\n"
        f"Authority: {scores['authority']}/10 — {authority}\n"
        f"Need:      {scores['need']}/10 — {need}\n"
        f"Timeline:  {scores['timeline']}/10 — {timeline}\n"
        f"{'─' * 40}\n"
        f"Total Score: {total}/40 | Rating: {rating}\n"
        f"Recommendation: {'Delegate to AE immediately' if rating == 'HOT' else 'Continue nurturing' if rating == 'WARM' else 'Deprioritize or nurture long-term'}"
    )


@tool
async def send_outreach(
    contact_name: str,
    email: str,
    message_type: str = "cold_email",
    personalization_notes: str = "",
) -> str:
    """Send personalized outreach to a prospect.

    Args:
        contact_name: Recipient name.
        email: Recipient email.
        message_type: Type of outreach (cold_email/follow_up/linkedin).
        personalization_notes: Notes for personalizing the message.
    """
    msg_id = f"MSG-{random.randint(10000, 99999)}"
    open_rate = random.randint(15, 45)
    return (
        f"Outreach sent successfully.\n"
        f"  ID: {msg_id}\n"
        f"  To: {contact_name} <{email}>\n"
        f"  Type: {message_type}\n"
        f"  Personalization: {personalization_notes[:100] or '(standard template)'}\n"
        f"  Predicted open rate: {open_rate}%\n"
        f"  Follow-up scheduled: {(datetime.now() + timedelta(days=3)).strftime('%Y-%m-%d')}"
    )


# =====================================================================
# Tools — AE (Account Executive)
# =====================================================================


@tool
async def create_opportunity(company_name: str, contact_name: str, deal_size: int, stage: str = "Discovery") -> str:
    """Create a new CRM opportunity.

    Args:
        company_name: Company name.
        contact_name: Primary contact.
        deal_size: Estimated deal size in USD.
        stage: Pipeline stage (Discovery/Demo/Proposal/Negotiation/Closed-Won).
    """
    opp_id = f"OPP-2026-{random.randint(1000, 9999)}"
    close_date = (datetime.now() + timedelta(days=random.randint(45, 120))).strftime("%Y-%m-%d")
    return (
        f"Opportunity created.\n"
        f"  ID: {opp_id}\n"
        f"  Account: {company_name}\n"
        f"  Contact: {contact_name}\n"
        f"  Amount: ${deal_size:,}\n"
        f"  Stage: {stage}\n"
        f"  Close Date: {close_date}\n"
        f"  Pipeline: Enterprise"
    )


@tool
async def schedule_demo(company_name: str, contact_name: str, demo_type: str = "standard") -> str:
    """Schedule a product demo.

    Args:
        company_name: Company name.
        contact_name: Attendee.
        demo_type: Demo type (standard/technical/executive).
    """
    demo_date = (datetime.now() + timedelta(days=random.randint(2, 7))).strftime("%Y-%m-%d %H:%M")
    return (
        f"Demo scheduled.\n"
        f"  Company: {company_name}\n"
        f"  Attendee: {contact_name}\n"
        f"  Type: {demo_type}\n"
        f"  Date: {demo_date}\n"
        f"  Duration: {'60 min' if demo_type == 'executive' else '45 min'}\n"
        f"  Prep: {'Executive deck + ROI analysis' if demo_type == 'executive' else 'Standard demo environment + use case walkthrough'}\n"
        f"  Calendar link: https://cal.app/demo-{random.randint(1000, 9999)}"
    )


@tool
async def prepare_proposal(company_name: str, use_case: str, deal_size: int, contract_term: int = 12) -> str:
    """Generate a sales proposal.

    Args:
        company_name: Company name.
        use_case: Primary use case.
        deal_size: Proposed deal size in USD.
        contract_term: Contract length in months.
    """
    discount = random.choice([0, 5, 10, 15])
    final_price = deal_size * (100 - discount) / 100
    roi_months = random.randint(4, 10)
    return (
        f"=== Proposal: {company_name} ===\n"
        f"Use Case: {use_case}\n"
        f"Contract Term: {contract_term} months\n"
        f"{'─' * 40}\n"
        f"List Price:     ${deal_size:,.0f}/yr\n"
        f"Discount:       {discount}%\n"
        f"Final Price:    ${final_price:,.0f}/yr\n"
        f"{'─' * 40}\n"
        f"ROI Projection: {roi_months}-month payback\n"
        f"Estimated savings: ${deal_size * 2.5:,.0f} over {contract_term} months\n"
        f"Implementation: 4-6 weeks\n"
        f"Support: Dedicated CSM + 24/7 technical support\n"
        f"Proposal ID: PROP-{random.randint(1000, 9999)}"
    )


@tool
async def update_deal_stage(deal_id: str, new_stage: str, notes: str = "") -> str:
    """Update pipeline stage for an opportunity.

    Args:
        deal_id: Opportunity ID.
        new_stage: New stage (Discovery/Demo/Proposal/Negotiation/Closed-Won/Closed-Lost).
        notes: Stage change notes.
    """
    return (
        f"Deal updated.\n"
        f"  ID: {deal_id}\n"
        f"  New Stage: {new_stage}\n"
        f"  Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"  Notes: {notes or '(none)'}\n"
        f"  Next review: {(datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')}"
    )


# =====================================================================
# Tools — SE (Solutions Engineer)
# =====================================================================


@tool
async def assess_technical_requirements(company_name: str, use_case: str, current_stack: str = "") -> str:
    """Evaluate technical fit and integration requirements.

    Args:
        company_name: Company name.
        use_case: Proposed use case.
        current_stack: Customer's current technology stack.
    """
    fit_score = random.randint(70, 98)
    risks = random.choice(
        [
            ["Legacy system integration may require custom connector"],
            ["Data migration complexity — 6+ data sources"],
            ["Compliance requirements need security review"],
            ["No significant technical risks identified"],
        ]
    )
    return (
        f"=== Technical Assessment: {company_name} ===\n"
        f"Use Case: {use_case}\n"
        f"Current Stack: {current_stack or 'Not specified'}\n"
        f"{'─' * 40}\n"
        f"Technical Fit Score: {fit_score}/100\n"
        f"Integration Points: API, SSO, data warehouse connector\n"
        f"Estimated Effort: {random.randint(2, 8)} weeks implementation\n"
        f"Risks:\n"
        + "\n".join(f"  - {r}" for r in risks)
        + f"\nRecommendation: {'Proceed with POC' if fit_score >= 80 else 'Address risks before proceeding'}"
    )


@tool
async def design_solution(company_name: str, requirements: str, constraints: str = "") -> str:
    """Create solution architecture for a prospect.

    Args:
        company_name: Company name.
        requirements: Key requirements.
        constraints: Technical or business constraints.
    """
    return (
        f"=== Solution Design: {company_name} ===\n"
        f"Requirements: {requirements}\n"
        f"Constraints: {constraints or 'None specified'}\n"
        f"{'─' * 40}\n"
        f"Architecture:\n"
        f"  1. Core platform deployment (cloud-native)\n"
        f"  2. API gateway for existing system integration\n"
        f"  3. Data pipeline: ETL from customer sources\n"
        f"  4. Custom workflow engine for {company_name}'s processes\n"
        f"  5. Dashboard & reporting layer\n"
        f"{'─' * 40}\n"
        f"Timeline: {random.randint(4, 8)} weeks to production\n"
        f"Team: 1 SA + 2 engineers + 1 project manager\n"
        f"Architecture doc: ARCH-{random.randint(1000, 9999)}"
    )


@tool
async def create_poc_plan(company_name: str, scope: str, success_criteria: str = "") -> str:
    """Design a proof-of-concept plan.

    Args:
        company_name: Company name.
        scope: POC scope description.
        success_criteria: What defines POC success.
    """
    return (
        f"=== POC Plan: {company_name} ===\n"
        f"Scope: {scope}\n"
        f"Success Criteria: {success_criteria or 'TBD with customer'}\n"
        f"{'─' * 40}\n"
        f"Duration: {random.randint(2, 4)} weeks\n"
        f"Milestones:\n"
        f"  Week 1: Environment setup + data onboarding\n"
        f"  Week 2: Core workflow implementation\n"
        f"  Week 3: Integration testing + user acceptance\n"
        f"  Week 4: Results review + go/no-go decision\n"
        f"Resources: Dedicated SE + shared dev environment\n"
        f"POC ID: POC-{random.randint(1000, 9999)}"
    )


@tool
async def review_security_compliance(company_name: str, requirements: str = "") -> str:
    """Check security and compliance readiness.

    Args:
        company_name: Company name.
        requirements: Specific compliance requirements (SOC2, HIPAA, GDPR, etc.).
    """
    checks = [
        ("SOC 2 Type II", "CERTIFIED"),
        ("GDPR", "COMPLIANT"),
        ("HIPAA", random.choice(["COMPLIANT", "COMPLIANT", "REVIEW NEEDED"])),
        ("Data encryption (at rest + transit)", "PASS"),
        ("SSO / SAML 2.0", "SUPPORTED"),
        ("Penetration test (last 90 days)", "PASS"),
    ]
    return (
        f"=== Security & Compliance: {company_name} ===\n"
        f"Requirements: {requirements or 'Standard enterprise'}\n"
        f"{'─' * 40}\n" + "\n".join(f"  {name}: {status}" for name, status in checks) + f"\n{'─' * 40}\n"
        f"Overall: {'Ready for enterprise deployment' if all(s != 'REVIEW NEEDED' for _, s in checks) else 'Minor items need review — see HIPAA status'}"
    )


# =====================================================================
# Tools — Marketing
# =====================================================================


@tool
async def analyze_campaign_performance(campaign_name: str = "all") -> str:
    """Analyze marketing campaign metrics.

    Args:
        campaign_name: Campaign name or "all" for overview.
    """
    campaigns = [
        ("AI Automation Webinar", 12_500, 1_890, 47, 34.20, "$156K"),
        ("Enterprise Platform Guide", 8_200, 1_230, 31, 42.10, "$89K"),
        ("Customer Success Stories", 15_800, 2_100, 58, 28.50, "$210K"),
    ]
    lines = ["=== Campaign Performance ===\n"]
    for name, impressions, clicks, conversions, cpl, pipeline in campaigns:
        lines.append(
            f"  {name}\n"
            f"    Impressions: {impressions:,} | Clicks: {clicks:,} | "
            f"Conversions: {conversions}\n"
            f"    CPL: ${cpl:.2f} | Pipeline generated: {pipeline}"
        )
    lines.append("\n  Total pipeline: $455K | Avg CPL: $34.93")
    return "\n".join(lines)


@tool
async def generate_content(content_type: str, topic: str, target_persona: str = "VP of Engineering") -> str:
    """Generate content brief for sales enablement.

    Args:
        content_type: Type (blog/whitepaper/case_study/webinar).
        topic: Content topic.
        target_persona: Target audience persona.
    """
    return (
        f"=== Content Brief ===\n"
        f"Type: {content_type}\n"
        f"Topic: {topic}\n"
        f"Target Persona: {target_persona}\n"
        f"{'─' * 40}\n"
        f"Headline: How {topic} Drives 3x ROI for Enterprise Teams\n"
        f"Key Messages:\n"
        f"  1. Industry challenge and pain points\n"
        f"  2. Solution approach with real metrics\n"
        f"  3. Customer proof points and outcomes\n"
        f"  4. Clear call-to-action for next step\n"
        f"SEO Keywords: {topic.lower()}, enterprise automation, ROI\n"
        f"Distribution: LinkedIn, email nurture, website resource center\n"
        f"Content ID: CNT-{random.randint(1000, 9999)}"
    )


@tool
async def score_lead_engagement(company_name: str) -> str:
    """Score a lead's engagement with marketing content.

    Args:
        company_name: Company to score.
    """
    activities = random.sample(
        [
            "Downloaded whitepaper (2 days ago)",
            "Attended webinar (1 week ago)",
            "Visited pricing page (3 times this month)",
            "Opened 4/5 recent emails",
            "Clicked demo CTA on blog post",
            "Viewed case study for similar industry",
            "Returned to site 6 times in 30 days",
        ],
        k=random.randint(2, 5),
    )
    score = random.randint(40, 95)
    mql = score >= 65
    return (
        f"=== Engagement Score: {company_name} ===\n"
        f"Score: {score}/100 {'(MQL)' if mql else '(Not yet MQL)'}\n"
        f"{'─' * 40}\n"
        f"Recent Activity:\n" + "\n".join(f"  - {a}" for a in activities) + f"\n{'─' * 40}\n"
        f"Recommendation: {'Ready for sales outreach — high intent signals' if mql else 'Continue nurturing with targeted content'}"
    )


@tool
async def create_abm_campaign(company_name: str, contacts: str = "", campaign_type: str = "awareness") -> str:
    """Create an account-based marketing campaign.

    Args:
        company_name: Target account.
        contacts: Key contacts to target.
        campaign_type: Campaign type (awareness/engagement/conversion).
    """
    return (
        f"=== ABM Campaign: {company_name} ===\n"
        f"Type: {campaign_type}\n"
        f"Contacts: {contacts or 'All identified stakeholders'}\n"
        f"{'─' * 40}\n"
        f"Touchpoints (4-week cadence):\n"
        f"  Week 1: Personalized LinkedIn ads + industry report\n"
        f"  Week 2: Custom email sequence (3 emails)\n"
        f"  Week 3: Targeted webinar invitation\n"
        f"  Week 4: Direct mail + executive brief\n"
        f"Budget: ${random.randint(2, 8) * 500:,}\n"
        f"Expected pipeline: ${random.randint(50, 200) * 1000:,}\n"
        f"Campaign ID: ABM-{random.randint(1000, 9999)}"
    )


# =====================================================================
# Tools — CS (Customer Success)
# =====================================================================


@tool
async def create_onboarding_plan(company_name: str, deal_size: int = 0, product_modules: str = "") -> str:
    """Create a 90-day customer onboarding plan.

    Args:
        company_name: New customer.
        deal_size: Contract value.
        product_modules: Modules purchased.
    """
    return (
        f"=== 90-Day Onboarding Plan: {company_name} ===\n"
        f"Contract: ${deal_size:,} | Modules: {product_modules or 'Core platform'}\n"
        f"{'─' * 40}\n"
        f"Days 1-30: Foundation\n"
        f"  - Kickoff call + stakeholder alignment\n"
        f"  - Environment provisioning + SSO setup\n"
        f"  - Data migration (batch 1)\n"
        f"  - Admin training (2 sessions)\n"
        f"Days 31-60: Activation\n"
        f"  - Core workflow configuration\n"
        f"  - User training rollout (4 sessions)\n"
        f"  - Integration testing + go-live\n"
        f"  - First value milestone: {random.choice(['50+ users active', '1st automated workflow live', '1st report generated'])}\n"
        f"Days 61-90: Optimization\n"
        f"  - Advanced feature enablement\n"
        f"  - Performance review + optimization\n"
        f"  - Executive business review (EBR)\n"
        f"  - Success metrics: adoption >70%, NPS >7\n"
        f"CSM Assigned: {random.choice(['Emily Watson', 'Ryan Torres', 'Aisha Johnson'])}\n"
        f"Plan ID: ONB-{random.randint(1000, 9999)}"
    )


@tool
async def check_health_score(company_name: str) -> str:
    """Check customer health score and key metrics.

    Args:
        company_name: Customer name.
    """
    customer = None
    for c in CRM_CUSTOMERS:
        if company_name.lower() in c["company"].lower():
            customer = c
            break

    if not customer:
        # Generate plausible data for unknown customer
        customer = {
            "company": company_name,
            "contract_value": random.randint(50, 300) * 1000,
            "renewal_date": (datetime.now() + timedelta(days=random.randint(60, 300))).strftime("%Y-%m-%d"),
            "health_score": random.randint(60, 95),
            "nps": random.randint(6, 10),
            "adoption_rate": random.randint(50, 95),
            "support_tickets_30d": random.randint(0, 12),
            "expansion_signals": random.sample(
                ["New department interested", "Usage up 40%", "Asked about advanced features", "Requested API access"],
                k=random.randint(0, 2),
            ),
        }

    health = customer["health_score"]
    status = "HEALTHY" if health >= 80 else "AT RISK" if health >= 60 else "CRITICAL"
    return (
        f"=== Customer Health: {customer['company']} ===\n"
        f"Health Score: {health}/100 ({status})\n"
        f"{'─' * 40}\n"
        f"Contract Value: ${customer['contract_value']:,}\n"
        f"Renewal Date: {customer['renewal_date']}\n"
        f"NPS: {customer['nps']}/10\n"
        f"Adoption Rate: {customer['adoption_rate']}%\n"
        f"Support Tickets (30d): {customer['support_tickets_30d']}\n"
        f"Expansion Signals:\n" + ("\n".join(f"  - {s}" for s in customer["expansion_signals"]) or "  (none detected)")
    )


@tool
async def identify_expansion_opportunity(company_name: str) -> str:
    """Identify upsell/cross-sell opportunities for a customer.

    Args:
        company_name: Customer name.
    """
    opportunities = random.sample(
        [
            {
                "type": "Upsell",
                "module": "Advanced Analytics",
                "value": random.randint(30, 80) * 1000,
                "signal": "Usage data shows heavy reporting needs",
            },
            {
                "type": "Cross-sell",
                "module": "Integration Hub",
                "value": random.randint(20, 50) * 1000,
                "signal": "Customer connecting 5+ external tools via workarounds",
            },
            {
                "type": "Expansion",
                "module": "Additional seats (50)",
                "value": random.randint(15, 40) * 1000,
                "signal": "New department onboarding request",
            },
            {
                "type": "Upsell",
                "module": "Enterprise Security",
                "value": random.randint(25, 60) * 1000,
                "signal": "Compliance audit triggered advanced security discussion",
            },
        ],
        k=random.randint(1, 3),
    )

    total = sum(int(o["value"]) for o in opportunities)
    lines = [f"=== Expansion Opportunities: {company_name} ===\n"]
    for o in opportunities:
        lines.append(f"  [{o['type']}] {o['module']} — ${o['value']:,}/yr\n" f"    Signal: {o['signal']}")
    lines.append(f"\n{'─' * 40}")
    lines.append(f"Total expansion potential: ${total:,}/yr")
    lines.append(f"Recommendation: Create opportunities and assign to AE for {company_name}")
    return "\n".join(lines)


@tool
async def schedule_qbr(company_name: str, quarter: str = "Q2 2026") -> str:
    """Schedule a quarterly business review.

    Args:
        company_name: Customer name.
        quarter: Quarter for the review.
    """
    qbr_date = (datetime.now() + timedelta(days=random.randint(10, 30))).strftime("%Y-%m-%d")
    return (
        f"QBR Scheduled.\n"
        f"  Customer: {company_name}\n"
        f"  Quarter: {quarter}\n"
        f"  Date: {qbr_date}\n"
        f"  Agenda:\n"
        f"    1. Business outcomes review\n"
        f"    2. Platform adoption metrics\n"
        f"    3. Roadmap preview + feature requests\n"
        f"    4. Expansion discussion\n"
        f"    5. Success plan for next quarter\n"
        f"  QBR ID: QBR-{random.randint(1000, 9999)}"
    )


# =====================================================================
# Agent creation
# =====================================================================


def create_sdr(model: str) -> Actor:
    return Actor(
        "sdr",
        prompt=(
            "You are an SDR (Sales Development Rep) for an enterprise software company.\n\n"
            "Your job is to find, research, and qualify leads, then hand off hot prospects "
            "to the Account Executive (AE).\n\n"
            "Workflow:\n"
            "1. Search for leads matching the given criteria using search_leads\n"
            "2. Enrich the most promising leads using enrich_lead\n"
            "3. Qualify leads using BANT framework with qualify_lead\n"
            "4. Send personalized outreach to qualified leads using send_outreach\n"
            "5. For HOT leads: use discover_agents to find the AE, then delegate_to "
            "the AE with full qualification details and recommend next steps\n\n"
            "Always include specific company data and qualification scores when delegating. "
            "Be thorough in research but efficient in execution."
        ),
        config=GeminiConfig(model=model, temperature=0.3),
        tools=[search_leads, enrich_lead, qualify_lead, send_outreach],
        observers=[TokenMonitor(warn_threshold=15_000, alert_threshold=40_000)],
    )


def create_ae(model: str) -> Actor:
    return Actor(
        "ae",
        prompt=(
            "You are an Account Executive (AE) managing enterprise software deals.\n\n"
            "Your job is to progress deals through the pipeline from demo to close.\n\n"
            "Workflow:\n"
            "1. Review incoming leads/deals and create opportunities using create_opportunity\n"
            "2. Schedule demos for new prospects using schedule_demo\n"
            "3. Prepare proposals for engaged prospects using prepare_proposal\n"
            "4. Update deal stages as they progress using update_deal_stage\n"
            "5. When technical validation is needed: discover_agents and delegate_to "
            "the SE (Solutions Engineer) with specific technical questions\n"
            "6. When a deal closes: delegate_to CS (Customer Success) for onboarding handoff\n\n"
            "Existing pipeline:\n"
            + "\n".join(
                f"  - {d['id']}: {d['company']} (${d['deal_size']:,}, {d['stage']}) — {d['next_step']}"
                for d in CRM_PIPELINE
            )
            + "\n\nAlways include deal context when delegating to other agents."
        ),
        config=GeminiConfig(model=model if "pro" in model else "gemini-3.1-pro-preview", temperature=0.3),
        tools=[create_opportunity, schedule_demo, prepare_proposal, update_deal_stage],
        observers=[TokenMonitor(warn_threshold=15_000, alert_threshold=40_000), LoopDetector()],
    )


def create_se(model: str) -> Actor:
    return Actor(
        "se",
        prompt=(
            "You are a Solutions Engineer (SE) providing technical validation for deals.\n\n"
            "Your job is to assess technical fit, design solutions, and support the AE.\n\n"
            "Workflow:\n"
            "1. Assess technical requirements using assess_technical_requirements\n"
            "2. Design solution architecture using design_solution if needed\n"
            "3. Create POC plans using create_poc_plan for complex evaluations\n"
            "4. Review security/compliance using review_security_compliance\n"
            "5. Report findings back — delegate_to AE with your technical recommendation\n\n"
            "Be thorough but business-focused. Frame technical findings in terms of "
            "risk, timeline, and investment required."
        ),
        config=GeminiConfig(model=model if "pro" in model else "gemini-3.1-pro-preview", temperature=0.3),
        tools=[assess_technical_requirements, design_solution, create_poc_plan, review_security_compliance],
        observers=[TokenMonitor(warn_threshold=15_000, alert_threshold=40_000)],
    )


def create_marketing(model: str) -> Actor:
    return Actor(
        "marketing",
        prompt=(
            "You are a Marketing Manager supporting the sales team.\n\n"
            "Your job is to provide lead intelligence, content, and campaign support.\n\n"
            "Workflow:\n"
            "1. Analyze campaign performance using analyze_campaign_performance\n"
            "2. Score lead engagement using score_lead_engagement\n"
            "3. Generate content for sales enablement using generate_content\n"
            "4. Create ABM campaigns for target accounts using create_abm_campaign\n"
            "5. Share lead intelligence with SDR — delegate_to SDR with engagement "
            "data and content recommendations\n\n"
            "Focus on actionable insights that help sales prioritize and personalize."
        ),
        config=GeminiConfig(model=model, temperature=0.3),
        tools=[analyze_campaign_performance, generate_content, score_lead_engagement, create_abm_campaign],
        observers=[TokenMonitor(warn_threshold=10_000, alert_threshold=25_000)],
    )


def create_cs(model: str) -> Actor:
    return Actor(
        "cs",
        prompt=(
            "You are a Customer Success Manager (CSM).\n\n"
            "Your job is to onboard customers, monitor health, and drive expansion.\n\n"
            "Workflow:\n"
            "1. Create onboarding plans for new customers using create_onboarding_plan\n"
            "2. Monitor customer health using check_health_score\n"
            "3. Identify expansion opportunities using identify_expansion_opportunity\n"
            "4. Schedule QBRs using schedule_qbr\n"
            "5. For expansion opportunities: delegate_to AE with opportunity details "
            "so they can create new deals\n\n"
            "Existing customers:\n"
            + "\n".join(
                f"  - {c['company']} (${c['contract_value']:,}, health: {c['health_score']}, "
                f"renewal: {c['renewal_date']})"
                for c in CRM_CUSTOMERS
            )
            + "\n\nProactively identify risks and expansion signals."
        ),
        config=GeminiConfig(model=model if "pro" in model else "gemini-3.1-pro-preview", temperature=0.3),
        tools=[create_onboarding_plan, check_health_score, identify_expansion_opportunity, schedule_qbr],
        observers=[TokenMonitor(warn_threshold=10_000, alert_threshold=25_000)],
    )


# =====================================================================
# Scenarios
# =====================================================================

SCENARIOS = {
    1: (
        "New Territory — SDR Prospecting",
        "sdr",
        "Research and qualify enterprise leads in the healthcare industry, West Coast "
        "region, 500+ employees. Find the best prospects, enrich their profiles with "
        "technographics and recent news, qualify them using the BANT framework, and "
        "send personalized outreach to the top candidates. For any HOT-rated leads, "
        "delegate them to the Account Executive with full qualification details.",
    ),
    2: (
        "Deal Acceleration — AE Pipeline Push",
        "ae",
        "We need to accelerate our pipeline this quarter. Review the current deals: "
        "CloudForge Technologies ($120K, Demo stage), Meridian Health Systems ($250K, "
        "Proposal stage), and NovaPay Financial ($180K, Negotiation stage). For CloudForge, "
        "schedule a technical deep-dive demo and bring in Solutions Engineering for "
        "validation. For Meridian, prepare the formal proposal. For NovaPay, update "
        "the deal stage and prepare to close. Progress each deal to the next stage.",
    ),
    3: (
        "Autonomous Pipeline — Scheduled Reviews",
        None,  # Scheduler-driven, no initial agent
        "",
    ),
    4: (
        "Customer Expansion — CS Revenue Growth",
        "cs",
        "Review our top customer accounts — Synapse Analytics, Guardian Insurance Group, "
        "and Velocity Logistics — for expansion opportunities. Check each customer's "
        "health score, identify upsell and cross-sell potential, and for any strong "
        "expansion opportunities, delegate to the Account Executive to create new "
        "pipeline. Also schedule QBRs for any customers due for quarterly review.",
    ),
}

# =====================================================================
# Main
# =====================================================================


async def main() -> None:
    parser = argparse.ArgumentParser(description="GTM Sales Pipeline Demo")
    parser.add_argument("--scenario", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--model", default="gemini-3-flash-preview")
    parser.add_argument("--duration", type=int, default=40, help="Scheduler duration (scenario 3)")
    parser.add_argument("message", nargs="?", default=None)
    args = parser.parse_args()

    scenario_title, entry_agent_name, message = SCENARIOS[args.scenario]
    if args.message:
        message = args.message
        scenario_title = "Custom Task"

    model = args.model

    # -- Create agents ------------------------------------------------
    sdr = create_sdr(model)
    ae = create_ae(model)
    se = create_se(model)
    marketing = create_marketing(model)
    cs = create_cs(model)

    agent_map = {"sdr": sdr, "ae": ae, "se": se, "marketing": marketing, "cs": cs}

    # -- Create plugins -----------------------------------------------
    pipeline_guard = PipelineGuard()
    deal_tracker = DealTracker()
    telemetry = TelemetryPlugin()

    # -- Create network -----------------------------------------------
    network = Network(
        topology=Pipeline(pipeline_guard),
        plugins=[deal_tracker, telemetry],
        max_delegation_depth=5,
    )

    await network.register(
        sdr,
        capabilities=["prospecting", "outreach", "qualification"],
        description="SDR — researches, qualifies, and nurtures leads",
    )
    await network.register(
        ae,
        capabilities=["deals", "negotiation", "demos", "proposals"],
        description="AE — manages deals from demo to close",
    )
    await network.register(
        se,
        capabilities=["technical", "architecture", "integration", "security"],
        description="SE — validates technical fit and designs solutions",
    )
    await network.register(
        marketing,
        capabilities=["content", "campaigns", "analytics", "lead-scoring"],
        description="Marketing — provides lead intelligence and content",
    )
    await network.register(
        cs,
        capabilities=["onboarding", "retention", "expansion", "health"],
        description="CS — onboards customers and drives expansion",
    )

    # -- Subscribe to hub stream for live logging ---------------------
    async def _on_hub_event(event: object) -> None:
        if isinstance(event, SchedulerTriggerFired):
            print()
            print(
                f"  {_DIM}{_ts()}{_RESET}  "
                f"{_CYAN}{_BOLD}=== SCHEDULER: {event.target.upper()} triggered ==={_RESET}"
            )
            task_preview = event.task[:120]
            print(f"  {_DIM}{' ' * 13}{_RESET}  {_CYAN}{task_preview}{_RESET}")
            print()
        elif isinstance(event, DelegationRequest):
            print(
                f"  {_DIM}{_ts()}{_RESET}  "
                f"{_MAGENTA}{_BOLD}DELEGATE{_RESET} "
                f"{_MAGENTA}{event.source} -> {event.target}{_RESET}"
            )
            task_preview = event.task[:120]
            print(
                f"  {_DIM}{' ' * 13}{_RESET}  " f"{_DIM}{task_preview}{'...' if len(event.task) > 120 else ''}{_RESET}"
            )
        elif isinstance(event, DelegationResult):
            preview = event.result[:120].replace("\n", " | ")
            print(
                f"  {_DIM}{_ts()}{_RESET}  "
                f"{_GREEN}{_BOLD}RESULT{_RESET}   "
                f"{_GREEN}{event.target} -> {event.source}{_RESET}"
            )
            print(f"  {_DIM}{' ' * 13}{_RESET}  " f"{_DIM}{preview}{'...' if len(event.result) > 120 else ''}{_RESET}")
        elif isinstance(event, DelegationRejected):
            print(
                f"  {_DIM}{_ts()}{_RESET}  "
                f"{_RED}{_BOLD}REJECTED{_RESET} "
                f"{_RED}{event.source} -> {event.target}: {event.reason}{_RESET}"
            )
        elif isinstance(event, Signal):
            sev = event.severity.upper() if isinstance(event.severity, str) else str(event.severity)
            color = _RED if "CRITICAL" in sev or "FATAL" in sev else _YELLOW
            print(f"  {_DIM}{_ts()}{_RESET}  " f"{color}{_BOLD}ALERT [{sev}]{_RESET} {color}{event.message}{_RESET}")

    network.hub.stream.subscribe(_on_hub_event)

    # -- Print header -------------------------------------------------
    print()
    print(f"  {_BOLD}{'=' * 64}{_RESET}")
    print(f"  {_BOLD}AI-POWERED GTM SALES PIPELINE{_RESET}")
    print(f"  {_BOLD}{'=' * 64}{_RESET}")
    print()
    print(f"  {_CYAN}Scenario:{_RESET}  {scenario_title}")
    print(f"  {_CYAN}Model:{_RESET}     {model}")
    print(f"  {_CYAN}Agents:{_RESET}    SDR, AE (pro), SE (pro), Marketing, CS (pro)")
    print()
    print(f"  {_BOLD}Pipeline Routing Rules (PipelineGuard):{_RESET}")
    print("    SDR -> AE, Marketing")
    print("    AE  -> SE, CS, SDR")
    print("    SE  -> AE")
    print("    Marketing -> SDR")
    print("    CS  -> AE")
    print()

    if message:
        print(f"  {_BOLD}Task:{_RESET}")
        words = message.split()
        line = "    "
        for w in words:
            if len(line) + len(w) + 1 > 76:
                print(line)
                line = "    " + w
            else:
                line += " " + w if len(line) > 4 else w
        if len(line) > 4:
            print(line)
        print()

    print(f"  {_DIM}{'─' * 64}{_RESET}")
    print(f"  {_BOLD}Log legend:{_RESET}")
    print(f"    {_MAGENTA}DELEGATE{_RESET}        agent-to-agent delegation")
    print(f"    {_GREEN}RESULT{_RESET}          delegation completed")
    print(f"    {_CYAN}SCHEDULER{_RESET}       scheduled watch fired")
    print(f"    {_YELLOW}PIPELINE GUARD{_RESET}  routing validation")
    print(f"    {_RED}ALERT{_RESET}           observer signal")
    print(f"  {_DIM}{'─' * 64}{_RESET}")
    print()

    # -- Run ----------------------------------------------------------
    t0 = time.monotonic()

    if args.scenario == 3:
        # Autonomous pipeline — scheduler-driven
        network.schedule(
            IntervalWatch(15),
            target="ae",
            task=(
                "Periodic pipeline review: Check all current deals. For any deal that "
                "has been in the same stage for over 10 days, take action to progress it. "
                "Schedule demos, prepare proposals, or bring in SE for technical validation "
                "as needed."
            ),
        )
        network.schedule(
            IntervalWatch(20),
            target="sdr",
            task=(
                "Lead nurture check: Search for new leads in technology and healthcare "
                "industries. Enrich and qualify any promising prospects. Send follow-up "
                "outreach to warm leads."
            ),
        )
        network.schedule(
            IntervalWatch(25),
            target="cs",
            task=(
                "Customer health check: Review health scores for existing customers. "
                "Flag any at-risk accounts. Identify expansion opportunities and "
                "delegate strong ones to the AE."
            ),
        )

        print(f"  {_BOLD}Starting autonomous pipeline operations...{_RESET}")
        print(f"  {_DIM}(Scheduler running for {args.duration}s with 3 autonomous cycles){_RESET}")
        print("    AE pipeline review: every 15s")
        print("    SDR lead nurture: every 20s")
        print("    CS health check: every 25s")
        print()

        async with network:
            await asyncio.sleep(args.duration)

    else:
        # Interactive scenario
        entry_agent = agent_map[entry_agent_name]  # type: ignore[index]
        async with network:
            reply = await network.ask(entry_agent, message)

        elapsed = time.monotonic() - t0
        print()
        print(f"  {_BOLD}{'=' * 64}{_RESET}")
        print(f"  {_GREEN}{_BOLD}PIPELINE RESULT{_RESET}  {_DIM}({elapsed:.1f}s){_RESET}")
        print(f"  {_BOLD}{'=' * 64}{_RESET}")
        print()
        for rline in (reply.content or "").split("\n"):
            print(f"  {rline}")
        print()

    # -- Print summary ------------------------------------------------
    elapsed_total = time.monotonic() - t0
    print(f"  {_DIM}{'─' * 64}{_RESET}")
    print(f"  {_BOLD}Pipeline Guard:{_RESET}")
    print(f"    Routes validated: {pipeline_guard.total_routed}")
    print(f"    Warnings: {_YELLOW}{pipeline_guard.total_warnings}{_RESET}")
    if pipeline_guard.route_log:
        print("    Route log:")
        for src, tgt, ok in pipeline_guard.route_log:
            status = f"{_GREEN}OK{_RESET}" if ok else f"{_YELLOW}WARN{_RESET}"
            print(f"      {src} -> {tgt}: {status}")
    print()
    print(f"  {_BOLD}Deal Tracker:{_RESET}")
    print(f"    Events logged: {len(deal_tracker.delegation_log)}")
    for entry in deal_tracker.delegation_log[-6:]:
        etype = entry["type"]
        if etype == "request":
            print(f"    {_DIM}{entry['time']}{_RESET} " f"{_MAGENTA}{entry['source']} -> {entry['target']}{_RESET}")
        else:
            print(f"    {_DIM}{entry['time']}{_RESET} " f"{_GREEN}{entry['target']} completed{_RESET}")
    print()
    print(f"  {_BOLD}Telemetry:{_RESET}")
    m = telemetry.metrics
    print(f"    Total delegations: {m.total_delegations}")
    print(f"    Total completions: {m.total_completions}")
    if m.by_target:
        print(f"    By target: {dict(m.by_target)}")
    print(f"    Total time: {elapsed_total:.1f}s")
    print()


if __name__ == "__main__":
    asyncio.run(main())
