#!/usr/bin/env python3
"""Loan Application Pipeline — human-in-the-loop agent network.

Category 7: Human-in-the-Loop Network

Three agents (intake, credit-analyst, underwriter) process loan applications
through a Hub. A custom ApprovalGate routing plugin sits in the delegation
path and pauses for human approval before critical agents receive work.

Usage:
    python playground/07_loan_pipeline/main.py
    python playground/07_loan_pipeline/main.py --scenario 2
    python playground/07_loan_pipeline/main.py --scenario 3
    python playground/07_loan_pipeline/main.py --model gemini-3-flash-preview
    python playground/07_loan_pipeline/main.py --auto-approve
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
    BasePlugin,
    DelegationRejected,
    DelegationRequest,
    DelegationResult,
    Envelope,
    HubContext,
    Network,
    Pipeline,
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
_BG_YELLOW = "\033[43m"
_BG_RED = "\033[41m"
_BG_GREEN = "\033[42m"
_BLACK = "\033[30m"


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _wrap(text: str, indent: int = 4, width: int = 72) -> str:
    """Word-wrap text with the given indent and line width."""
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
# ApprovalGate — routing plugin for human-in-the-loop
# =====================================================================


class ApprovalGate(BasePlugin):
    """Routing plugin that pauses delegation for human approval.

    Sits in the Pipeline topology (delegation path). When a delegation
    targets an agent in the require_approval_for list, the plugin
    pauses and prompts the human operator to approve or reject.

    If rejected, returns None — the Hub emits DelegationRejected and
    the calling agent receives an error message.
    """

    def __init__(
        self,
        require_approval_for: list[str] | None = None,
        auto_approve: bool = False,
    ) -> None:
        super().__init__()
        self._require_for = require_approval_for or []
        self._auto_approve = auto_approve
        # Metrics
        self.total_reviewed = 0
        self.total_approved = 0
        self.total_rejected = 0

    async def process(self, envelope: Envelope, ctx: HubContext) -> Envelope | None:
        if envelope.recipient not in self._require_for:
            return envelope

        self.total_reviewed += 1

        # Display the approval prompt
        source = envelope.sender
        target = envelope.recipient
        task_preview = str(getattr(envelope.event, "task", "(no task)"))[:300]

        print()
        print(f"  {_BG_YELLOW}{_BLACK}{_BOLD} APPROVAL GATE {_RESET}")
        print(f"  {_YELLOW}{'=' * 56}{_RESET}")
        print(f"  {_YELLOW}Delegation requires human approval{_RESET}")
        print()
        print(f"    {_BOLD}From:{_RESET}   {source}")
        print(f"    {_BOLD}To:{_RESET}     {target}")
        print(f"    {_BOLD}Task:{_RESET}")
        for line in _wrap(task_preview, indent=12, width=68).split("\n"):
            print(f"  {line}")
        if len(task_preview) >= 300:
            print(f"            {_DIM}(truncated){_RESET}")
        print()
        print(f"  {_YELLOW}{'=' * 56}{_RESET}")

        if self._auto_approve:
            print(f"  {_DIM}[--auto-approve] Automatically approved{_RESET}")
            print()
            self.total_approved += 1
            return envelope

        # Prompt for human input (blocking input in async context)
        response = await asyncio.get_event_loop().run_in_executor(None, input, f"  {_BOLD}Approve? [Y/n]: {_RESET}")
        print()

        if response.strip().lower() in ("n", "no", "reject"):
            self.total_rejected += 1
            print(
                f"  {_BG_RED}{_WHITE}{_BOLD} REJECTED {_RESET}"
                f"  {_RED}Delegation to {target} was rejected by loan officer{_RESET}"
            )
            print()
            return None

        self.total_approved += 1
        print(f"  {_BG_GREEN}{_BLACK}{_BOLD} APPROVED {_RESET}  {_GREEN}Delegation to {target} proceeding{_RESET}")
        print()
        return envelope


# =====================================================================
# Tools — Intake Agent
# =====================================================================


@tool
async def collect_application(
    applicant_name: str,
    loan_amount: float,
    loan_purpose: str,
    annual_income: float,
    employment_years: int,
) -> str:
    """Collect and create a new loan application.

    Args:
        applicant_name: Full name of the applicant.
        loan_amount: Requested loan amount in dollars.
        loan_purpose: Purpose of the loan (mortgage, business, personal, auto).
        annual_income: Applicant's annual income in dollars.
        employment_years: Years at current employer.
    """
    app_id = f"LA-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000, 9999)}"
    return (
        f"Application Created\n"
        f"{'=' * 40}\n"
        f"  Application ID: {app_id}\n"
        f"  Applicant: {applicant_name}\n"
        f"  Loan Amount: ${loan_amount:,.2f}\n"
        f"  Purpose: {loan_purpose}\n"
        f"  Annual Income: ${annual_income:,.2f}\n"
        f"  Employment: {employment_years} years at current employer\n"
        f"  Status: INTAKE\n"
        f"  Submitted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


@tool
async def verify_documents(applicant_name: str, document_types: str) -> str:
    """Verify submitted documents for a loan application.

    Args:
        applicant_name: Name of the applicant.
        document_types: Comma-separated list of document types to verify
            (e.g. "W2, pay stubs, bank statements, tax returns").
    """
    docs = [d.strip() for d in document_types.split(",")]
    lines = [f"Document Verification for {applicant_name}", "=" * 40]
    for doc in docs:
        status = random.choice(["VERIFIED", "VERIFIED", "VERIFIED", "PENDING"])
        lines.append(f"  {doc}: {status}")
    all_verified = all(line.strip() for line in lines if "PENDING" not in line.strip())
    lines.append(f"\n  Overall: {'ALL VERIFIED' if all_verified else 'SOME PENDING — follow up required'}")
    return "\n".join(lines)


@tool
async def calculate_dti(monthly_income: float, monthly_debts: float) -> str:
    """Calculate debt-to-income ratio.

    Args:
        monthly_income: Monthly gross income in dollars.
        monthly_debts: Total monthly debt obligations in dollars.
    """
    dti = (monthly_debts / monthly_income * 100) if monthly_income > 0 else 0
    assessment = (
        "Excellent (under 20%)"
        if dti < 20
        else (
            "Good (20-35%)"
            if dti < 36
            else "Acceptable (36-43%)"
            if dti < 44
            else "High (over 43%) — may affect approval"
        )
    )
    return (
        f"Debt-to-Income Calculation\n"
        f"{'=' * 40}\n"
        f"  Monthly Income: ${monthly_income:,.2f}\n"
        f"  Monthly Debts: ${monthly_debts:,.2f}\n"
        f"  DTI Ratio: {dti:.1f}%\n"
        f"  Assessment: {assessment}"
    )


intake_tools = [collect_application, verify_documents, calculate_dti]


# =====================================================================
# Tools — Credit Analyst
# =====================================================================


@tool
async def pull_credit_report(applicant_name: str) -> str:
    """Pull a credit report for the applicant.

    Args:
        applicant_name: Full name of the applicant.
    """
    score = random.randint(620, 800)
    accounts = random.randint(5, 20)
    derogatories = random.randint(0, 3)
    oldest_account = random.randint(3, 25)
    utilization = random.randint(10, 65)
    return (
        f"Credit Report — {applicant_name}\n"
        f"{'=' * 40}\n"
        f"  Credit Score: {score}\n"
        f"  Open Accounts: {accounts}\n"
        f"  Derogatory Marks: {derogatories}\n"
        f"  Oldest Account: {oldest_account} years\n"
        f"  Credit Utilization: {utilization}%\n"
        f"  Recent Inquiries: {random.randint(0, 4)}\n"
        f"  Payment History: {random.choice(['98%', '95%', '100%', '92%'])} on-time"
    )


@tool
async def assess_risk(
    credit_score: int,
    dti_ratio: float,
    loan_amount: float,
    employment_years: int,
) -> str:
    """Assess the overall risk level of a loan application.

    Args:
        credit_score: Applicant's credit score.
        dti_ratio: Debt-to-income ratio as a percentage.
        loan_amount: Requested loan amount.
        employment_years: Years at current employer.
    """
    # Risk scoring
    score_risk = "low" if credit_score >= 740 else "medium" if credit_score >= 680 else "high"
    dti_risk = "low" if dti_ratio < 30 else "medium" if dti_ratio < 43 else "high"
    emp_risk = "low" if employment_years >= 3 else "medium" if employment_years >= 1 else "high"

    risk_map = {"low": 1, "medium": 2, "high": 3}
    avg = (risk_map[score_risk] + risk_map[dti_risk] + risk_map[emp_risk]) / 3
    overall = "LOW" if avg < 1.5 else "MEDIUM" if avg < 2.5 else "HIGH"

    return (
        f"Risk Assessment\n"
        f"{'=' * 40}\n"
        f"  Credit Score Risk: {score_risk.upper()} (score: {credit_score})\n"
        f"  DTI Risk: {dti_risk.upper()} (ratio: {dti_ratio:.1f}%)\n"
        f"  Employment Risk: {emp_risk.upper()} ({employment_years} years)\n"
        f"  Loan Amount: ${loan_amount:,.2f}\n"
        f"\n"
        f"  Overall Risk Level: {overall}\n"
        f"  Recommendation: {'Proceed to underwriting' if overall != 'HIGH' else 'Additional review required'}"
    )


@tool
async def check_fraud_indicators(applicant_name: str) -> str:
    """Check for fraud indicators in the application.

    Args:
        applicant_name: Full name of the applicant.
    """
    flags = random.choice([
        [],
        [],
        [],
        ["Address mismatch between application and credit report"],
        ["Multiple recent applications at different lenders"],
    ])
    lines = [
        f"Fraud Check — {applicant_name}",
        "=" * 40,
        "  Identity Verification: PASSED",
        "  OFAC/SDN Check: CLEAR",
        f"  Address Verification: {'FLAG' if any('Address' in f for f in flags) else 'CLEAR'}",
        f"  Application Velocity: {'FLAG' if any('Multiple' in f for f in flags) else 'NORMAL'}",
    ]
    if flags:
        lines.append("\n  Flags Found:")
        for flag in flags:
            lines.append(f"    - {flag}")
        lines.append("  Recommendation: Proceed with caution, flag for manual review")
    else:
        lines.append("\n  No fraud indicators detected.")
        lines.append("  Recommendation: Clear to proceed")
    return "\n".join(lines)


credit_tools = [pull_credit_report, assess_risk, check_fraud_indicators]


# =====================================================================
# Tools — Underwriter
# =====================================================================


@tool
async def review_application(
    application_id: str,
    risk_level: str,
    credit_score: int,
    dti_ratio: float,
    loan_amount: float,
) -> str:
    """Perform detailed underwriting review of a loan application.

    Args:
        application_id: The application ID.
        risk_level: Overall risk level from credit analysis (LOW/MEDIUM/HIGH).
        credit_score: Applicant's credit score.
        dti_ratio: Debt-to-income ratio as a percentage.
        loan_amount: Requested loan amount.
    """
    ltv = random.randint(60, 95)
    return (
        f"Underwriting Review — {application_id}\n"
        f"{'=' * 40}\n"
        f"  Risk Level: {risk_level}\n"
        f"  Credit Score: {credit_score} ({'meets' if credit_score >= 680 else 'below'} minimum threshold)\n"
        f"  DTI Ratio: {dti_ratio:.1f}% ({'within' if dti_ratio < 43 else 'exceeds'} guidelines)\n"
        f"  Loan Amount: ${loan_amount:,.2f}\n"
        f"  Estimated LTV: {ltv}% ({'acceptable' if ltv <= 80 else 'requires PMI'})\n"
        f"\n"
        f"  Guideline Compliance: {'PASS' if credit_score >= 680 and dti_ratio < 43 else 'CONDITIONAL'}\n"
        f"  Exceptions Required: {'None' if credit_score >= 680 and dti_ratio < 43 else 'DTI/credit exception needed'}"
    )


@tool
async def set_loan_terms(
    loan_amount: float,
    interest_rate: float,
    term_months: int,
) -> str:
    """Set the final loan terms.

    Args:
        loan_amount: Approved loan amount.
        interest_rate: Annual interest rate as a percentage.
        term_months: Loan term in months.
    """
    monthly_payment = loan_amount * (interest_rate / 100 / 12) / (1 - (1 + interest_rate / 100 / 12) ** (-term_months))
    total_interest = monthly_payment * term_months - loan_amount
    return (
        f"Loan Terms Set\n"
        f"{'=' * 40}\n"
        f"  Principal: ${loan_amount:,.2f}\n"
        f"  Interest Rate: {interest_rate:.2f}%\n"
        f"  Term: {term_months} months ({term_months // 12} years)\n"
        f"  Monthly Payment: ${monthly_payment:,.2f}\n"
        f"  Total Interest: ${total_interest:,.2f}\n"
        f"  Total Cost: ${loan_amount + total_interest:,.2f}"
    )


@tool
async def issue_decision(
    application_id: str,
    decision: str,
    conditions: str = "",
) -> str:
    """Issue the final underwriting decision.

    Args:
        application_id: The application ID.
        decision: Decision (approve/deny/conditional).
        conditions: Any conditions attached to the decision.
    """
    decision_upper = decision.upper()
    lines = [
        "UNDERWRITING DECISION",
        "=" * 40,
        f"  Application: {application_id}",
        f"  Decision: {decision_upper}",
        f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "  Underwriter: Automated Review + Human Approval",
    ]
    if conditions:
        lines.append("\n  Conditions:")
        for cond in conditions.split(";"):
            cond = cond.strip()
            if cond:
                lines.append(f"    - {cond}")
    if decision_upper == "APPROVE":
        lines.append("\n  Next Steps: Proceed to closing, schedule appraisal, title search")
    elif decision_upper == "CONDITIONAL":
        lines.append("\n  Next Steps: Applicant must satisfy conditions before final approval")
    else:
        lines.append("\n  Next Steps: Notify applicant, provide adverse action notice")
    return "\n".join(lines)


underwriter_tools = [review_application, set_loan_terms, issue_decision]


# =====================================================================
# Scenarios
# =====================================================================

SCENARIOS = {
    1: (
        "Strong Applicant",
        "New loan application: John Smith, $350,000 mortgage, annual income "
        "$120,000, 8 years at current employer. Documents: W2, pay stubs, "
        "bank statements, tax returns.",
    ),
    2: (
        "Weak Applicant",
        "New loan application: Jane Doe, $500,000 mortgage, annual income "
        "$75,000, 1 year at current employer. Limited documentation: pay "
        "stubs only, no tax returns available.",
    ),
    3: (
        "Business Loan",
        "New loan application: Mike Chen, $150,000 small business loan for "
        "restaurant expansion. Annual personal income $85,000, business "
        "revenue $450,000. 5 years in business. Documents: business "
        "financials, personal tax returns, business plan.",
    ),
}


# =====================================================================
# Main
# =====================================================================


async def main() -> None:
    # -- Parse CLI args -----------------------------------------------
    model = "gemini-3.1-pro-preview"
    scenario_num = 1
    auto_approve = False

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--scenario" and i + 1 < len(args):
            scenario_num = int(args[i + 1])
            i += 2
        elif args[i] == "--model" and i + 1 < len(args):
            model = args[i + 1]
            i += 2
        elif args[i] == "--auto-approve":
            auto_approve = True
            i += 1
        else:
            i += 1

    title, loan_message = SCENARIOS[scenario_num]

    # -- Create actors ------------------------------------------------
    intake = Actor(
        "intake",
        prompt=(
            "You are a Loan Intake Specialist.\n\n"
            "Your job is to collect the loan application, verify documents, "
            "and calculate the debt-to-income ratio. Then hand off to the "
            "credit analyst with all the details.\n\n"
            "Steps:\n"
            "1. Use collect_application with the applicant's details\n"
            "2. Use verify_documents to check submitted documentation\n"
            "3. Use calculate_dti to compute the debt-to-income ratio "
            "(estimate monthly income from annual, estimate monthly debts "
            "as ~30% of monthly income if not specified)\n"
            "4. Use discover_agents to find a credit analysis agent\n"
            "5. Use delegate_to to send ALL collected information to the "
            "credit analyst — include the application ID, applicant name, "
            "loan amount, income, employment years, DTI ratio, and document "
            "verification status\n"
            "6. Return the final result\n\n"
            "Be thorough — the credit analyst needs complete information."
        ),
        config=GeminiConfig(model=model, temperature=0.3),
        tools=intake_tools,
        observers=[TokenMonitor(warn_threshold=10_000, alert_threshold=30_000)],
    )

    credit_analyst = Actor(
        "credit-analyst",
        prompt=(
            "You are a Credit Analyst.\n\n"
            "Your job is to pull the credit report, assess risk, check for "
            "fraud, and hand off to the underwriter with your recommendation.\n\n"
            "Steps:\n"
            "1. Use pull_credit_report to get the applicant's credit data\n"
            "2. Use assess_risk with the credit score, DTI ratio, loan amount, "
            "and employment years\n"
            "3. Use check_fraud_indicators to screen for fraud\n"
            "4. Use discover_agents to find an underwriting agent\n"
            "5. Use delegate_to to send everything to the underwriter — "
            "include application ID, credit score, risk level, DTI ratio, "
            "loan amount, fraud check results, and your recommendation\n"
            "6. Return the final result\n\n"
            "Your analysis is critical for the underwriting decision."
        ),
        config=GeminiConfig(model=model, temperature=0.3),
        tools=credit_tools,
        observers=[LoopDetector(repeat_threshold=3)],
    )

    underwriter = Actor(
        "underwriter",
        prompt=(
            "You are a Loan Underwriter.\n\n"
            "Your job is to make the final lending decision. Review all "
            "information from the credit analyst, apply underwriting "
            "guidelines, set loan terms if appropriate, and issue the "
            "final decision.\n\n"
            "Steps:\n"
            "1. Use review_application with the application details\n"
            "2. If the application looks approvable, use set_loan_terms "
            "to define the loan terms (use current market rates: ~6.5-7.5% "
            "for mortgages, ~8-12% for business loans)\n"
            "3. Use issue_decision to issue the final approve/deny/conditional "
            "decision with any conditions\n"
            "4. Return a comprehensive summary of the decision, terms, and "
            "any conditions\n\n"
            "Be thorough and justify your decision."
        ),
        config=GeminiConfig(model=model, temperature=0.2),
        tools=underwriter_tools,
        observers=[
            TokenMonitor(warn_threshold=10_000, alert_threshold=30_000),
            LoopDetector(repeat_threshold=3),
        ],
    )

    # -- Create network with ApprovalGate -----------------------------
    approval_gate = ApprovalGate(
        require_approval_for=["underwriter"],
        auto_approve=auto_approve,
    )

    telemetry = TelemetryPlugin()

    network = Network(
        topology=Pipeline(approval_gate),
        plugins=[telemetry],
        max_delegation_depth=4,
    )

    await network.register(
        intake,
        capabilities=["intake", "application", "documents"],
        description="Collects loan applications, verifies documents, calculates DTI",
    )
    await network.register(
        credit_analyst,
        capabilities=["credit", "analysis", "scoring"],
        description="Pulls credit reports, assesses risk, checks for fraud",
    )
    await network.register(
        underwriter,
        capabilities=["underwriting", "approval", "terms"],
        description="Reviews applications, sets loan terms, issues final decisions",
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
            preview = event.result[:100].replace("\n", " | ")
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
    print(f"  {_BOLD}LOAN APPLICATION PIPELINE{_RESET}  {_DIM}(07_loan_pipeline){_RESET}")
    print(f"  {_BOLD}{'=' * 60}{_RESET}")
    print()
    print(f"  {_CYAN}Category:{_RESET}  Human-in-the-Loop Network")
    print(f"  {_CYAN}Scenario:{_RESET}  {title}")
    print(f"  {_CYAN}Model:{_RESET}     {model}")
    print(f"  {_CYAN}Agents:{_RESET}    intake -> credit-analyst -> underwriter")
    print(f"  {_CYAN}Gate:{_RESET}      ApprovalGate on [underwriter]{' (auto-approve)' if auto_approve else ''}")
    print()
    print(f"  {_BOLD}Application:{_RESET}")
    print(_wrap(loan_message, indent=4, width=72))
    print()
    print(f"  {_DIM}{'─' * 60}{_RESET}")
    print(
        f"  {_BOLD}Log legend:{_RESET}  "
        f"{_YELLOW}TOOL{_RESET}=tool call  "
        f"{_MAGENTA}HUB{_RESET}=delegation  "
        f"{_GREEN}HUB{_RESET}=completed  "
        f"{_RED}REJECTED{_RESET}=blocked"
    )
    print(f"  {_DIM}{'─' * 60}{_RESET}")
    print()

    # -- Run ----------------------------------------------------------
    t0 = time.monotonic()
    reply = await network.ask(intake, loan_message)
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

    # -- Print gate metrics -------------------------------------------
    print(f"  {_DIM}{'─' * 60}{_RESET}")
    print(f"  {_BOLD}Approval Gate Metrics:{_RESET}")
    print(f"    Reviewed:  {approval_gate.total_reviewed}")
    print(f"    Approved:  {_GREEN}{approval_gate.total_approved}{_RESET}")
    print(f"    Rejected:  {_RED}{approval_gate.total_rejected}{_RESET}")
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
