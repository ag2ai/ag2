# Loan Application Pipeline

**Category 7: Human-in-the-Loop Network**

Three agents process loan applications through a Hub, but a human loan officer must approve before the underwriter receives any work. A custom `ApprovalGate` routing plugin sits in the delegation path and pauses for human input at critical decision points.

```
                     ┌─────────────────────────────┐
                     │            HUB              │
                     │   topology=Pipeline(        │
                     │     ApprovalGate(            │
                     │       ["underwriter"])       │
                     │   )                         │
                     └──┬──────────┬───────────────┘
                        │          │
                        │          │  APPROVAL GATE
                        │          │  ┌───────────┐
                        │          └──┤ Human Y/N │
                        │             └─────┬─────┘
               ┌────────┘                   │
               │                            ▼
               ▼                     ┌──────────────┐
        ┌──────────────┐             │  UNDERWRITER  │
        │CREDIT ANALYST│             │  (review,     │
        │ (score, risk,│             │   terms,      │
        │  fraud)      │─────────────│   decision)   │
        └──────┬───────┘             └──────────────┘
               ▲
               │
        ┌──────┴───────┐
        │    INTAKE     │
        │ (application, │
        │  documents,   │
        │  DTI)         │
        └──────────────┘

Intake -> Credit Analyst -> [APPROVAL GATE] -> Underwriter
```

## How the Approval Gate Works

The `ApprovalGate` is a **routing plugin** — it sits inside `Pipeline(...)` in the Hub's topology, which means it intercepts every delegation **before** it reaches the target agent.

When a delegation targets an agent in its `require_approval_for` list:
1. The plugin pauses execution and displays the delegation details
2. The human loan officer sees who is delegating, to whom, and the task summary
3. The human enters `Y` (approve) or `N` (reject)
4. **Approve**: the envelope passes through and the target agent receives the task
5. **Reject**: the plugin returns `None`, the Hub emits `DelegationRejected`, and the calling agent receives an error message ("delegation rejected by Hub topology")

This is different from a system plugin (in `plugins=[...]`), which only observes events but cannot block delegations.

## Prerequisites

- Python 3.11+
- A Gemini API key exported as `GOOGLE_API_KEY`
- AG2 installed with beta dependencies (the `.venv-beta` environment)

```bash
source .venv-beta/bin/activate
export GOOGLE_API_KEY="your-key-here"  # pragma: allowlist secret
```

## Running

**Default scenario** (strong applicant):
```bash
python playground/07_loan_pipeline/main.py
```

**Choose a scenario:**
```bash
python playground/07_loan_pipeline/main.py --scenario 1   # Strong applicant
python playground/07_loan_pipeline/main.py --scenario 2   # Weak applicant
python playground/07_loan_pipeline/main.py --scenario 3   # Business loan
```

**Auto-approve** (skips human input, for automated testing):
```bash
python playground/07_loan_pipeline/main.py --auto-approve
python playground/07_loan_pipeline/main.py --scenario 2 --auto-approve
```

**Different model:**
```bash
python playground/07_loan_pipeline/main.py --model gemini-3-flash-preview
```

## Scenarios

### Scenario 1: Strong Applicant (default)

John Smith, $350,000 mortgage, $120,000 annual income, 8 years at employer. Full documentation (W2, pay stubs, bank statements, tax returns).

Expected outcome: Low risk, straightforward approval if human approves.

### Scenario 2: Weak Applicant

Jane Doe, $500,000 mortgage, $75,000 annual income, 1 year at employer. Limited documentation (pay stubs only).

Expected outcome: Higher risk, likely conditional approval or denial. The approval gate is where the human can decide whether to even let the underwriter see this application.

### Scenario 3: Business Loan

Mike Chen, $150,000 small business loan for restaurant expansion. Mixed personal and business income, 5 years in business.

Expected outcome: Different risk profile than consumer mortgage. Tests how agents handle business loan specifics.

## What to Watch For

### The pause at the approval gate
When the credit analyst delegates to the underwriter, the pipeline will stop and display:
```
  APPROVAL GATE
  ========================================================
  Delegation requires human approval

    From:   credit-analyst
    To:     underwriter
    Task:   [preview of what the credit analyst is sending]

  ========================================================
  Approve? [Y/n]:
```

This is the key moment. The human sees exactly what information is being passed and decides whether to proceed.

### Approve vs. reject

**When you approve (Y):**
- The underwriter receives the task and runs its tools (review, set terms, issue decision)
- The full pipeline completes and you see the final lending decision

**When you reject (N):**
- The Hub emits a `DelegationRejected` event
- The credit analyst receives an error: "delegation rejected by Hub topology"
- The credit analyst must respond to this — it cannot reach the underwriter
- The pipeline ends with the credit analyst's response about the rejection

### Tool calls per agent

- **Intake**: `collect_application`, `verify_documents`, `calculate_dti`
- **Credit Analyst**: `pull_credit_report`, `assess_risk`, `check_fraud_indicators`
- **Underwriter**: `review_application`, `set_loan_terms`, `issue_decision`

### Gate metrics

At the end of the run, the demo prints approval gate metrics (reviewed, approved, rejected) and telemetry (total delegations, completions, routing breakdown).

## Try This

1. **Run scenario 1 and approve** — see the full pipeline from intake through underwriting decision:
   ```bash
   python playground/07_loan_pipeline/main.py --scenario 1
   # When prompted, enter: Y
   ```

2. **Run scenario 1 and reject** — see how the system handles a blocked delegation:
   ```bash
   python playground/07_loan_pipeline/main.py --scenario 1
   # When prompted, enter: N
   ```

3. **Run scenario 2 (weak applicant) and approve** — watch how the underwriter handles a risky application:
   ```bash
   python playground/07_loan_pipeline/main.py --scenario 2
   ```

4. **Compare with auto-approve** — run without human interaction for quick testing:
   ```bash
   python playground/07_loan_pipeline/main.py --auto-approve --scenario 1
   ```

## Architecture Notes

- The `ApprovalGate` extends `BasePlugin` and overrides `process()`. It is a routing plugin because it is placed in the `topology=Pipeline(...)`, not in `plugins=[...]`.
- `TelemetryPlugin` is a system plugin (in `plugins=[...]`). It observes delegation events on the Hub stream but does not block anything.
- The gate uses `asyncio.get_event_loop().run_in_executor(None, input, ...)` to perform blocking terminal input without stalling the event loop.
- All agents run in-process via `LocalChannel`. No HTTP, no distribution overhead.
- The `Network` convenience class wires Hub + Scheduler. Here we only use the Hub routing (no scheduled watches).
