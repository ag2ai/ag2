# 12 — Decision Simulation

**Category:** Many-Actor Network (Scale-Out Simulation)

Demonstrates that the AG2 network naturally scales to any number of actors.
A business decision is distributed to **9 autonomous persona actors** — each
analyzing impact from their domain perspective — then an analyst synthesizes
all perspectives into a unified recommendation.

**11 actors total** — nearly double the largest previous example.

## Architecture

```
                         ┌─────────────┐
                         │ coordinator │
                         └──────┬──────┘
                                │ discover_agents("simulation")
           ┌────────────────────┼────────────────────┐
           │     delegate_to    │     (×9 personas)  │
     ┌─────▼─┐  ┌─────▼─┐  ┌──▼──┐  ┌──────▼──────┐
     │  cfo  │  │  cmo  │  │ cto │  │  ...6 more  │
     └───────┘  └───────┘  └─────┘  └─────────────┘
           │                │                │
           └────────────────┼────────────────┘
                            │ all results compiled
                     ┌──────▼──────┐
                     │   analyst   │
                     └─────────────┘
```

## Personas

| Actor | Role | Tool |
|-------|------|------|
| `cfo` | Chief Financial Officer | `run_financial_model` |
| `cmo` | Chief Marketing Officer | `analyze_brand_impact` |
| `cto` | Chief Technology Officer | `assess_technical_feasibility` |
| `hr-director` | HR Director | `assess_workforce_impact` |
| `legal-counsel` | General Counsel | `check_regulatory_compliance` |
| `customer-advocate` | VP of Customer Success | `predict_customer_impact` |
| `operations-lead` | VP of Operations | `analyze_supply_chain_impact` |
| `competitor-analyst` | Competitive Intelligence Lead | `predict_competitor_response` |
| `board-member` | Independent Board Director | `evaluate_strategic_alignment` |

## Scenarios

| # | Title | Decision |
|---|-------|----------|
| 1 | SaaS Price Increase | Raise subscription prices by 25% across all tiers |
| 2 | European Market Expansion | Open Frankfurt data center + EU sales team ($8M) |
| 3 | Legacy Product Sunset | Discontinue on-prem product (20% revenue, 40% eng cost) |

## Usage

```bash
python playground/12_simulation/main.py                     # scenario 1
python playground/12_simulation/main.py --scenario 2        # EU expansion
python playground/12_simulation/main.py --scenario 3        # legacy sunset
python playground/12_simulation/main.py "Your custom decision here"
python playground/12_simulation/main.py --model gemini-3-flash-preview
```

## What to watch for

- **Scale:** 11 actors registered, 10 delegations (9 personas + analyst)
- **Discovery:** Coordinator finds all 9 personas via `discover_agents("simulation")`
- **Sequential fan-out:** Each persona runs independently with its own LLM call
- **Stance detection:** Log shows SUPPORT/OPPOSE/NEUTRAL extracted from each response
- **Synthesis:** Analyst aggregates all perspectives into consensus + recommendation
- **Telemetry:** Final metrics show delegation counts by target
