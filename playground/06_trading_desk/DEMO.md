# Algorithmic Trading Desk

**Category 6: Full Customization (Real-World Complexity)**

The "kitchen sink" demo -- custom priority, custom events, custom plugins, custom harness, custom conflict resolution. Shows every extensibility point of the AG2 network framework in one coherent use case.

```
                          ┌──────────────────────────────────────┐
                          │              NETWORK                 │
                          │                                      │
                          │  Topology: Pipeline                  │
                          │  ┌────────────┐  ┌───────────────┐  │
                          │  │  RiskGate  │->│  RateLimiter  │  │
                          │  │  (routing) │  │  (routing)    │  │
                          │  └────────────┘  └───────────────┘  │
                          │                                      │
                          │  System Plugins:                     │
                          │  [ComplianceAudit] [TelemetryPlugin] │
                          │                                      │
                          │  Priority: TradePriorityScheme       │
                          │  Conflict: HighestPriorityWins       │
                          └──────────┬───┬───┬───┬───────────────┘
                                     │   │   │   │
                ┌────────────────────┘   │   │   └──────────────────┐
                │           ┌────────────┘   └───────────┐          │
                ▼           ▼                             ▼          ▼
          ┌──────────┐ ┌──────────┐              ┌────────────┐ ┌────────────┐
          │ ANALYST  │ │  TRADER  │              │    RISK    │ │ COMPLIANCE │
          │ Network  │ │ Network  │              │  MANAGER   │ │  OFFICER   │
          │ Harness  │ │ Harness  │              │  Conv.     │ │  Conv.     │
          └──────────┘ └──────────┘              │  Harness   │ │  Harness   │
                                                 └────────────┘ └────────────┘

Analyst ──► Trader ──► Risk Manager ◄──► Compliance Officer
```

## Customization Points Demonstrated

| Extension Point | Implementation | Purpose |
|---|---|---|
| **Custom Priority** | `TradePriorityScheme` with `TradePriority` enum (ANALYSIS=0, LIMIT_ORDER=1, MARKET_ORDER=2, RISK_ALERT=3) | Market orders beat limit orders; risk alerts trump everything |
| **Custom Events** | `PriceSignal`, `TradeExecution` via `@register_event` | Domain-specific event types for the trading domain |
| **Routing Plugin** | `RiskGate` in `Pipeline` topology | Vetoes delegations when portfolio exposure exceeds limits; sits in the delegation path |
| **System Plugin** | `ComplianceAudit` in `plugins=[]` | Observes hub.stream for delegation events; builds audit trail without affecting routing |
| **Context Harness** | `NetworkHarness` for analyst/trader, `ConversationHarness` for risk/compliance | Controls what events each agent's LLM sees |
| **Conflict Resolver** | `HighestPriorityWins` | When two delegations conflict, higher priority wins |
| **Topology** | `Pipeline(RiskGate, RateLimiter)` | Sequential processing: risk check first, then rate limiting |

## Prerequisites

```bash
source .venv-beta/bin/activate
export GOOGLE_API_KEY="..."
```

## Quick Start

```bash
# Default scenario (market analysis via analyst)
python playground/06_trading_desk/main.py

# Risk alert scenario (via risk manager)
python playground/06_trading_desk/main.py --scenario 2

# End of day compliance (via compliance officer)
python playground/06_trading_desk/main.py --scenario 3

# Different model
python playground/06_trading_desk/main.py --model gemini-3-flash-preview
```

## Scenarios

### Scenario 1: Market Analysis (default)

Entry agent: **Analyst**. Analyzes NVDA, AAPL, TSLA. Generates trade signals for strong opportunities and delegates to the Trader for execution.

**Expected delegation chain:**
```
Client -> Analyst
             └── delegate_to(trader) -> Hub -> Pipeline(RiskGate, RateLimiter) -> Trader
                                                                                    └── delegate_to(risk-manager) -> Hub -> Risk Manager
```

### Scenario 2: Risk Alert

Entry agent: **Risk Manager**. Portfolio tech exposure at 85% of limit, VaR exceeds threshold. Assesses risk, delegates to Compliance for review.

**Expected delegation chain:**
```
Client -> Risk Manager
              └── delegate_to(compliance-officer) -> Hub -> Compliance Officer
```

### Scenario 3: End of Day

Entry agent: **Compliance Officer**. Generates audit report, reviews trades, delegates to Risk Manager for exposure data, flags violations.

**Expected delegation chain:**
```
Client -> Compliance Officer
              └── delegate_to(risk-manager) -> Hub -> Risk Manager
```

## What to Watch For

1. **RiskGate in action** -- the routing plugin checks every delegation that passes through the topology pipeline. At the end, the metrics show how many delegations were checked vs. vetoed. If you set exposure high enough in the state store, it will block trade delegations.

2. **ComplianceAudit trail** -- printed at the end of execution. Every DelegationRequest, DelegationResult, and DelegationRejected event is recorded with timestamps. This is the full audit log a compliance team would review.

3. **Topology pipeline flow** -- each delegation routes through `RiskGate -> RateLimiter` before reaching the target agent. The `ROUTE` log lines show delegations entering the Hub; `DONE` lines confirm completion.

4. **NetworkHarness vs ConversationHarness** -- the Analyst and Trader use `NetworkHarness`, which means their LLM sees delegation results and signals alongside normal conversation. The Risk Manager and Compliance Officer use `ConversationHarness`, which only shows chat and tool events.

5. **TelemetryPlugin metrics** -- printed at the end, showing total delegations, completions, and breakdowns by source and target agent.

6. **Priority and conflict resolution** -- `TradePriorityScheme` defines the ordering (ANALYSIS < LIMIT_ORDER < MARKET_ORDER < RISK_ALERT). `HighestPriorityWins` resolver is configured for any envelope conflicts.

## Plugin Architecture

**Routing plugins** (in topology) intercept envelopes in the delegation path:
```
Delegation -> Pipeline -> RiskGate.process() -> RateLimiter.process() -> Target Agent
                              │                        │
                              ▼                        ▼
                         Veto (None)             Rate limit (None)
                         or pass through         or pass through
```

**System plugins** (in plugins list) observe the hub stream without affecting routing:
```
Hub.stream ──► ComplianceAudit._on_request()     (records audit entry)
           ──► ComplianceAudit._on_result()      (records audit entry)
           ──► TelemetryPlugin._on_request()     (increments counters)
           ──► TelemetryPlugin._on_result()      (increments counters)
```

This separation means you can add logging, metrics, and compliance without touching the delegation path, while routing plugins like RiskGate can veto or reroute traffic.
