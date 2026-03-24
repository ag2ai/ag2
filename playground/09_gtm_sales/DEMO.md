# AI-Powered GTM Sales Pipeline

Category 9: GTM / Sales Pipeline — multi-agent sales automation with pipeline routing enforcement and autonomous operations.

```
                    ┌─────────────────────┐
            ┌──────>│  AE (Account Exec)  │──────┐
            │       │  gemini-3.1-pro     │      │
            │       └──────┬──────────────┘      │
            │              │                     │
   ┌────────┴──────┐  ┌───▼──────────────┐  ┌───▼──────────────┐
   │  SDR          │  │  SE (Solutions)   │  │  CS (Success)    │
   │  gemini-3-fl  │  │  gemini-3.1-pro   │  │  gemini-3.1-pro  │
   └───────┬───────┘  └──────────────────┘  └───────┬──────────┘
           │                                         │
   ┌───────▼───────┐                                 │
   │  Marketing    │         expansion leads ────────┘
   │  gemini-3-fl  │
   └───────────────┘

   PipelineGuard enforces routing rules:
   SDR -> AE, Marketing  |  AE -> SE, CS, SDR
   SE -> AE  |  Marketing -> SDR  |  CS -> AE
```

## What This Demo Shows

- **PipelineGuard plugin**: Routing rules enforce valid handoffs between sales roles (soft enforcement — warns but allows)
- **DealTracker plugin**: System plugin observes all delegations for pipeline visibility
- **5 specialized agents**: SDR, AE (pro model), SE (pro model), Marketing, CS (pro model)
- **Realistic CRM data**: 8 prospect companies, 4 pipeline deals, 3 existing customers
- **BANT qualification**: SDR uses Budget/Authority/Need/Timeline framework
- **Autonomous pipeline**: Scheduler drives periodic pipeline reviews and lead nurturing
- **Cross-functional coordination**: Agents discover and delegate to each other through the Hub

## Prerequisites

```bash
source .venv-beta/bin/activate
export GOOGLE_API_KEY="..."
```

## Scenarios

### Scenario 1: New Territory — SDR Prospecting (default)

```bash
python playground/09_gtm_sales/main.py
```

SDR researches healthcare leads, enriches profiles, runs BANT qualification, sends personalized outreach, and hands off HOT leads to the AE.

**Delegation chain:** SDR -> AE (hot leads with full qualification data)

### Scenario 2: Deal Acceleration — AE Pipeline Push

```bash
python playground/09_gtm_sales/main.py --scenario 2
```

AE progresses 3 deals: schedules technical demo (delegates to SE), prepares proposal for Meridian, and advances NovaPay negotiation.

**Delegation chain:** AE -> SE (technical validation) -> AE (findings)

### Scenario 3: Autonomous Pipeline — Scheduled Reviews

```bash
python playground/09_gtm_sales/main.py --scenario 3
python playground/09_gtm_sales/main.py --scenario 3 --duration 60
```

Three scheduled watches run autonomously:
- **AE pipeline review** (every 15s): Progress stale deals
- **SDR lead nurture** (every 20s): Find and qualify new leads
- **CS health check** (every 25s): Monitor customers, flag expansion

### Scenario 4: Customer Expansion — CS Revenue Growth

```bash
python playground/09_gtm_sales/main.py --scenario 4
```

CS reviews 3 customer accounts, checks health scores, identifies expansion opportunities, and delegates strong ones to the AE for new deals.

**Delegation chain:** CS -> AE (expansion opportunity with revenue estimate)

## What to Watch For

1. **Pipeline Guard**: Watch for routing validation — the guard logs allowed/warned routes
2. **BANT Qualification**: SDR uses structured qualification before handing off leads
3. **Cross-agent delegation**: See how agents discover each other and delegate with full context
4. **Pro model for complex tasks**: AE, SE, and CS use gemini-3.1-pro for deal strategy
5. **Deal Tracker**: End-of-run summary shows all delegation flow through the pipeline
6. **Scheduler autonomy** (scenario 3): Multiple agents operating without human input

## Architecture

- **PipelineGuard** (routing plugin): Sits in `Pipeline` topology, validates every delegation
- **DealTracker** (system plugin): Observes `hub.stream` for delegation events
- **TelemetryPlugin** (system plugin): Tracks delegation metrics
- **Network**: Hub + Scheduler with `max_delegation_depth=5`
- **Model split**: SDR/Marketing use flash (high volume), AE/SE/CS use pro (complex reasoning)
