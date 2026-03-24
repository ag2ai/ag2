# Voice of Customer Intelligence — NovaBand X1

Category 11: Voice of Customer (VoC) — always-on social listening network with AI-powered classification and content-based routing to specialist agents.

```
                         ┌────────────────┐
                         │   Scheduler    │
                         │ (IntervalWatch)│
                         └───────┬────────┘
                                 │ triggers periodically
                         ┌───────▼────────┐
                         │   collector    │  tools: search_x, search_reddit,
                         │   (flash)      │         search_reviews, search_news
                         └───────┬────────┘
                                 │ delegates all mentions
                         ┌───────▼────────┐
                         │    analyst     │  tools: analyze_mention,
                         │    (pro)       │         check_trending, create_brief
                         └───────┬────────┘
                                 │ routes by AI classification
            ┌────────────┬───────┴────────┬──────────────┐
    ┌───────▼──────┐ ┌───▼──────┐ ┌───────▼───────┐ ┌───▼──────────┐
    │   product    │ │  market  │ │      pr       │ │  escalation  │
    │  inspector   │ │  intel   │ │   responder   │ │   (pro)      │
    │   (flash)    │ │ (flash)  │ │    (pro)      │ │              │
    └──────────────┘ └──────────┘ └───────────────┘ └──────────────┘
     defects/quality  competitors   viral/PR crisis   safety/legal

    Observers:                        Plugins:
    ├ VolumeTracker (analyst)         ├ ContentRouter (routing)
    ├ SentimentMonitor (analyst)      ├ InsightTracker (system)
    ├ SourceHealthCheck (collector)   └ TelemetryPlugin (system)
    ├ TokenMonitor (all agents)
    └ LoopDetector (analyst, escalation)
```

## What This Demo Shows

- **Always-on social listening**: Collector agent scans X, Reddit, reviews, and news — continuously in scheduler mode or on-demand
- **AI-powered classification**: Analyst uses semantic understanding (not keyword rules) to categorize mentions into defects, competitor intel, PR risks, safety concerns, praise, and feature requests
- **Content-based routing**: Analyst dynamically routes to different specialists based on what each mention is about — the core pattern that makes AI agents superior to rule-based VoC tools
- **3 custom observers**: VolumeTracker detects mention spikes (especially safety), SentimentMonitor tracks sentiment drift, SourceHealthCheck monitors data source reliability
- **ContentRouter plugin**: Validates analyst-to-specialist routing with soft enforcement
- **InsightTracker plugin**: System plugin tracks all VoC insights flowing through the network
- **6 specialized agents**: Each with domain-specific tools and tailored prompts
- **Realistic simulated data**: 29 social media mentions across 5 platforms with realistic engagement metrics

## Why This Pattern Matters

Traditional VoC tools (Brandwatch, Sprout Social, Medallia) rely on keyword matching and rule-based routing. They miss:
- Context: "this charger melts after 2 hours" is a **safety issue**, not just a mention containing "charger"
- Nuance: a viral thread from a 500K-follower influencer needs **PR response**, not just a defect ticket
- Cross-category routing: one post comparing price AND mentioning a defect needs to reach **both** market-intel and product-inspector

AI agents understand context, assess urgency semantically, and route with judgment — turning VoC from a reporting tool into an autonomous response system.

## Prerequisites

```bash
source .venv-beta/bin/activate
export GOOGLE_API_KEY="..."
```

## Scenarios

### Scenario 1: Product Launch Monitoring (default)

```bash
python playground/11_voice_of_customer/main.py
```

Full channel scan 3 days after launch. Collector pulls from all sources, analyst classifies ~29 mentions, and routes to all four specialist teams based on content.

**Delegation chain:** collector -> analyst -> product-inspector, market-intel, pr-responder, escalation (in parallel based on category)

### Scenario 2: Crisis Detection — Safety Issue Spike

```bash
python playground/11_voice_of_customer/main.py --scenario 2
```

Focused scan on safety-related mentions (overheating, burns, skin irritation). Analyst prioritizes safety/legal routing. VolumeTracker observer fires CRITICAL alert on safety spike. Escalation agent creates urgent tickets and notifies stakeholders.

**Delegation chain:** collector -> analyst -> escalation (primary), product-inspector (secondary)

### Scenario 3: Autonomous Monitoring — Continuous VoC Pipeline

```bash
python playground/11_voice_of_customer/main.py --scenario 3
python playground/11_voice_of_customer/main.py --scenario 3 --duration 90
```

Two scheduled watches run autonomously:
- **Collector channel scan** (every 20s): Scans all channels for new mentions
- **Analyst trend review** (every 35s): Reviews trending topics and routes emerging patterns

### Scenario 4: Competitive Intelligence

```bash
python playground/11_voice_of_customer/main.py --scenario 4
```

Focused collection on competitor comparison mentions (NovaBand X1 vs Apple Watch, Samsung, Fitbit). Analyst routes to market-intel for competitive analysis and briefing.

**Delegation chain:** collector -> analyst -> market-intel (primary)

## What to Watch For

1. **Content-based routing**: Watch how the analyst classifies the same batch of mentions into different categories and routes them to different specialists
2. **Observer alerts**: VolumeTracker fires when safety mentions spike; SentimentMonitor warns on negative sentiment drift
3. **ContentRouter validation**: The routing plugin logs whether analyst delegations match expected category-to-specialist mapping
4. **Specialist depth**: Each specialist uses domain-specific tools — product-inspector cross-references the defect database, market-intel runs feature comparisons, PR assesses viral risk scores
5. **InsightTracker dashboard**: End-of-run summary shows which specialists received the most work and the full delegation flow
6. **Pro model for complex tasks**: Analyst, PR responder, and escalation use gemini-3.1-pro for nuanced classification and crisis communication
7. **Scheduler autonomy** (scenario 3): Continuous pipeline operating without human input

## Architecture

- **ContentRouter** (routing plugin): Sits in `Pipeline` topology, validates analyst -> specialist routing
- **InsightTracker** (system plugin): Observes `hub.stream` for delegation events, tracks specialist activity
- **TelemetryPlugin** (system plugin): Tracks delegation metrics
- **VolumeTracker** (observer on analyst): `BatchWatch(n=3, ToolCallEvent)` — detects category spikes
- **SentimentMonitor** (observer on analyst): `EventWatch(ToolResultEvent)` — tracks sentiment drift
- **SourceHealthCheck** (observer on collector): `EventWatch(ToolResultEvent)` — monitors source reliability
- **Network**: Hub + Scheduler with `max_delegation_depth=6`
- **Model split**: Collector/product-inspector/market-intel use flash, analyst/pr-responder/escalation use pro

## Simulated Data

- **Product**: NovaBand X1 smartwatch by NovaTech ($349)
- **8 X mentions**: Battery complaints, positive reviews, competitor comparisons, safety concerns, influencer rant
- **6 Reddit posts**: Battery drain threads, competitor comparisons, charging hazard warning, class action discussion
- **6 Reviews**: Google Reviews and App Store, 1-5 stars covering defects, praise, and safety
- **4 News articles**: TechCrunch review, The Verge battery coverage, Consumer Reports safety investigation, Bloomberg stock impact
- **5 known defects**: Battery drain, screen cracking, charger overheating, skin irritation, HR sensor inaccuracy
- **4 competitors**: Apple Watch Ultra 3, Samsung Galaxy Watch 7, Fitbit Sense 4, Garmin Venu 4
