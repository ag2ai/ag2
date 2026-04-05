# Newsroom Pipeline

**Category 3: Multiple Actors + Hub (Network of Agents)**

A content creation pipeline where four specialized agents collaborate through a Hub to research, write, edit, and publish an article. Each agent discovers the next agent dynamically and delegates work through the Hub's routing layer.

## Pipeline

```
Researcher  -->  Writer  -->  Editor  -->  Publisher
 (search,       (draft,      (review,     (format,
  analyze)       word count)  grammar,     social posts,
                              fact-check)  schedule)
```

## Prerequisites

- Python 3.11+
- A Gemini API key exported as `GOOGLE_API_KEY`
- AG2 installed with beta dependencies (the `.venv-beta` environment)

```bash
export GOOGLE_API_KEY="your-key-here"  # pragma: allowlist secret
```

## Running

**Default scenario** (AI agent frameworks):
```bash
python playground/03_newsroom/main.py
```

**Scenario 2** (crypto mining environmental impact):
```bash
python playground/03_newsroom/main.py --scenario 2
```

**Scenario 3** (quantum computing advances):
```bash
python playground/03_newsroom/main.py --scenario 3
```

**Custom topic:**
```bash
python playground/03_newsroom/main.py "Write an article about Mars colonization timelines"
```

**Different model:**
```bash
python playground/03_newsroom/main.py --model gemini-3-flash-preview --scenario 1
```

## What to Watch For

### Discovery calls
Each agent uses `discover_agents` to find the next agent in the pipeline at runtime. Watch for the `DISCOVER` log lines -- the agents are not hard-coded to know each other. The Hub's registry provides dynamic lookup by capability.

### Delegation chain
The core flow is a 4-step delegation chain:
1. **Researcher** searches and analyzes, then delegates to **Writer** with findings
2. **Writer** drafts the article, then delegates to **Editor** with the draft
3. **Editor** reviews, grammar-checks, and fact-checks, then delegates to **Publisher**
4. **Publisher** formats, generates social posts, schedules, and returns the result

Each delegation shows up twice in the logs:
- `DELEGATE` (agent-level) -- the agent calling the delegate_to tool
- `HUB DELEGATE` (hub-level) -- the Hub routing the request

### Tool usage per agent
- **Researcher**: `search_web`, `analyze_sources`
- **Writer**: `draft_article`, `check_word_count`
- **Editor**: `review_article`, `check_grammar`, `verify_facts`
- **Publisher**: `format_for_web`, `generate_social_posts`, `schedule_publication`

### Depth tracking
The Hub tracks delegation depth (max 5). Since the pipeline is 4 levels deep (researcher -> writer -> editor -> publisher), it stays within limits. If an agent tried to delegate further from the publisher, the Hub would reject it.

## Architecture Notes

- All agents run **in-process** -- no HTTP, no distribution. The Hub manages everything in memory.
- The Hub auto-injects `discover_agents` and `delegate_to` tools into each agent when it processes a task. Agents do not carry these tools by default.
- Each agent has its own MemoryStream for local events (tool calls, responses). The Hub has a separate stream for cross-agent delegation events.
- Observers (TokenMonitor, LoopDetector) run on individual agents for safety guardrails.
