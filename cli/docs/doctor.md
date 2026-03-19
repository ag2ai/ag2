# ag2 doctor

> AI-powered diagnostics for AG2 agent configurations and conversations.

## Problem

Multi-agent systems fail in subtle ways: infinite loops, wrong speaker selection,
tools that error silently, oversized system messages eating token budgets.
Debugging requires deep AG2 expertise. No framework offers automated diagnostics.

## Commands

```bash
# Analyze agent code for common issues
ag2 doctor my_team.py

# Diagnose a failed conversation log
ag2 doctor --log conversation_log.json

# AI-powered fix suggestions (generates diffs)
ag2 doctor my_team.py --fix

# Profile token usage and cost
ag2 doctor my_team.py --profile

# Check specific categories
ag2 doctor my_team.py --check termination,tools,config
```

## Diagnostic Categories

### 1. Static Analysis (no LLM needed)

These checks run instantly by analyzing the Python AST and AG2 objects:

**Termination**
- Agent has no termination condition → infinite loop risk
- `max_consecutive_auto_reply` is set too high or too low
- GroupChat `max_round` is missing or unreasonable

**Configuration**
- `LLMConfig` uses a model that doesn't support tool calling
- Temperature is 0 for creative tasks or 1.0 for structured output
- Multiple agents share the same system message
- System message exceeds 2000 tokens (cost warning)

**Tools**
- Tool registered for LLM but not for execution (or vice versa)
- Tool function missing type annotations (schema generation will fail)
- Tool has no description (LLM won't know when to use it)
- Too many tools registered (>20, LLM may get confused)

**Group Chat**
- `auto` speaker selection with >5 agents (consider explicit handoffs)
- No handoff conditions defined (agents can't route to each other)
- Circular handoff references
- Agent registered in group but never reachable via handoffs

**Code Execution**
- `LocalCommandlineCodeExecutor` without Docker (security warning)
- No timeout on code executor
- Code executor work directory doesn't exist

**Imports**
- Using deprecated `autogen` imports (suggest migration)
- Missing extras for used features (e.g., MCP without `ag2[mcp]`)

### 2. AI-Powered Analysis (requires API key)

These use an LLM to provide deeper insights:

**System Message Review**
```
Agent 'researcher' system message analysis:
  ⚠ Message is 1,847 tokens — consider condensing (target: <500)
  ⚠ Missing output format instructions — agent may produce inconsistent formats
  ✓ Clear role definition
  ✓ Tool usage instructions included
  Suggestion: Add "Always respond with structured JSON" for consistent parsing
```

**Conversation Log Diagnosis**
```bash
ag2 doctor --log conversation_log.json
```
```
Conversation Analysis (23 turns):
  ⚠ Turn 8: Agent 'researcher' made a tool call that returned an error,
    but continued without acknowledging the error. The final output
    is missing data that would have come from that tool.
  ⚠ Turn 12-18: Agents entered a repetitive loop — 'critic' kept asking
    for revisions and 'writer' made the same changes repeatedly.
    Suggestion: Add a revision counter to context variables, terminate after 2 rounds.
  ✗ Turn 23: Conversation hit max_round without producing a final answer.
    Suggestion: Add a 'summarizer' agent that activates when rounds > 80%.
```

**Cost Profile**
```bash
ag2 doctor my_team.py --profile
```
```
╭─ Cost Profile ─ research-team ─────────────────────╮
│                                                     │
│ Estimated cost per run: $0.42                       │
│                                                     │
│ Token breakdown by agent:                           │
│   researcher    1,200 in / 800 out    $0.03         │
│   critic        2,400 in / 600 out    $0.04         │
│   writer        8,000 in / 3,200 out  $0.18         │
│   summarizer    6,000 in / 1,500 out  $0.11         │
│   (system)      4,200 in              $0.06         │
│                                                     │
│ Bottleneck: 'writer' uses 43% of total tokens       │
│                                                     │
│ Recommendations:                                    │
│   1. Shorten writer's system message (2,100 tokens) │
│   2. Use structured output for writer to reduce     │
│      output tokens                                  │
│   3. Consider gpt-4o-mini for critic (low-stakes    │
│      review task)                                   │
╰─────────────────────────────────────────────────────╯
```

## `--fix` Mode

When `--fix` is passed, the doctor generates code patches:

```bash
ag2 doctor my_team.py --fix
```
```
Found 3 issues. Generating fixes...

Fix 1/3: Add termination condition to 'researcher'
─────────────────────────────────────────────────────
  researcher = AssistantAgent(
      name="researcher",
      system_message="...",
+     max_consecutive_auto_reply=3,
  )

Apply this fix? [Y/n]

Fix 2/3: Add error handling for web_search tool
─────────────────────────────────────────────────────
  @tool
  def web_search(query: str) -> str:
+     try:
          results = search_api(query)
+     except Exception as e:
+         return f"Search failed: {e}. Try rephrasing the query."
      return format_results(results)

Apply this fix? [Y/n]
```

## Output Format

```
╭─ AG2 Doctor ─ my_team.py ──────────────────────────╮
│ Analyzing 4 agents, 6 tools, 1 group chat...       │
╰─────────────────────────────────────────────────────╯

  🔴 Critical (1)
    Agent 'assistant' has LocalCommandlineCodeExecutor with no Docker
    isolation. User-provided code will execute with full system access.
    → Use DockerCommandlineCodeExecutor or add a timeout

  🟡 Warning (3)
    GroupChat uses 'auto' selection with 6 agents — LLM may select
    wrong speaker. Consider explicit handoffs or reduce to 3-4 agents.

    Agent 'writer' system message is 2,100 tokens. This costs ~$0.03
    per turn in context. Consider condensing.

    Tool 'fetch_data' has no description — LLM will guess when to use it.

  🟢 Good (5)
    All tools have type annotations ✓
    LLMConfig model supports tool calling ✓
    GroupChat has max_round set (10) ✓
    All agents have unique system messages ✓
    Handoff conditions cover all agents ✓

  Score: 6/9 checks passed (67%)
```

## Implementation Notes

### Static Analyzer
Use Python's `ast` module to parse the agent file and extract:
- Agent instantiations and their parameters
- Tool registrations
- GroupChat configurations
- LLMConfig settings

Alternatively, execute the file in a sandbox and inspect the live objects.

### AG2 Knowledge Base
The doctor's AI analysis uses the same knowledge from the skills pack
(agent-patterns, tool-registration, llm-config rules). This ensures
the doctor's advice is consistent with what the IDE skills teach.

### Profile Mode
Run the agent once with a standard prompt, capture all events via AG2's
event system, and compute token/cost metrics. Use `autogen.token_count_utils`
for counting.
