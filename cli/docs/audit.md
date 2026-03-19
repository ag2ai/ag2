# ag2 audit

> Security and safety scanning for multi-agent systems.

## Problem

Agents execute code, call APIs, and access databases. A misconfigured agent
can be a serious security risk — prompt injection, code execution without
sandboxing, credential leakage, unvalidated tool inputs. No framework offers
automated security scanning.

## Commands

```bash
# Scan agent configuration for security issues
ag2 audit my_team.py

# Generate a formatted safety report
ag2 audit my_team.py --report audit_report.html

# Check specific categories only
ag2 audit my_team.py --check execution,credentials,injection

# Scan an entire project
ag2 audit ./my-project/

# CI-friendly — exit code 1 if critical issues found
ag2 audit my_team.py --strict
```

## Security Checks

### Code Execution Safety
- `LocalCommandlineCodeExecutor` with no Docker → **critical**
- Code executor with no timeout → **warning**
- Code executor work directory is `/tmp` or world-writable → **warning**
- Code executor allows arbitrary shell commands → **critical**

### Credential Safety
- API keys hardcoded in source files → **critical**
- `.env` file in git-tracked directory without `.gitignore` entry → **critical**
- LLMConfig with `api_key` parameter (should use env var) → **warning**
- Credentials passed in tool arguments visible in logs → **warning**

### Prompt Injection
- Agent accepts unvalidated user input as system message → **critical**
- Tool outputs are inserted into prompts without sanitization → **warning**
- No guardrails configured on user-facing agents → **warning**

### Tool Safety
- Tools with file system write access and no path validation → **warning**
- Tools that execute shell commands from LLM-generated input → **critical**
- Database tools with write access on production connection → **warning**
- HTTP tools that follow redirects to internal networks → **warning**

### Configuration
- Agent has no `max_consecutive_auto_reply` (potential runaway) → **warning**
- GroupChat `max_round` > 50 without cost controls → **info**
- Multiple agents sharing credentials → **info**

## Output

```
╭─ AG2 Audit ─ my_team.py ──────────────────────────╮
│ Scanning 4 agents, 6 tools, 1 group chat...       │
╰────────────────────────────────────────────────────╯

  🔴 CRITICAL (2)

    [SEC-001] LocalCommandlineCodeExecutor without sandbox
    File: my_team.py:45
    Agent 'coder' uses LocalCommandlineCodeExecutor. LLM-generated
    code executes with full system privileges.
    Fix: Use DockerCommandlineCodeExecutor or add PythonCodeExecutionTool

    [SEC-002] API key hardcoded in source
    File: my_team.py:12
    OpenAI API key found in LLMConfig constructor.
    Fix: Use OPENAI_API_KEY environment variable

  🟡 WARNING (3)

    [SEC-003] No guardrails on user-facing agent
    [SEC-004] Code executor has no timeout (default: unlimited)
    [SEC-005] Tool 'write_file' accepts arbitrary paths

  🟢 PASSED (4)

    Tools have input validation ✓
    No credential leakage in tool outputs ✓
    GroupChat has bounded max_round ✓
    All agents have termination conditions ✓

╭─ Summary ──────────────────────────────────────────╮
│ Score: 4/9 checks passed                           │
│ Critical: 2 | Warning: 3 | Passed: 4              │
│                                                    │
│ Fix critical issues before deploying to production │
╰────────────────────────────────────────────────────╯
```

## CI Integration

```yaml
# .github/workflows/agent-security.yml
- name: Audit agent security
  run: ag2 audit ./agents/ --strict
  # Exit code 1 if any CRITICAL issues found
```

## Implementation Notes

### Static Analysis
Use Python AST to detect:
- `LocalCommandlineCodeExecutor` instantiations
- String literals matching API key patterns (`sk-...`, `AKIA...`)
- Tool functions with `subprocess`, `os.system`, `eval`, `exec`
- File operations without path validation

### Runtime Analysis (optional)
With `--deep`, actually import the agent file and inspect live objects:
- Check if executors have timeout set
- Verify guardrails are registered
- Test tool input validation with fuzzy inputs

### SARIF Output
Support `--format sarif` for integration with GitHub Advanced Security
and other SAST platforms.
