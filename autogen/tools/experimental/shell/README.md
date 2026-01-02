---
title: Shell Tool
description: Execute shell commands securely with GPT-5.1 using sandboxing and validation
---

# Shell Tool

The `ShellTool` enables agents to execute shell commands safely through OpenAI's Responses API with GPT-5.1. It provides multiple layers of security including command filtering, path restrictions, to prevent dangerous operations while allowing legitimate system interactions.

## Features

- **Secure Command Execution**: Multi-layer security with pattern filtering, whitelist/blacklist, and path restrictions
- **Workspace Isolation**: Execute commands within a restricted working directory
- **Command Validation**: Block dangerous commands and patterns by default
- **Path Restrictions**: Control file system access with glob patterns
- **Production Ready**: Designed for safe use in production environments

## Requirements

- Python >= 3.10
- GPT-5.1 access (currently in beta)
- OpenAI API key
- AG2 installed with OpenAI support

## Installation

Install AG2 with OpenAI support:

pip install ag2[openai]For more information, refer to the [installation guide](/docs/user-guide/basic-concepts/installing-ag2).

## Quick Start

The simplest way to use the shell tool is through the LLM configuration:
thon
import os
from autogen import ConversableAgent, LLMConfig
from dotenv import load_dotenv

load_dotenv()

# Configure LLM with shell tool
```python
llm_config = LLMConfig(
    config_list={
        "api_type": "responses",
        "model": "gpt-5.1",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "built_in_tools": ["shell"],
    },
)

# Create agent - tool is automatically registered
system_agent = ConversableAgent(
    name="system_assistant",
    llm_config=llm_config,
    system_message="""You are a helpful system assistant. You can execute shell commands
    to help with system tasks. Always use safe commands and explain what you're doing.""",
)

# Use the agent
result = system_agent.initiate_chat(
    recipient=system_agent,
    message="List all Python files in the current directory",
    max_turns=2,
)## Security Model
```

The Shell Tool implements a **defense-in-depth** security model with multiple layers:

### 1. Command Pattern Filtering

Blocks dangerous command patterns by default, including:
- Root filesystem deletion (`rm -rf /`)
- Disk device operations (`dd of=/dev/sda`)
- Filesystem formatting (`mkfs`, `format`)
- Fork bombs and other denial-of-service attacks
- System directory deletion

**Default Patterns**: The tool includes a comprehensive set of dangerous patterns in `DEFAULT_DANGEROUS_PATTERNS`. These patterns are checked using regex matching against the full command string.

### 2. Command Whitelist/Blacklist

- **Whitelist (`allowed_commands`)**: If provided, only commands in this list can be executed
- **Blacklist (`denied_commands`)**: Commands in this list are always blocked, even if they pass other checks

**Important**: When both whitelist and blacklist are configured, the whitelist check happens first. If a command is not in the whitelist, it will be rejected before the blacklist is checked.

### 3. Working Directory Restriction

All commands execute within a specified `workspace_dir`. This provides chroot-like isolation, preventing commands from accessing files outside the workspace.

### 4. Path Access Control

The `allowed_paths` parameter controls which file system paths can be accessed:

- **Pattern Matching**: Uses glob-style patterns (e.g., `src/**`, `*.py`)
- **Workspace Relative**: All paths are validated relative to `workspace_dir`
- **Command Path Validation**: Extracts and validates paths from commands

**Important Limitations**: Path validation in commands uses regex pattern `r"(?:^|\s)([/~]|\.\./)[^\s]*"` which only catches:
- Absolute paths starting with `/`
- Home directory paths starting with `~`
- Relative paths starting with `../`

**Paths that may bypass validation**:
- Simple relative paths: `cat file.txt` (doesn't start with `/`, `~`, or `../`)
- Quoted paths: `cat "/some/path"` (quotes may interfere with regex)
- Environment variables: `cat $HOME/file` (variable expansion happens after validation)

**Recommendation**: Use `allowed_paths` patterns to restrict access at the workspace level, and rely on command whitelisting for additional security.

## Configuration Options

### Workspace Directory

Specify a dedicated workspace directory for command execution:
``` python
llm_config = LLMConfig(
    config_list={
        "api_type": "responses",
        "model": "gpt-5.1",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "built_in_tools": ["shell"],
        "workspace_dir": "./sandbox",  # Commands execute here
    },
)
```

If the directory doesn't exist, it will be created automatically.

### Allowed Paths

Control which paths can be accessed within the workspace:
```python
llm_config = LLMConfig(
    config_list={
        "api_type": "responses",
        "model": "gpt-5.1",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "built_in_tools": ["shell"],
        "workspace_dir": "./sandbox",
        "allowed_paths": ["src/**", "data/*.json"],  # Only these paths
    },
)
```

**Path Pattern Examples**:
- `["**"]` - Allow all paths within workspace (default)
- `["src/**"]` - Allow all files in `src/` and subdirectories
- `["*.py", "*.json"]` - Allow Python and JSON files in root
- `["src/**", "tests/**"]` - Allow paths in multiple directories

**Note**: The `allowed_paths` parameter uses glob-style pattern matching for filesystem paths. This is separate from the command path validation regex, which has different limitations.

### Command Restrictions

#### Whitelist (Allowed Commands)

Only allow specific commands:
```python
llm_config = LLMConfig(
    config_list={
        "api_type": "responses",
        "model": "gpt-5.1",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "built_in_tools": ["shell"],
        "allowed_commands": ["ls", "cat", "grep", "find"],  # Only these commands
    },
)
```

#### Blacklist (Denied Commands)

Block specific commands:
```python
llm_config = LLMConfig(
    config_list={
        "api_type": "responses",
        "model": "gpt-5.1",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "built_in_tools": ["shell"],
        "denied_commands": ["rm", "dd", "format"],  # Block these commands
    },
)
```

### Custom Dangerous Patterns

Override default dangerous patterns:
```python
from autogen.tools.experimental.shell.shell_tool import ShellExecutor

custom_patterns = [
    (r"\bmy_dangerous_command\b", "Custom dangerous command is not allowed."),
]

# Note: This requires direct ShellExecutor usage
# The shell tool in LLM config uses default patterns### Disable Command Filtering

Disable pattern-based filtering (not recommended for production):

# This is only available when using ShellExecutor directly
executor = ShellExecutor(
    enable_command_filtering=False,  # Disable pattern checks
)## Production Configuration
```

For production use, follow these security best practices:

### Recommended Production Setup
```python
llm_config = LLMConfig(
    config_list={
        "api_type": "responses",
        "model": "gpt-5.1",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "built_in_tools": ["shell"],
        "workspace_dir": "/var/sandbox/user_workspace",  # Isolated directory
        "allowed_paths": ["**"],  # Or restrict to specific paths
        "allowed_commands": [  # Whitelist approach (most secure)
            "ls", "cat", "grep", "find", "head", "tail",
            "wc", "sort", "uniq", "cut", "awk", "sed"
        ],
        "denied_commands": ["rm", "dd", "format", "mkfs"],  # Extra safety
    },
)
```

### Security Checklist

- ✅ Use a dedicated, isolated `workspace_dir` outside system directories
- ✅ Implement command whitelisting (`allowed_commands`) for maximum security
- ✅ Set restrictive `allowed_paths` patterns when possible
- ✅ Keep `enable_command_filtering=True` (default) to use dangerous pattern checks
- ✅ Monitor command execution logs
- ✅ Regularly review and update dangerous patterns as needed

## Executor Reuse Behavior

**Important**: The `ShellExecutor` instance is **reused across multiple calls** within the same `OpenAIResponsesClient` instance. This means:

1. **Settings Persist**: Configuration changes (workspace_dir, allowed_paths, etc.) persist across calls
2. **Shared State**: The executor maintains state between calls
3. **Performance**: Reusing the executor is more efficient than creating new instances

**Implications**:
- If different calls require different security settings, they will share the same executor configuration
- The last call's settings may override previous settings
- Consider creating separate client instances if you need isolated executor configurations


### Example

Process data files:
```python
llm_config = LLMConfig(
    config_list={
        "api_type": "responses",
        "model": "gpt-5.1",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "built_in_tools": ["shell"],
        "workspace_dir": "./data_processing",
        "allowed_paths": ["data/**", "output/**"],
        "allowed_commands": ["cat", "grep", "awk", "sort", "uniq"],
    },
)

result = agent.run(
    recipient=agent,
    message="Count unique values in column 3 of data.csv",
    max_turns=2,
)
```

### Command Structure

Commands are executed through the `shell_call` action:
```python
{
  "type": "shell_call",
  "call_id": "call_123",
  "action": {
    "commands": ["ls -la", "cat file.txt"],
    "timeout_ms": 5000,
    "max_output_length": 10000
  }
}### Output Format

Command execution returns `ShellCallOutput`:
on
{
  "call_id": "call_123",
  "type": "shell_call_output",
  "max_output_length": 10000,
  "output": [
    {
      "stdout": "command output",
      "stderr": "error output",
      "outcome": {
        "type": "exit",  # or "timeout"
        "exit_code": 0
      }
    }
  ]
}
```

### Multiple Commands

Multiple commands are executed sequentially. Each command's output is captured separately:

# Commands execute one after another
```python
action = {
    "commands": ["cd /tmp", "ls -la", "pwd"],
    "timeout_ms": 10000
}
```

## Troubleshooting

### Command Blocked by Pattern Filter

**Error**: `Potentially dangerous command detected: ...`

**Solution**: The command matches a dangerous pattern. Either:
- Use a different command that doesn't match dangerous patterns
- Disable filtering (not recommended) or customize patterns
- Use allowed_commands whitelist to bypass pattern checks for specific commands

### Command Not in Whitelist

**Error**: `Command 'X' is not in the allowed commands list`

**Solution**: Add the command to `allowed_commands` or remove the whitelist restriction.

### Path Access Denied

**Error**: `Access to path 'X' is not allowed`

**Solution**:
- Add the path pattern to `allowed_paths`
- Use `["**"]` to allow all paths (less secure)
- Note: Some paths may bypass validation (see Path Access Control section)

## API Reference

### ShellExecutor

The core executor class for shell command execution.

**Parameters**:
- `workspace_dir` (str | Path | None): Working directory (default: current directory)
- `allowed_paths` (list[str] | None): Allowed path patterns (default: ["**"])
- `allowed_commands` (list[str] | None): Whitelist of allowed commands (default: None)
- `denied_commands` (list[str] | None): Blacklist of denied commands (default: [])
- `enable_command_filtering` (bool): Enable pattern-based filtering (default: True)
- `dangerous_patterns` (list[tuple[str, str]] | None): Custom dangerous patterns (default: DEFAULT_DANGEROUS_PATTERNS)

**Methods**:
- `run(cmd: str, timeout: float | None = None) -> CmdResult`: Execute a single command
- `run_commands(commands: list[str], timeout_ms: int | None = None) -> list[ShellCommandOutput]`: Execute multiple commands

### CmdResult

Result of executing a shell command.

**Attributes**:
- `stdout` (str): Standard output
- `stderr` (str): Standard error
- `exit_code` (int | None): Exit code
- `timed_out` (bool): Whether command timed out

## Limitations and Known Issues

1. **Path Validation Regex**: The path validation regex has limitations (see Path Access Control section)
2. **Executor Reuse**: Executor settings persist across calls (see Executor Reuse Behavior)
3. **Timeout Validation**: Timeout values are not validated for positivity
4. **Command Parsing**: Simple command parsing may not handle all shell syntax correctly
5. **Platform Differences**: Some commands may behave differently on Windows vs Unix systems

## Contributing

When contributing to the shell tool:

1. Add new dangerous patterns to `DEFAULT_DANGEROUS_PATTERNS` as needed
2. Test path validation with various path formats
3. Consider cross-platform compatibility
4. Document any new security considerations
5. Add tests for new features

## See Also

- [Apply Patch Tool](../apply_patch/README.md) - For file operations
- [OpenAI Responses API Documentation](https://platform.openai.com/docs/api-reference/responses)
- [AG2 Tools Documentation](/docs/user-guide/advanced-concepts/tools)
