# BaseEvent types

* ToolCall
* StreamToolCall
* ToolResult
* ToolError
* StreamToolResult

* MoveSpeaker

* ModelRequest
* ModelReasoning
* ModelResponse
* StreamModelResult

* HITL
* UserMessage

* ThreadStarted (conversation begin)
* ThreadComplete | ThreadError (conversation end)

* RunStarted (agent run begin)
* RunComplete | RunError (agent run end)

## History example


```
|- UserMessage                            # checkpoint
|- ThreadStarted
    |- MoveSpeaker("agent", ...)          # checkpoint
    |- RunStarted
        |- ModelRequest
            |- Thinking
            |- Thinking
            |- ToolCall   (builtin tool)
            |- ToolResult (builtin tool)
            |- Thinking
        |- ModelResult                    # checkpoint
        |- ToolCall("func", "arguments")
        |- ToolResult("func", "result")   # checkpoint
        |- ModelRequest
            |- Thinking
            |- Thinking
        |- ModelResult                    # checkpoint
    |- RunComplete | RunError             # checkpoint
    |- MoveSpeaker("agent2")              # checkpoint
    |- RunStarted
        |- ModelRequest
            |- Thinking
            |- Thinking
            |- StreamToolCall
            |- StreamToolCall
        |- ModelResult                    # checkpoint
        |- ToolCall("func", "arguments")
        |- ToolResult("func", "result")   # checkpoint
        |- HITL
        |- UserMessage
    |- RunComplete | RunError             # checkpoint
|- ThreadComplete | ThreadError
```
