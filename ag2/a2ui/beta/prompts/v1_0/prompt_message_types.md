Each message in the JSON array must be one of:
- `createSurface`: Initialize a new surface. Requires `surfaceId` and `catalogId` ("{catalog_id}").
- `updateComponents`: Add/update components on a surface. Requires `surfaceId` and `components` array.
- `updateDataModel`: Update the data model. Requires `surfaceId`, optional `path` and `value`.
- `deleteSurface`: Remove a surface. Requires `surfaceId`.
- `callRendererFunction`: Invoke a named renderer (client) function from the server. Requires `callRendererFunction` with `functionCallId` (a unique id the renderer copies verbatim into its response) and `callFunction` (with `@call`, `catalogId` and optional `args`).
- `agentFunctionResponse`: Reply to a function the renderer called on the agent. Requires `agentFunctionResponse` with the `functionCallId` of the call and exactly one of `value` or `error`.

All messages must include `"version": "{version_string}"`.

### Data Binding

To bind a component's value to the data model, use a **DataBinding object** with a `path` property — NOT a bare string.

Correct: `"value": {"path": "/customTime"}`
Wrong: `"value": "/customTime"`

This applies to ChoicePicker `value`, TextField `value`, Slider `value`, and any other component that reads from or writes to the data model. Button action context also uses this format: `"context": {"time": {"path": "/customTime"}}`.

### Function calls (v1.0, experimental)

To call a renderer function from the server, emit a `callRendererFunction` message; the renderer answers with a `rendererFunctionResponse` carrying the same `functionCallId`:

`{"version": "v1.0", "callRendererFunction": {"functionCallId": "fc-1", "callFunction": {"@call": "myFunction", "catalogId": "{catalog_id}", "args": {...}}}}`

When the renderer calls a function on you (a `callAgentFunction` request), answer with an `agentFunctionResponse` message carrying either a `value` or an `error`:

`{"version": "v1.0", "agentFunctionResponse": {"functionCallId": "fc-1", "value": {...}}}`
`{"version": "v1.0", "agentFunctionResponse": {"functionCallId": "fc-1", "error": {"code": "NOT_FOUND", "message": "..."}}}`
