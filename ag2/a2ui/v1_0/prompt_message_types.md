Each message in the JSON array must be one of:
- `createSurface`: Initialize a new surface. Requires `surfaceId` and `catalogId` ("{catalog_id}").
- `updateComponents`: Add/update components on a surface. Requires `surfaceId` and `components` array.
- `updateDataModel`: Update the data model. Requires `surfaceId`, optional `path` and `value`.
- `deleteSurface`: Remove a surface. Requires `surfaceId`.
- `callFunction`: Invoke a named client function from the server. Requires `callFunction` (with `call`) and `functionCallId` (a unique id the client copies verbatim into its response).
- `actionResponse`: Respond to a client-initiated action. Requires `actionResponse` and `actionId`; the `actionResponse` body must contain exactly one of `value` or `error`.

All messages must include `"version": "{version_string}"`.

### Data Binding

To bind a component's value to the data model, use a **DataBinding object** with a `path` property — NOT a bare string.

Correct: `"value": {"path": "/customTime"}`
Wrong: `"value": "/customTime"`

This applies to ChoicePicker `value`, TextField `value`, Slider `value`, and any other component that reads from or writes to the data model. Button action context also uses this format: `"context": {"time": {"path": "/customTime"}}`.

### Server-initiated function calls (v1.0)

To call a client function from the server, emit a `callFunction` message:

`{"version": "v1.0", "functionCallId": "fc-1", "callFunction": {"call": "myFunction", "args": {...}}}`

To respond to a client action, emit an `actionResponse` message carrying either a `value` or an `error`:

`{"version": "v1.0", "actionId": "act-1", "actionResponse": {"value": {...}}}`
`{"version": "v1.0", "actionId": "act-1", "actionResponse": {"error": {"code": "NOT_FOUND", "message": "..."}}}`
