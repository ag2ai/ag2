Each message in the JSON array must be one of:
- `createSurface`: Initialize a new surface. Requires `surfaceId`. `catalogId` ("{catalog_id}") sets the surface's default catalog. It may also carry the initial `components` array and `dataModel` object, so a whole UI can be sent in this single message.
- `updateComponents`: Add/update components on a surface. Requires `surfaceId` and `components` array.
- `updateDataModel`: Update the data model. Requires `surfaceId` and `value`; `path` is optional. Set `value` to `null` to delete the key at `path`.
- `deleteSurface`: Remove a surface. Requires `surfaceId`.
- `callRendererFunction`: Ask the renderer to run a catalog function. Requires `functionCallId` (a unique id the renderer copies verbatim into its response) and `callFunction` (with `@call`, the function name, and `catalogId`; `args` is optional). Only functions the catalog allows an agent to call can be used.

All messages must include `"version": "{version_string}"`.

Do not emit `agentFunctionResponse`: the server sends it itself when the renderer calls a server function.

### Data Binding and Function Calls

To bind a component's value to the data model, use a **DataBinding object** with an `@path` property — NOT a bare string and NOT a plain `path` key.

Correct: `"value": {"@path": "/customTime"}`
Wrong: `"value": "/customTime"`, `"value": {"path": "/customTime"}`

This applies to ChoicePicker `value`, TextField `value`, Slider `value`, and any other component that reads from or writes to the data model. Button action context also uses this format: `"context": {"time": {"@path": "/customTime"}}`.

To call a catalog function inside a property, use a **FunctionCall object** with an `@call` property: `{"@call": "formatString", "args": {"value": "Hello ${/user/name}"}}`.

Two places keep the plain `path` spelling: a list template, `"children": {"componentId": "itemTemplate", "path": "/items"}`, and the `path` of an `updateDataModel` message.

### Calling a renderer function (v1.0)

To run a function on the renderer, emit a `callRendererFunction` message; the renderer replies with the result, and that reply reaches you as your next input:

`{"version": "v1.0", "callRendererFunction": {"functionCallId": "fc-1", "callFunction": {"@call": "myFunction", "catalogId": "{catalog_id}", "args": {...}}}}`
