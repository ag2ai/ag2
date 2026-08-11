Each message in the JSON array must be one of:
- `createSurface`: Initialize a new surface. Requires `surfaceId` and `catalogId` ("{catalog_id}").
- `updateComponents`: Add/update components on a surface. Requires `surfaceId` and `components` array.
- `updateDataModel`: Update the data model. Requires `surfaceId`, optional `path` and `value`.
- `deleteSurface`: Remove a surface. Requires `surfaceId`.

All messages must include `"version": "{version_string}"`.

### Data Binding

To bind a component's value to the data model, use a **DataBinding object** with a `path` property — NOT a bare string.

Correct: `"value": {"path": "/customTime"}`
Wrong: `"value": "/customTime"`

This applies to ChoicePicker `value`, TextField `value`, Slider `value`, and any other component that reads from or writes to the data model. Button action context also uses this format: `"context": {"time": {"path": "/customTime"}}`.
