---
name: ag2-multimodal-input
description: Send images, audio, video, or documents into an AG2 beta `Agent` alongside text. Pass `ImageInput`, `AudioInput`, `VideoInput`, or `DocumentInput` as positional args to `agent.ask(...)`. Use when the user wants the agent to process non-text input — describe a photo, transcribe audio, summarise a PDF, analyse a video. Covers per-provider support matrix, the four ways to source data (URL / path / bytes / file_id), Gemini-specific YouTube + media-resolution + clipping, OpenAI image-detail, Anthropic prompt-caching on attachments, and `FilesAPI` for upload lifecycle.
license: Apache-2.0
---

# Multimodal inputs

## When to use

The user wants the agent to process non-text input: an image to describe, audio to transcribe, video to summarise, or a PDF / document to extract from. The same factory pattern works across providers; per-provider support varies.

## 60-second recipe

```python
from autogen.beta import Agent
from autogen.beta.config import GeminiConfig
from autogen.beta.events import ImageInput

agent = Agent(
    "vision",
    "You describe images.",
    config=GeminiConfig(model="gemini-3-flash-preview"),
)

image = ImageInput("https://example.com/photo.jpg")
reply = await agent.ask("Describe this image in detail.", image)
print(reply.body)
```

Multiple inputs in one ask are fine:

```python
reply = await agent.ask(
    "Compare these two images.",
    ImageInput("https://example.com/before.jpg"),
    ImageInput("https://example.com/after.jpg"),
)
```

## Input factories

| Factory | Formats |
|---|---|
| `ImageInput(...)` | JPEG, PNG, GIF, WebP |
| `AudioInput(...)` | WAV, MP3, OGG, FLAC, AAC |
| `VideoInput(...)` | MP4, WebM, MOV, MKV, MPEG |
| `DocumentInput(...)` | PDF, TXT, HTML, Markdown, CSV, JSON, Office formats |

Each accepts the same four data sources:

```python
from autogen.beta.events import ImageInput

ImageInput("https://example.com/photo.jpg")     # URL
ImageInput(path="photo.jpg")                    # local file
ImageInput(data=raw_bytes, media_type="image/png")  # bytes
ImageInput(file_id="file-abc123")               # provider-uploaded
```

## Provider matrix

| Input type | OpenAI | OpenAI Responses | Gemini | Anthropic |
|---|:---:|:---:|:---:|:---:|
| Text | ✓ | ✓ | ✓ | ✓ |
| Image (URL) | ✓ | ✓ | ✓ | ✓ |
| Image (binary) | ✓ | ✓ | ✓ | ✓ |
| Audio (URL) | – | – | ✓ | – |
| Audio (binary) | ✓ | – | ✓ | – |
| Video (URL) | – | – | ✓ | – |
| Video (binary) | – | – | ✓ | – |
| Document (URL) | – | ✓ | ✓ | ✓ |
| Document (binary) | – | – | ✓ | ✓ |
| File ID | – | ✓ | – | ✓ |

Unsupported combinations raise `UnsupportedInputError` with a clear message.

**Gemini has the broadest multimodal support.** If you don't know which provider to pick for a multimodal task, start there.

## Provider-specific niceties

### Gemini — YouTube URLs work directly

```python
from autogen.beta.events import VideoInput

video = VideoInput("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
reply = await agent.ask("Summarize this video.", video)
```

### Gemini — large files (> 20MB) via Google Files API

```python
from google import genai
from autogen.beta.events import VideoInput
import time

client = genai.Client()
uploaded = client.files.upload(file="large_video.mp4")
while uploaded.state.name == "PROCESSING":
    time.sleep(2)
    uploaded = client.files.get(name=uploaded.name)

video = VideoInput(uploaded.uri)
```

### Gemini — `vendor_metadata`

| Key | Purpose |
|---|---|
| `media_resolution` | `MEDIA_RESOLUTION_LOW/MEDIUM/HIGH/ULTRA_HIGH` — token vs cost |
| `video_metadata` | Clipping (`start_offset`, `end_offset`) and `fps` |
| `display_name` | Display name for the file |

```python
ImageInput(data=raw, media_type="image/jpeg", vendor_metadata={"media_resolution": "MEDIA_RESOLUTION_LOW"})

VideoInput(path="lecture.mp4", vendor_metadata={
    "video_metadata": {"start_offset": "60s", "end_offset": "120s", "fps": 0.5},
})
```

### OpenAI — image detail

```python
ImageInput(data=raw, media_type="image/png", vendor_metadata={"detail": "low"})  # "low" | "high" | "auto"
```

### Anthropic — File ID + prompt caching

```python
import anthropic
from autogen.beta.events import ImageInput, DocumentInput

client = anthropic.Anthropic()
uploaded = client.beta.files.upload(file=("photo.jpg", open("photo.jpg", "rb"), "image/jpeg"))

# filename determines block type (image vs document)
image = ImageInput(file_id=uploaded.id, filename="photo.jpg")

# Cache an attachment so subsequent turns skip re-uploading
doc = DocumentInput(path="report.pdf", vendor_metadata={"cache_control": {"type": "ephemeral"}})
```

## `FilesAPI` — upload lifecycle, provider-agnostic

For any provider that has a file API (`OpenAIConfig`, `OpenAIResponsesConfig`, `AnthropicConfig`, `GeminiConfig`):

```python
from autogen.beta import FilesAPI
from autogen.beta.config import OpenAIResponsesConfig

files = FilesAPI(OpenAIResponsesConfig(model="gpt-5-mini"))

uploaded = await files.upload(path="report.pdf", purpose="assistants")
print(uploaded.file_id)

# Or from bytes (filename required)
uploaded = await files.upload(data=b"...", filename="hello.txt", purpose="assistants")

# List, read, delete
all_files = await files.list()
data = await files.read(uploaded.file_id)        # NotImplementedError on Gemini
await files.delete(uploaded.file_id)
```

Pass the `file_id` to `DocumentInput`, `ImageInput`, etc.:

```python
from autogen.beta.events import DocumentInput

doc = DocumentInput(file_id=uploaded.file_id)
reply = await agent.ask("Summarize this report.", doc)
```

## Going deeper

- `website/docs/beta/inputs/inputs.mdx` — full provider matrix and `vendor_metadata` reference.
- `website/docs/beta/advanced/files.mdx` — `FilesAPI` reference (upload / list / read / delete).
- For tools that **return** images / binary back to the LLM, see `ag2-add-custom-tool` (`ImageInput`, `BinaryInput`, `ToolResult`).

## Common pitfalls

- **Picking a provider that doesn't support your input type** — silently you'll get `UnsupportedInputError`. Check the matrix; Gemini is broadest.
- **`FilesAPI.read()` on Gemini** — raises `NotImplementedError`. Gemini doesn't expose download.
- **Calling `files.upload(data=...)` without `filename=`** — raises `ValueError`. Filename is required for in-memory uploads.
- **Providing `path=` and `data=` to the same factory** — pick one source. Same for `file_id=`.
- **Anthropic `ImageInput(file_id=...)` without `filename=`** — Anthropic decides block type (image vs document) by filename extension. Pass it.
- **Gemini `vendor_metadata` keys are nested** — `video_metadata` itself takes a dict. Check the doc table for shape.
- **Forgetting to wait for Gemini file processing** — large uploads have a `PROCESSING` state. Poll `client.files.get(name=...)` until ready before referencing the URI.
