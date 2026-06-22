---
title: Multimodal Inputs
sidebarTitle: Inputs
---

# Multimodal Inputs

AG2 agents can process images, audio, video, and documents alongside text. The input event system provides a unified API across providers — you create inputs the same way regardless of which model you use.

## Input Types

| Factory Function | Creates | Description |
| :--- | :--- | :--- |
| `#!python ImageInput(...)` | Image input | JPEG, PNG, GIF, WebP |
| `#!python AudioInput(...)` | Audio input | WAV, MP3, OGG, FLAC, AAC |
| `#!python VideoInput(...)` | Video input | MP4, WebM, MOV, MKV, MPEG |
| `#!python DocumentInput(...)` | Document input | PDF, TXT, HTML, Markdown, CSV, JSON, Office formats |

Each factory function supports multiple ways to provide the data:

```python linenums="1"
from autogen.beta.events import ImageInput, AudioInput, VideoInput, DocumentInput

# From a URL
image = ImageInput("https://example.com/photo.jpg")

# From a local file path
image = ImageInput(path="photo.jpg")

# From raw bytes
image = ImageInput(data=raw_bytes, media_type="image/png")

# From a pre-uploaded file ID (provider-specific)
image = ImageInput(file_id="file-abc123")
```

---

## Using Inputs with Agents

Pass inputs directly to `#!python agent.ask()` as positional arguments alongside text:

```python linenums="1"
from autogen.beta import Agent
from autogen.beta.config import GeminiConfig
from autogen.beta.events import ImageInput

agent = Agent(
    "vision_agent",
    "You are a helpful assistant that describes images.",
    config=GeminiConfig(model="gemini-3-flash-preview"),
)

image = ImageInput("https://example.com/photo.jpg")
reply = await agent.ask("Describe this image in detail.", image)
print(reply.body)
```

You can pass multiple inputs in a single request:

```python linenums="1"
image1 = ImageInput("https://example.com/before.jpg")
image2 = ImageInput("https://example.com/after.jpg")

reply = await agent.ask("Compare these two images.", image1, image2)
```

---

## Provider Support

Not all providers support all input types. The table below shows what each provider accepts:

| Input Type | OpenAI | OpenAI Responses | Gemini | Anthropic | xAI | Bedrock |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Text** | Yes | Yes | Yes | Yes | Yes | Yes |
| **Image (URL)** | Yes | Yes | Yes | Yes | Yes | - |
| **Image (binary)** | Yes | Yes | Yes | Yes | Yes | Yes |
| **Audio (URL)** | - | - | Yes | - | - | - |
| **Audio (binary)** | Yes | - | Yes | - | - | - |
| **Video (URL)** | - | - | Yes | - | - | - |
| **Video (binary)** | - | - | Yes | - | - | Yes |
| **Document (URL)** | - | Yes | Yes | Yes | Yes | - |
| **Document (binary)** | - | - | Yes | Yes | Yes | Yes |
| **File ID** | - | Yes | - | Yes | Yes | - |

If you pass an unsupported input type to a provider, an `#!python UnsupportedInputError` is raised with a clear message indicating what is not supported and by which provider.

---

## Provider-Specific Details

### Gemini

Gemini has the broadest multimodal support — it accepts images, audio, video, and documents in all forms (URL, binary, and local file path).

**YouTube URLs** are supported directly:

```python linenums="1"
from autogen.beta.events import VideoInput

video = VideoInput("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
reply = await agent.ask("Summarize this video.", video)
```

**Google Files API** — for large files (>20MB), upload via the Google Files API first and pass the returned URI:

```python linenums="1"
from google import genai
from autogen.beta.events import VideoInput

client = genai.Client()
uploaded = client.files.upload(file="large_video.mp4")

# Wait for processing to complete
import time
while uploaded.state.name == "PROCESSING":
    time.sleep(2)
    uploaded = client.files.get(name=uploaded.name)

video = VideoInput(uploaded.uri)
reply = await agent.ask("Describe this video.", video)
```

#### Vendor Metadata

Gemini supports provider-specific settings via `#!python vendor_metadata` on binary inputs. These map to Gemini Part fields:

| Key | Type | Description |
| :--- | :--- | :--- |
| `media_resolution` | `str` | Controls token allocation per image/video frame |
| `video_metadata` | `dict` | Video clipping (`start_offset`, `end_offset`) and frame rate (`fps`) |
| `display_name` | `str` | Display name for the file |

**Media resolution** — control quality vs cost tradeoff for images and video frames:

```python linenums="1"
from autogen.beta.events import ImageInput

# Lower resolution = fewer tokens = lower cost
image = ImageInput(
    data=raw_bytes,
    media_type="image/jpeg",
    vendor_metadata={"media_resolution": "MEDIA_RESOLUTION_LOW"},
)
```

Available values: `MEDIA_RESOLUTION_LOW`, `MEDIA_RESOLUTION_MEDIUM`, `MEDIA_RESOLUTION_HIGH`, `MEDIA_RESOLUTION_ULTRA_HIGH`.

**Video clipping and frame rate** — process only a portion of a video or adjust the sampling rate:

```python linenums="1"
from autogen.beta.events import VideoInput

video = VideoInput(
    path="lecture.mp4",
    vendor_metadata={
        "video_metadata": {
            "start_offset": "60s",
            "end_offset": "120s",
            "fps": 0.5,
        },
    },
)
reply = await agent.ask("Summarize this section of the video.", video)
```

**Display name** — attach a name to the file for reference:

```python linenums="1"
from autogen.beta.events import DocumentInput

doc = DocumentInput(
    path="report.pdf",
    vendor_metadata={"display_name": "Q4 Financial Report"},
)
```

### OpenAI

OpenAI supports images via both the Completions and Responses APIs. Audio binary input (WAV, MP3) is supported in the Completions API. The Responses API additionally supports file IDs and document URLs.

#### Vendor Metadata

OpenAI supports `#!python vendor_metadata` for image detail control:

```python linenums="1"
from autogen.beta.events import ImageInput

image = ImageInput(
    data=raw_bytes,
    media_type="image/png",
    vendor_metadata={"detail": "low"},  # "low", "high", or "auto"
)
```

### Anthropic

Anthropic supports images (JPEG, PNG, GIF, WebP) and documents (PDF) via URL, base64, or File ID. Audio and video are not supported.

**File ID** — upload files via the Anthropic Files API (beta) and reference by ID:

```python linenums="1"
import anthropic
from autogen.beta.events import ImageInput, DocumentInput

client = anthropic.Anthropic()

# Upload an image
uploaded = client.beta.files.upload(
    file=("photo.jpg", open("photo.jpg", "rb"), "image/jpeg"),
)

# Reference by file_id — filename determines block type (image vs document)
image = ImageInput(file_id=uploaded.id, filename="photo.jpg")
reply = await agent.ask("Describe this image.", image)
```

#### Vendor Metadata

Anthropic supports `#!python vendor_metadata` for prompt caching on content blocks:

```python linenums="1"
from autogen.beta.events import DocumentInput

doc = DocumentInput(
    path="report.pdf",
    vendor_metadata={"cache_control": {"type": "ephemeral"{{ "}}" }},
)
```

### xAI

xAI supports images (URL and binary), documents (URL and binary), and pre-uploaded file IDs. Audio and video are not currently supported — passing them raises `#!python UnsupportedInputError`.

**File ID** — reference a file previously uploaded via the xAI Files API:

```python linenums="1"
from autogen.beta.events import ImageInput, DocumentInput

image = ImageInput(file_id="file-abc123", filename="photo.jpg")
doc = DocumentInput(file_id="file-xyz789", filename="report.pdf")
```

#### Vendor Metadata

xAI reads `detail` for image quality control from two **different** attributes depending on the input source — `#!python vendor_metadata` for binary, `#!python metadata` for URL. Mixing them up means the value is silently ignored and xAI falls back to `"auto"`.

**Binary image** — set `detail` via `#!python vendor_metadata`:

```python linenums="1"
from autogen.beta.events import BinaryInput, BinaryType

image = BinaryInput(
    raw_bytes,
    media_type="image/png",
    kind=BinaryType.IMAGE,
    vendor_metadata={"detail": "low"},  # "low", "high", or "auto"
)
```

**URL image** — set `detail` via `#!python metadata` (not `vendor_metadata`):

```python linenums="1"
from autogen.beta.events import UrlInput, BinaryType

image = UrlInput(
    "https://example.com/photo.jpg",
    kind=BinaryType.IMAGE,
    metadata={"detail": "low"},
)
```

!!! note
    The factory `#!python ImageInput(url=...)` does not forward `metadata`. To configure `detail` on a URL image, construct `#!python UrlInput` directly as shown above.

**Document filename** — xAI requires a filename for binary documents. When sending raw bytes, either provide one via `#!python vendor_metadata={"filename": ...}`, or rely on the auto-derived fallback (`file.<subtype>` from the media type, e.g. `file.pdf` for `application/pdf`):

```python linenums="1"
from autogen.beta.events import BinaryInput, BinaryType

doc = BinaryInput(
    pdf_bytes,
    media_type="application/pdf",
    kind=BinaryType.DOCUMENT,
    vendor_metadata={"filename": "Q4-report.pdf"},
)
```

### Amazon Bedrock

The Bedrock Converse API accepts **binary sources only** — images (JPEG, PNG, GIF, WebP), documents (PDF, CSV, DOC, DOCX, XLS, XLSX, HTML, TXT, Markdown), and video (MP4, WebM, MOV, MKV, and more; Amazon Nova models). URL inputs and file IDs raise `#!python UnsupportedInputError` — Bedrock has no Files API, so source data from a URL must be downloaded and passed as bytes:

```python linenums="1"
from autogen.beta import Agent
from autogen.beta.config import BedrockConfig
from autogen.beta.events import DocumentInput, ImageInput

agent = Agent(
    "vision_agent",
    "You describe images and summarize documents.",
    config=BedrockConfig(model="us.amazon.nova-lite-v1:0", region_name="us-east-1"),
)

image = ImageInput(path="photo.jpg")
doc = DocumentInput(data=pdf_bytes, media_type="application/pdf")
reply = await agent.ask("Describe the image and summarize the document.", image, doc)
```

!!! note
    Modality support also depends on the **model** behind the Converse API: Amazon Nova models accept images, documents, and video; many others (e.g. DeepSeek) are text-only and return a `ValidationException` from AWS for non-text blocks. The provider raises `#!python UnsupportedInputError` only for inputs the Converse API itself cannot carry.

**Document name** — Converse requires a name for document blocks. It is taken from `#!python vendor_metadata={"filename": ...}` (set automatically when using `path=`), sanitized to the characters Converse allows (alphanumerics, single spaces, hyphens, parentheses, brackets), and falls back to `"document"` when absent:

```python linenums="1"
from autogen.beta.events import BinaryInput, BinaryType

doc = BinaryInput(
    pdf_bytes,
    media_type="application/pdf",
    kind=BinaryType.DOCUMENT,
    vendor_metadata={"filename": "Q4 report.pdf"},
)
```
