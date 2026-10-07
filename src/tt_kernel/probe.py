# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""The request `tt-model curl` sends, chosen by the task the served model performs.

Tasks use the Hugging Face ``pipeline_tag`` vocabulary, which the model card already uses,
so a caller that knows a model from elsewhere (tt-cli's catalog) can name it with ``--task``.
"""

from __future__ import annotations

import base64
import json
import struct
import zlib
from dataclasses import dataclass
from typing import Optional

from .runtime import DEFAULT_MAX_TOKENS, DEFAULT_PROMPT

TEXT_GENERATION = "text-generation"
IMAGE_TEXT_TO_TEXT = "image-text-to-text"
TEXT_TO_IMAGE = "text-to-image"
FEATURE_EXTRACTION = "feature-extraction"

# Tasks `curl` sends a real request for.
TASKS = (TEXT_GENERATION, IMAGE_TEXT_TO_TEXT, TEXT_TO_IMAGE, FEATURE_EXTRACTION)
# Tasks a served model can have that `curl` cannot exercise yet: it only checks the server is up.
HEALTH_ONLY_TASKS = (
    "automatic-speech-recognition",
    "text-to-speech",
    "text-to-video",
    "image-classification",
)

_PROMPTS = {
    TEXT_GENERATION: DEFAULT_PROMPT,
    IMAGE_TEXT_TO_TEXT: "Which colours are in this image, and where?",
    TEXT_TO_IMAGE: "A photo of an astronaut riding a horse on the moon",
    FEATURE_EXTRACTION: "Tenstorrent builds hardware for AI.",
}
TOOLS_PROMPT = "What is the weather in Paris right now?"

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name."}},
            "required": ["city"],
        },
    },
}

_TIMEOUT_S = 120
_IMAGE_TIMEOUT_S = 600  # a diffusion pipeline takes minutes on a cold server


@dataclass(frozen=True)
class Capabilities:
    """What the served model can do, and where that came from (for the status line)."""

    task: Optional[str]  # None: not knowable, so only a health check is possible
    tools: bool = False
    source: str = "default"


@dataclass(frozen=True)
class Request:
    path: str
    body: dict
    timeout: int = _TIMEOUT_S


@dataclass(frozen=True)
class Reply:
    text: str
    ok: bool = True  # False when the model answered, but not the way the request asked
    image: Optional[bytes] = None


def default_prompt(task: str, *, tools: bool) -> str:
    return TOOLS_PROMPT if tools and task == TEXT_GENERATION else _PROMPTS[task]


def build(task: str, model: str, prompt: str, *, tools: bool = False,
          params: Optional[dict] = None) -> Request:
    """The request for ``task``; ``params`` override or extend the body."""
    if task == TEXT_TO_IMAGE:
        path, body, timeout = "/v1/images/generations", {"prompt": prompt}, _IMAGE_TIMEOUT_S
    elif task == FEATURE_EXTRACTION:
        path, body, timeout = "/v1/embeddings", {"input": prompt}, _TIMEOUT_S
    else:
        content: object = prompt
        if task == IMAGE_TEXT_TO_TEXT:
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": sample_image_url()}},
            ]
        path, timeout = "/v1/chat/completions", _TIMEOUT_S
        body = {"messages": [{"role": "user", "content": content}],
                "max_tokens": DEFAULT_MAX_TOKENS}
        if tools:
            body["tools"] = [WEATHER_TOOL]
    return Request(path, {"model": model, **body, **(params or {})}, timeout)


def sample_image_url() -> str:
    """A 224x224 PNG, red above blue, as a data URL: small, and checkable in the answer."""
    size = 224
    top, bottom = b"\xff\x00\x00" * size, b"\x00\x00\xff" * size
    rows = b"".join(b"\x00" + (top if y < size // 2 else bottom) for y in range(size))

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data)
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(rows))
           + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


def read_reply(task: str, raw: str, *, tools: bool = False) -> Reply:
    """The server's answer as text for the terminal (plus image bytes, for an image).

    Raises ``ValueError`` with the server's own words when the reply is an error or not a
    shape this task produces, so the caller can show it instead of a traceback.
    """
    try:
        doc = json.loads(raw)
    except ValueError:
        raise ValueError(raw.strip() or "empty reply") from None
    if not isinstance(doc, dict):
        raise ValueError(raw.strip())
    error = _error_message(doc)
    if error:
        raise ValueError(error)
    try:
        if task == TEXT_TO_IMAGE:
            return _image_reply(doc)
        if task == FEATURE_EXTRACTION:
            vector = doc["data"][0]["embedding"]
            head = ", ".join(f"{v:.4f}" for v in vector[:4])
            return Reply(f"{len(vector)}-dimensional embedding: [{head}, ...]")
        message = doc["choices"][0]["message"]
        calls = message.get("tool_calls") or []
        lines = [f"tool call: {c['function']['name']}({c['function']['arguments']})"
                 for c in calls]
        content = (message.get("content") or message.get("reasoning_content")
                   or message.get("reasoning") or "")
        if content:
            lines.append(content.strip())
        return Reply("\n".join(lines), ok=bool(calls) or not tools)
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"unexpected {task} reply: {raw.strip()[:500]}") from exc


def _error_message(doc: dict) -> Optional[str]:
    """vLLM's ``{"error": {...}}`` / ``{"object": "error"}`` or FastAPI's ``{"detail": ...}``."""
    error = doc.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error)
    if error:
        return str(error)
    if doc.get("object") == "error":
        return str(doc.get("message") or doc)
    if "detail" in doc:
        return str(doc["detail"])
    return None


def _image_reply(doc: dict) -> Reply:
    # tt-inference-server's media server answers {"images": [b64]}; OpenAI answers
    # {"data": [{"b64_json": ...} | {"url": ...}]}.
    if doc.get("images"):
        return Reply("", image=base64.b64decode(doc["images"][0]))
    item = doc["data"][0]
    if item.get("b64_json"):
        return Reply("", image=base64.b64decode(item["b64_json"]))
    return Reply(f"image url: {item['url']}")


def image_suffix(data: bytes) -> str:
    return ".jpg" if data.startswith(b"\xff\xd8") else ".png"
