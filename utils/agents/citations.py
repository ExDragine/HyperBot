"""Turn model-provided web annotations into numbered reply references."""

from collections.abc import Iterator, Mapping
from typing import Any
from urllib.parse import urlsplit

from utils.agent_protocol import WebCitation

_MAX_CITATIONS = 8


def _text_blocks(content: Any) -> Iterator[tuple[str, Any]]:
    if isinstance(content, str):
        yield content, ()
    elif isinstance(content, list):
        for block in content:
            yield from _text_blocks(block)
    elif isinstance(content, Mapping):
        kind = content.get("type")
        if kind in (None, "text", "output_text") and isinstance(text := content.get("text"), str):
            yield text, content.get("annotations") or content.get("citations", ())
        elif kind in (None, "message"):
            yield from _text_blocks(content.get("content"))


def _web_citation(annotation: Any) -> WebCitation | None:
    if not isinstance(annotation, Mapping) or annotation.get("type") not in (
        "url_citation", "citation", "web_search_result_location",
    ):
        return None
    details = annotation.get("url_citation", annotation)
    if not isinstance(details, Mapping) or not isinstance(url := details.get("url"), str):
        return None
    if not 0 < len(url) <= 1500 or any(char.isspace() or not char.isprintable() for char in url):
        return None
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username is not None:
            return None
    except ValueError:
        return None
    title = details.get("title")
    return WebCitation(" ".join(title.split())[:160] if isinstance(title, str) else "", url)


def _citation_offset(annotation, text: str) -> int:
    details = annotation.get("url_citation", annotation)
    offset = details.get("end_index", annotation.get("end_index"))
    if type(offset) is int and 0 < offset <= len(text):
        return offset
    cited_text = details.get("cited_text")
    return len(cited_text) if isinstance(cited_text, str) and cited_text and text.startswith(cited_text) else len(text)


def _message_text_blocks(message: Any) -> Iterator[tuple[str, Any]]:
    if isinstance(message, Mapping):
        content, extra = message.get("content"), message.get("additional_kwargs", {})
    else:
        content, extra = getattr(message, "content", None), getattr(message, "additional_kwargs", {})
    # Chat Completions puts citations beside the full text, including when it
    # streams the text in several chunks. Responses carries them on text blocks.
    annotations = extra.get("annotations") if isinstance(extra, Mapping) else None
    blocks = _text_blocks(content)
    if annotations:
        yield "".join(text for text, _ in blocks), annotations
    else:
        yield from blocks


def cited_message(message: Any) -> tuple[str, tuple[WebCitation, ...]]:
    sources: list[WebCitation] = []
    numbers: dict[str, int] = {}
    parts: list[str] = []
    for text, annotations in _message_text_blocks(message):
        references = []
        for annotation in annotations if isinstance(annotations, (list, tuple)) else ():
            if (source := _web_citation(annotation)) is not None:
                references.append((_citation_offset(annotation, text), source))
        markers: dict[int, list[int]] = {}
        for end, source in sorted(references, key=lambda item: item[0]):
            if source.url not in numbers and len(sources) < _MAX_CITATIONS:
                sources.append(source)
                numbers[source.url] = len(sources)
            if number := numbers.get(source.url):
                ids = markers.setdefault(end, [])
                if number not in ids:
                    ids.append(number)
        start = 0
        segments = []
        for end, ids in markers.items():
            segments.extend((text[start:end], "".join(f"〔{number}〕" for number in ids)))
            start = end
        parts.append("".join(segments) + text[start:])
    metadata = message.get("response_metadata", {}) if isinstance(message, Mapping) else getattr(message, "response_metadata", {})
    separator = "" if metadata.get("model_provider") == "anthropic" else "\n"
    return separator.join(parts), tuple(sources)
