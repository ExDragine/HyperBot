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
            yield text, content.get("annotations", ())
        elif kind in (None, "message"):
            yield from _text_blocks(content.get("content"))


def _web_citation(annotation: Any) -> WebCitation | None:
    if not isinstance(annotation, Mapping) or annotation.get("type") not in ("url_citation", "citation"):
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


def cited_message(message: Any) -> tuple[str, tuple[WebCitation, ...]]:
    content = message.get("content") if isinstance(message, Mapping) else getattr(message, "content", None)
    sources: list[WebCitation] = []
    numbers: dict[str, int] = {}
    parts: list[str] = []
    for text, annotations in _text_blocks(content):
        references = []
        for annotation in annotations if isinstance(annotations, (list, tuple)) else ():
            if (source := _web_citation(annotation)) is not None:
                offset = annotation.get("end_index")
                end = offset if type(offset) is int and 0 < offset <= len(text) else len(text)
                references.append((end, source))
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
    return "\n".join(parts), tuple(sources)
