"""Filename-stem and annotation-id parsing.

Image stems look like ``20150125172714Mh``: a 14-digit timestamp
(YYYYMMDDHHMMSS), a one-letter GONG site code, and a trailing ``h``.
Annotation ``image_id`` values look like ``040301-20140609195854Bh``:
``<annotator_code>-<image_stem>``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

STEM_RE = re.compile(r"^(?P<prefix>\d+_)?(?P<ts>\d{14})(?P<site>[A-Za-z])h$")


@dataclass(frozen=True)
class ParsedStem:
    raw: str
    prefix: Optional[str]
    timestamp: datetime
    site: str


def parse_stem(stem: str) -> Optional[ParsedStem]:
    """Parse a filename stem. Returns ``None`` (never raises) if it doesn't match."""
    m = STEM_RE.match(stem)
    if m is None:
        return None
    ts_str = m.group("ts")
    try:
        ts = datetime.strptime(ts_str, "%Y%m%d%H%M%S")
    except ValueError:
        return None
    prefix = m.group("prefix")
    return ParsedStem(
        raw=stem,
        prefix=prefix[:-1] if prefix else None,
        timestamp=ts,
        site=m.group("site").upper(),
    )


def parse_image_id(image_id: str) -> tuple[str, str]:
    """Split an annotation ``image_id`` into ``(annotator_code, image_stem)``."""
    annotator, stem = image_id.split("-", 1)
    return annotator, stem
