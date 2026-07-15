"""Portable content hashes for versioned documentation sources."""

from __future__ import annotations

import hashlib
from pathlib import Path

TEXT_SUFFIXES = {".ipynb", ".md", ".txt"}


def canonical_sha256(path: Path) -> str:
    """Hash text with LF endings and binary files byte-for-byte.

    Git may expose the same tracked text as CRLF on Windows and LF on Linux. Normalizing
    line endings before hashing keeps the documentation inventory portable across both
    environments while preserving exact hashes for PDFs and other binary sources.
    """

    if path.suffix.lower() in TEXT_SUFFIXES:
        content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        return hashlib.sha256(content).hexdigest()

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
