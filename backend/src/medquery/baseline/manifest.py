"""SHA-256 checksum manifest proving the read-only research archive is unchanged.

The manifest uses the `sha256sum` text format (`<hex digest>  <relative/posix/path>`),
so it can also be checked with standard tools.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

_BLOCK_SIZE = 1 << 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(root: Path) -> dict[str, str]:
    """Hash every file under `root`, keyed by POSIX path relative to `root`."""
    files = sorted(p for p in root.rglob("*") if p.is_file())
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in files}


def format_manifest(entries: dict[str, str]) -> str:
    return "".join(f"{digest}  {path}\n" for path, digest in sorted(entries.items()))


def parse_manifest(text: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        digest, sep, path = line.partition("  ")
        if not sep or len(digest) != 64 or not path:
            raise ValueError(f"Malformed manifest line {line_number}: {line!r}")
        entries[path] = digest
    return entries


@dataclass(frozen=True)
class ManifestDiff:
    missing: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.missing or self.added or self.changed)


def compare_manifests(expected: dict[str, str], actual: dict[str, str]) -> ManifestDiff:
    return ManifestDiff(
        missing=sorted(expected.keys() - actual.keys()),
        added=sorted(actual.keys() - expected.keys()),
        changed=sorted(p for p in expected.keys() & actual.keys() if expected[p] != actual[p]),
    )


def verify_manifest(root: Path, manifest_path: Path) -> ManifestDiff:
    expected = parse_manifest(manifest_path.read_text(encoding="utf-8"))
    return compare_manifests(expected, build_manifest(root))
