"""Create or verify the checksum manifest of the read-only research archive.

Usage:
    uv run python scripts/build_manifest.py            # verify (default)
    uv run python scripts/build_manifest.py --write    # (re)create the manifest

Re-creating the manifest should only happen deliberately, e.g. if the archive is
intentionally extended; it is the evidence that the original results are untouched.
"""

from __future__ import annotations

import argparse
import sys

from medquery.baseline.manifest import build_manifest, format_manifest, verify_manifest
from medquery.config import get_settings


def main() -> int:
    parser = argparse.ArgumentParser(description="Archive checksum manifest.")
    parser.add_argument("--write", action="store_true", help="Create/overwrite the manifest.")
    args = parser.parse_args()

    settings = get_settings()
    manifest_path = settings.ensure_writable(settings.baseline_manifest_path)

    if args.write:
        entries = build_manifest(settings.archive_dir)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(format_manifest(entries), encoding="utf-8", newline="\n")
        print(f"Wrote {len(entries)} entries to {manifest_path}")
        return 0

    diff = verify_manifest(settings.archive_dir, manifest_path)
    if diff.ok:
        print(f"Archive matches manifest: {manifest_path}")
        return 0
    for label, paths in (
        ("MISSING", diff.missing),
        ("ADDED", diff.added),
        ("CHANGED", diff.changed),
    ):
        for path in paths:
            print(f"{label:8} {path}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
