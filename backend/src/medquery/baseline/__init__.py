"""Frozen dissertation baseline: integrity manifest and typed result loaders."""

from medquery.baseline.loader import BaselineResults, load_baseline
from medquery.baseline.manifest import ManifestDiff, build_manifest, verify_manifest

__all__ = ["BaselineResults", "ManifestDiff", "build_manifest", "load_baseline", "verify_manifest"]
