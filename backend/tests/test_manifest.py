from pathlib import Path

import pytest

from medquery.baseline.manifest import (
    build_manifest,
    compare_manifests,
    format_manifest,
    parse_manifest,
    sha256_file,
    verify_manifest,
)
from medquery.config import get_settings


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    root = tmp_path / "archive"
    (root / "results").mkdir(parents=True)
    (root / "a.txt").write_bytes(b"alpha")
    (root / "results" / "b.csv").write_bytes(b"x,y\n1,2\n")
    return root


def test_sha256_known_value(tmp_path: Path) -> None:
    path = tmp_path / "f"
    path.write_bytes(b"abc")
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_manifest_uses_posix_relative_paths(tree: Path) -> None:
    assert sorted(build_manifest(tree)) == ["a.txt", "results/b.csv"]


def test_format_parse_round_trip(tree: Path) -> None:
    entries = build_manifest(tree)
    assert parse_manifest(format_manifest(entries)) == entries


def test_parse_rejects_malformed_lines() -> None:
    with pytest.raises(ValueError):
        parse_manifest("not-a-digest path\n")


def test_verify_detects_changed_missing_and_added(tree: Path, tmp_path: Path) -> None:
    manifest = tmp_path / "MANIFEST.sha256"
    manifest.write_text(format_manifest(build_manifest(tree)), encoding="utf-8")
    assert verify_manifest(tree, manifest).ok

    (tree / "a.txt").write_bytes(b"tampered")
    (tree / "results" / "b.csv").unlink()
    (tree / "new.txt").write_bytes(b"new")

    diff = verify_manifest(tree, manifest)
    assert not diff.ok
    assert diff.changed == ["a.txt"]
    assert diff.missing == ["results/b.csv"]
    assert diff.added == ["new.txt"]


def test_compare_identical_is_ok() -> None:
    entries = {"x": "0" * 64}
    assert compare_manifests(entries, dict(entries)).ok


@pytest.mark.archive
def test_real_archive_matches_committed_manifest() -> None:
    settings = get_settings()
    if not settings.archive_dir.is_dir() or not settings.baseline_manifest_path.is_file():
        pytest.skip("Research archive or manifest not available on this machine")
    diff = verify_manifest(settings.archive_dir, settings.baseline_manifest_path)
    assert diff.ok, f"Archive drifted: {diff}"
