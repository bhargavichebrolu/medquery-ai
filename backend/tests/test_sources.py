from pathlib import Path

import pytest

from medquery.ingestion import MalformedDocumentError, load_text_document


def test_loads_utf8_and_defaults_source_id_to_filename(tmp_path: Path) -> None:
    path = tmp_path / "asthma.txt"
    path.write_bytes("# Asthma\n\n• wheezing".encode())
    doc = load_text_document(path)
    assert doc.source_id == "asthma.txt"
    assert doc.text == "# Asthma\n\n• wheezing"
    assert doc.metadata == {}


def test_strips_utf8_bom(tmp_path: Path) -> None:
    path = tmp_path / "bom.txt"
    path.write_bytes(b"\xef\xbb\xbfhello")
    assert load_text_document(path).text == "hello"


def test_explicit_source_id_and_metadata_are_kept(tmp_path: Path) -> None:
    path = tmp_path / "x.txt"
    path.write_bytes(b"x")
    doc = load_text_document(path, source_id="kb1/x", metadata={"kb": "nhs", "version": 1})
    assert (doc.source_id, doc.metadata) == ("kb1/x", {"kb": "nhs", "version": 1})


def test_empty_file_loads_as_empty_document(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_bytes(b"")
    assert load_text_document(path).text == ""


def test_invalid_utf8_raises_malformed_document_error(tmp_path: Path) -> None:
    path = tmp_path / "latin1.txt"
    path.write_bytes("café".encode("latin-1"))
    with pytest.raises(MalformedDocumentError, match=r"latin1\.txt.*byte offset 3"):
        load_text_document(path)
