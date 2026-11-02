"""Check the research archive against the pipeline described in docs/research-pipeline.md.

Every check only reads files. The notebook checks read Healthcare_RAG/notebooks and
Healthcare_RAG/results (both tracked in git). The data checks read Healthcare_RAG/data (or
MEDQUERY_DATA_DIR) and are skipped, with the reason shown, when it is absent. Run with `-rP`
to see the values each check measured.

These checks show that the files are consistent with the notebook code. They do not show which
notebook run produced a file.
"""

import ast
import inspect
import json
import math
import re
import struct
import sys
from array import array
from pathlib import Path

import pytest

from medquery_api import config

NOTEBOOKS = config.REPO_ROOT / "Healthcare_RAG" / "notebooks"
RESULTS = config.REPO_ROOT / "Healthcare_RAG" / "results"
DATA = config.data_dir()

NOTEBOOK_FILES = {
    "01": "01_Data_Collection.ipynb",
    "02": "02_Data_Cleaning.ipynb",
    "03": "03_Text_Chunking.ipynb",
    "04": "04_Embedding_Generation.ipynb",
    "05": "05_Vector_Database_FAISS(Storing Vectors in FAISS).ipynb",
    "06": "06_RAG_Pipeline.ipynb",
    "06(1)": "06_RAG_Pipeline(1).ipynb",
}
MODEL_DIMENSIONS = {"MiniLM": 384, "MPNet": 768, "E5": 768, "BGE": 384}
CHUNK_FIELDS = ("chunk_id", "source_file", "chunk_length", "text")

# Notebook 04 normalised every vector to length 1 (normalize_embeddings=True) using float32,
# which keeps about 7 significant digits. Rounding in that calculation can move a length away
# from 1 by up to roughly dimension x 2^-24 (768 x 6e-8 = 4.6e-5). 1e-4 allows for that rounding
# and nothing larger.
NORM_TOLERANCE = 1e-4

needs_data = pytest.mark.skipif(
    not (DATA / "chunks").is_dir(),
    reason=f"research data not found at {DATA}; the archive checks run only where it exists "
    "(MEDQUERY_DATA_DIR can point to another folder)",
)


# --- The cleaning function from notebook 02, cell 4 (copied; comments removed) ---


def clean_text(text):
    """Copy of notebook 02's clean_text. A test below checks the copy is unchanged."""

    lines = text.splitlines()
    cleaned_lines = []
    seen = set()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"\s+", " ", line)
        if line in seen:
            continue
        seen.add(line)
        cleaned_lines.append(line)
    cleaned_text = "\n\n".join(cleaned_lines)
    return cleaned_text


# --- Reading helpers ---


def notebook_cells(key: str) -> list[str]:
    """The source code of every cell in a notebook, by cell number."""
    with (NOTEBOOKS / NOTEBOOK_FILES[key]).open("r", encoding="utf-8") as file:
        return ["".join(cell["source"]) for cell in json.load(file)["cells"]]


def parse_cell(source: str) -> ast.Module:
    """Parse a cell as Python, leaving out shell commands such as `!pip install` (and any
    lines they continue onto with a trailing backslash)."""
    kept, in_shell_command = [], False
    for line in source.splitlines():
        if line.startswith("!") or in_shell_command:
            in_shell_command = line.rstrip().endswith("\\")
        else:
            kept.append(line)
    return ast.parse("\n".join(kept))


def calls(tree: ast.AST, name: str) -> list[ast.Call]:
    """Every call in the tree to a function or method with this name."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (getattr(node.func, "attr", None) == name or getattr(node.func, "id", None) == name)
    ]


def function_without_docstring(tree: ast.AST, name: str) -> str:
    """A function's code structure, ignoring comments, formatting and its docstring."""
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    first = function.body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
        function.body = function.body[1:]
    return ast.dump(function)


def squash(text: str) -> str:
    return " ".join(text.split())


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def embedding_to_float32(record: dict) -> dict:
    if "embedding" in record:
        record["embedding"] = array("f", record["embedding"])
    return record


def read_embeddings(paths: list[Path]) -> tuple[list[dict], array, set[int]]:
    """Chunk fields, all vectors as one float32 array, and the vector sizes found.

    Each vector is turned into float32 as soon as it is parsed, so the millions of numbers in
    a file are not all kept as Python floats. The values were float32 when the notebook saved
    them, so this conversion is exact. Peak memory is still about three times the file size,
    because each file is a single JSON array that the standard library must read in one piece.
    """
    fields, vectors, sizes = [], array("f"), set()
    for path in paths:
        with path.open("r", encoding="utf-8") as file:
            for record in json.load(file, object_hook=embedding_to_float32):
                embedding = record.pop("embedding")
                sizes.add(len(embedding))
                vectors.extend(embedding)
                fields.append(record)
    return fields, vectors, sizes


def read_flat_index(path: Path) -> dict:
    """Read a FAISS flat index file without the faiss library.

    The layout faiss.write_index uses for an IndexFlatIP: the type code "IxFI", int32 vector
    size, int64 vector count, two int64 placeholders, one byte "is trained", int32 metric
    (0 = inner product), uint64 number of floats, then the vectors as little-endian float32.
    """
    data = path.read_bytes()
    dimension, count = struct.unpack_from("<iq", data, 4)
    (metric,) = struct.unpack_from("<i", data, 33)
    (floats,) = struct.unpack_from("<Q", data, 37)
    vectors = array("f")
    vectors.frombytes(data[45:])
    if sys.byteorder != "little":
        vectors.byteswap()
    return {
        "type": data[:4],
        "dimension": dimension,
        "count": count,
        "metric": metric,
        "floats": floats,
        "file_size_matches": len(data) == 45 + 4 * floats,
        "vectors": vectors,
    }


def length_deviations(vectors: array, dimension: int) -> list[float]:
    """For each vector, how far its length is from 1 (computed in double precision)."""
    return [
        abs(math.sqrt(math.fsum(x * x for x in vectors[i : i + dimension])) - 1.0)
        for i in range(0, len(vectors), dimension)
    ]


def compare_vectors(a: array, b: array, dimension: int) -> dict:
    """How many vectors are bit-identical, and the largest differences between the rest."""
    identical, max_difference, max_cosine_gap = 0, 0.0, 0.0
    for i in range(0, len(a), dimension):
        x, y = a[i : i + dimension], b[i : i + dimension]
        if x.tobytes() == y.tobytes():
            identical += 1
            continue
        max_difference = max(max_difference, max(abs(p - q) for p, q in zip(x, y, strict=True)))
        dot = math.fsum(p * q for p, q in zip(x, y, strict=True))
        lengths = math.sqrt(math.fsum(p * p for p in x) * math.fsum(q * q for q in y))
        max_cosine_gap = max(max_cosine_gap, 1.0 - dot / lengths)
    return {
        "identical": identical,
        "max_difference": max_difference,
        "max_cosine_gap": max_cosine_gap,
    }


@pytest.fixture(scope="module")
def chunk_files() -> list[Path]:
    # Sorted by file name, as notebook 04 did (sorted(CHUNK_DIR.glob("*.json")) on Linux).
    return sorted((DATA / "chunks").glob("*.json"), key=lambda path: path.name)


@pytest.fixture(scope="module")
def chunks(chunk_files) -> list[dict]:
    """Every chunk record, in the order notebook 04 embedded them."""
    return [record for path in chunk_files for record in read_json(path)]


# --- Notebook code: the settings the documentation quotes ---


def test_clean_text_copy_matches_notebook_02():
    notebook = function_without_docstring(parse_cell(notebook_cells("02")[4]), "clean_text")
    copy = function_without_docstring(ast.parse(inspect.getsource(clean_text)), "clean_text")

    assert copy == notebook


def test_chunking_settings_match_notebook_03():
    (splitter,) = calls(parse_cell(notebook_cells("03")[5]), "RecursiveCharacterTextSplitter")
    settings = {keyword.arg: ast.literal_eval(keyword.value) for keyword in splitter.keywords}

    assert settings == {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "separators": ["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    }


def test_embedding_settings_match_notebook_04():
    cells = notebook_cells("04")
    models = next(
        ast.literal_eval(node.value)
        for node in ast.walk(parse_cell(cells[4]))
        if isinstance(node, ast.Assign)
        and getattr(node.targets[0], "id", None) == "EMBEDDING_MODELS"
    )
    encode_calls = [call for cell in cells for call in calls(parse_cell(cell), "encode")]

    assert models == {
        "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
        "MPNet": "sentence-transformers/all-mpnet-base-v2",
        "E5": "intfloat/multilingual-e5-base",
        "BGE": "BAAI/bge-small-en-v1.5",
    }
    assert len(encode_calls) == 4
    for call in encode_calls:
        keywords = {keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords}
        assert keywords["normalize_embeddings"] is True
    assert "f\"passage: {chunk['text']}\"" in cells[15]  # E5 passages only


def test_index_and_retrieval_settings_match_notebooks_05_and_06():
    index_cell = squash(notebook_cells("05")[5])
    retrieval_cell = squash(notebook_cells("06(1)")[5])

    assert "index = faiss.IndexFlatIP(dimension)" in index_cell
    assert "index.add(vectors)" in index_cell
    assert (
        'if model_name == "E5": query_text = f"query: {question}" else: query_text = question'
        in retrieval_cell
    )
    assert "normalize_embeddings=True" in retrieval_cell
    assert "scores, indices = index.search( query_embedding, top_k )" in retrieval_cell


def test_the_two_rag_notebook_copies_differ_only_in_two_interface_cells():
    first, second = notebook_cells("06"), notebook_cells("06(1)")
    differing = [i for i, (a, b) in enumerate(zip(first, second, strict=True)) if a != b]

    assert differing == [26, 43]
    for i in differing:  # both are Gradio web-interface cells
        assert "gr.Blocks(" in first[i] and "gr.Blocks(" in second[i]


# Saved cells that are not valid Python as stored:
# - 01 cell 7: a comment divider line is broken onto a second line, yet its saved output shows
#   a successful run, so the stored code is not exactly the code that ran.
# - 06 cell 26: the interface code is mis-indented; its saved output is that IndentationError.
#   The same cell in 06_RAG_Pipeline(1).ipynb is valid and its output shows the interface ran.
UNPARSEABLE_CELLS = [("01", 7), ("06", 26)]


def test_only_two_saved_cells_are_not_valid_python():
    unparseable = []
    for key in NOTEBOOK_FILES:
        for i, source in enumerate(notebook_cells(key)):
            try:
                parse_cell(source)
            except SyntaxError:
                unparseable.append((key, i))

    assert unparseable == UNPARSEABLE_CELLS
    parse_cell(notebook_cells("06(1)")[26])


def test_which_archive_files_no_notebook_writes():
    cells = {key: notebook_cells(key) for key in NOTEBOOK_FILES}
    parsed = {
        (key, i): parse_cell(source)
        for key, sources in cells.items()
        for i, source in enumerate(sources)
        if (key, i) not in UNPARSEABLE_CELLS
    }
    # The two cells that cannot be parsed are checked by their text instead.
    for key, i in UNPARSEABLE_CELLS:
        assert not re.search(r"write_index|EMBEDDING_DIR|basic_rag_test_results", cells[key][i])
    index_writes = [cell for cell, tree in parsed.items() if calls(tree, "write_index")]
    embedding_writes = [
        cell
        for cell, tree in parsed.items()
        if (calls(tree, "dump") or calls(tree, "write"))
        and "EMBEDDING_DIR" in cells[cell[0]][cell[1]]
    ]
    mentions_basic_results = [
        cell for cell in parsed if "basic_rag_test_results.json" in cells[cell[0]][cell[1]]
    ]

    # The only index written is one per model folder, so faiss/faiss_index.bin has no writer.
    assert index_writes == [("05", 5)]
    assert 'model_faiss_dir / "faiss_index.bin"' in squash(cells["05"][5])
    # Embedding JSON is written only into the four model folders, so the files at the top of
    # data/embeddings have no writer.
    assert embedding_writes == [("04", 8), ("04", 12), ("04", 17), ("04", 21)]
    for _, i in embedding_writes:
        assert re.search(r'[A-Z0-9]+_DIR / "all_embeddings\.json"', cells["04"][i])
    # basic_rag_test_results.json is read (if present) but never written. The cell's saved
    # output reports 5 saved test cases, which matches the file in the archive.
    assert mentions_basic_results == [("05", 13)]
    assert not calls(parsed[("05", 13)], "dump")
    assert len(read_json(RESULTS / "basic_rag_test_results.json")) == 5


# --- Data files: each stage of the pipeline ---


@needs_data
def test_url_list_reduces_to_one_address_per_article():
    # Counts only: no address is printed or compared, even if the check fails.
    with (DATA / "urls" / "nhs_urls.txt").open("r", encoding="utf-8") as file:
        lines = file.readlines()
    non_blank = [line.strip() for line in lines if line.strip()]
    line_count = len(lines)
    unique_count = len(dict.fromkeys(url.replace("http://", "https://") for url in non_blank))
    raw_count = len(list((DATA / "raw").glob("*.txt")))
    print(f"measured: {line_count} lines, {unique_count} unique addresses, {raw_count} raw files")

    assert line_count == 840
    assert unique_count == 698
    assert raw_count == 698


@needs_data
def test_every_article_passes_through_every_stage(chunk_files):
    raw = sorted(path.stem for path in (DATA / "raw").glob("*.txt"))
    cleaned = sorted(path.stem for path in (DATA / "cleaned").glob("*.txt"))
    chunked = sorted(path.stem for path in chunk_files)

    assert len(raw) == 698
    assert raw == cleaned == chunked
    for model in MODEL_DIMENSIONS:
        assert (DATA / "embeddings" / model / "all_embeddings.json").is_file()
        assert (DATA / "faiss" / model / "faiss_index.bin").is_file()
        assert (DATA / "faiss" / model / "metadata.json").is_file()


@needs_data
def test_cleaning_reproduces_every_cleaned_article_exactly():
    reproduced, different = 0, []
    for raw_path in sorted((DATA / "raw").glob("*.txt")):
        # Read the raw file as notebook 02 did; read the cleaned file exactly as stored.
        with raw_path.open("r", encoding="utf-8") as file:
            raw = file.read()
        with (DATA / "cleaned" / raw_path.name).open("r", encoding="utf-8", newline="") as file:
            stored = file.read()
        if clean_text(raw) == stored:
            reproduced += 1
        else:
            different.append(raw_path.name)
    print(f"measured: {reproduced} of {reproduced + len(different)} reproduced byte for byte")

    assert different == []
    assert reproduced == 698


@needs_data
def test_chunk_records_match_the_chunking_settings(chunk_files, chunks):
    lengths = [record["chunk_length"] for record in chunks]
    print(
        f"measured: {len(chunks)} chunks, length min {min(lengths)}, max {max(lengths)}, "
        f"mean {sum(lengths) / len(lengths):.2f}"
    )

    assert len(chunks) == 3796
    for path in chunk_files:
        records = read_json(path)
        # chunk_id counts from 1 within each article; (source_file, chunk_id) is the unique key.
        assert [record["chunk_id"] for record in records] == list(range(1, len(records) + 1))
        assert {record["source_file"] for record in records} == {f"{path.stem}.txt"}
    for record in chunks:
        assert tuple(record) == CHUNK_FIELDS
        assert record["chunk_length"] == len(record["text"]) <= 1000
        assert record["text"] == record["text"].strip()
    # The same statistics notebook 03 printed after chunking.
    assert (min(lengths), max(lengths), round(sum(lengths) / len(lengths), 2)) == (16, 999, 645.59)


@needs_data
def test_chunks_are_ordered_overlapping_pieces_that_cover_each_article(chunk_files):
    overlaps, uncovered = [], 0
    for path in chunk_files:
        with (DATA / "cleaned" / f"{path.stem}.txt").open("r", encoding="utf-8", newline="") as f:
            article = f.read()
        covered = [False] * len(article)
        previous_start, previous_end = -1, None
        for record in read_json(path):
            start = article.find(record["text"], previous_start + 1)
            assert start >= 0, f"{path.name} chunk {record['chunk_id']} is not in the article"
            end = start + len(record["text"])
            if previous_end is not None:
                overlaps.append(max(0, previous_end - start))
            covered[start:end] = [True] * (end - start)
            previous_start, previous_end = start, end
        uncovered += sum(
            1 for char, cov in zip(article, covered, strict=True) if not cov and not char.isspace()
        )
    overlapping = sum(1 for overlap in overlaps if overlap > 0)
    print(
        f"measured: {overlapping} of {len(overlaps)} neighbouring pairs overlap, "
        f"largest overlap {max(overlaps)} characters, {uncovered} non-space characters uncovered"
    )

    assert max(overlaps) <= 200
    assert uncovered == 0


@needs_data
@pytest.mark.parametrize("model", MODEL_DIMENSIONS)
def test_embeddings_and_index_match_the_chunks(model, chunks):
    dimension = MODEL_DIMENSIONS[model]
    fields, vectors, sizes = read_embeddings([DATA / "embeddings" / model / "all_embeddings.json"])
    index = read_flat_index(DATA / "faiss" / model / "faiss_index.bin")
    metadata = read_json(DATA / "faiss" / model / "metadata.json")
    deviations = length_deviations(vectors, dimension)
    exactly_one = sum(1 for deviation in deviations if deviation == 0.0)
    mean_deviation = sum(deviations) / len(deviations)
    print(
        f"measured: {model} {len(deviations)} vectors of size {sorted(sizes)}; "
        f"largest |length - 1| = {max(deviations):.3e}, mean {mean_deviation:.3e}; "
        f"{exactly_one} exactly 1, {len(deviations) - exactly_one} other within "
        f"tolerance {NORM_TOLERANCE:g}"
    )

    # Embedding records: same chunks, same order, stored text without the E5 "passage: " prefix.
    assert fields == chunks
    assert sizes == {dimension}
    # Unit length within the float32 tolerance, consistent with normalize_embeddings=True.
    assert max(deviations) <= NORM_TOLERANCE
    # The index: an exact inner-product index holding exactly these vectors, in this order.
    assert (index["type"], index["metric"]) == (b"IxFI", 0)
    assert (index["dimension"], index["count"]) == (dimension, 3796)
    assert index["floats"] == dimension * 3796 and index["file_size_matches"]
    assert index["vectors"].tobytes() == vectors.tobytes()
    # metadata.json: row i describes vector i.
    assert metadata == chunks


@needs_data
def test_leftover_files_hold_one_set_of_vectors_almost_identical_to_bge(chunk_files, chunks):
    embeddings = DATA / "embeddings"
    per_article = sorted(
        (p for p in embeddings.glob("*.json") if p.name != "all_embeddings.json"),
        key=lambda path: path.name,
    )
    combined_fields, combined, combined_sizes = read_embeddings(
        [embeddings / "all_embeddings.json"]
    )
    article_fields, by_article, article_sizes = read_embeddings(per_article)
    index = read_flat_index(DATA / "faiss" / "faiss_index.bin")
    metadata = read_json(DATA / "faiss" / "metadata.json")
    _, bge, _ = read_embeddings([embeddings / "BGE" / "all_embeddings.json"])
    _, minilm, _ = read_embeddings([embeddings / "MiniLM" / "all_embeddings.json"])
    to_bge = compare_vectors(combined, bge, 384)
    to_minilm = compare_vectors(combined, minilm, 384)
    print(
        f"measured: vs BGE {to_bge['identical']} of 3796 bit-identical, "
        f"largest difference {to_bge['max_difference']:.3e}, "
        f"largest 1 - cosine {to_bge['max_cosine_gap']:.3e}; "
        f"vs MiniLM {to_minilm['identical']} bit-identical, "
        f"largest difference {to_minilm['max_difference']:.3e}"
    )

    # embeddings/all_embeddings.json, the 698 embeddings/<article>.json files and
    # faiss/faiss_index.bin (+ metadata.json) describe the same chunks with the same vectors.
    assert [path.name for path in per_article] == [path.name for path in chunk_files]
    assert combined_fields == article_fields == metadata == chunks
    assert combined_sizes == article_sizes == {384}
    assert (index["type"], index["metric"], index["dimension"], index["count"]) == (
        b"IxFI",
        0,
        384,
        3796,
    )
    assert combined.tobytes() == by_article.tobytes() == index["vectors"].tobytes()
    # Not a copy of the BGE file, but within float32 rounding of it; far from another model.
    assert to_bge["identical"] < 3796
    assert to_bge["max_difference"] <= 1e-6
    assert to_minilm["max_difference"] > 0.1
