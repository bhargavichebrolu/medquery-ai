"""Read a FAISS flat inner-product index file (IndexFlatIP) without the faiss library.

This is the only index type the research used (notebook 05). The layout, as faiss.write_index
writes it, was confirmed against all the archive's index files in PRD 2.0:

    bytes 0-3    type code "IxFI" (flat index, inner product)
    4-7          int32   vector size (dimension)
    8-15         int64   number of vectors
    16-31        two int64 placeholders, both 2**20
    32           byte    "is trained", 1
    33-36        int32   metric, 0 = inner product
    37-44        uint64  number of floats that follow (vector size x number of vectors)
    45-          the vectors, little-endian float32, one after another

The header and the exact file size are checked before any vector data is read, so a damaged
or unexpected file is rejected without allocating memory for it.
"""

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HEADER_SIZE = 45
TYPE_CODE = b"IxFI"
PLACEHOLDER = 2**20
METRIC_INNER_PRODUCT = 0


class IndexFormatError(ValueError):
    """The file is not a FAISS flat inner-product index in the expected layout."""


@dataclass(frozen=True)
class FlatIndex:
    path: Path
    vectors: np.ndarray  # shape (count, dimension), float32, read-only

    @property
    def dimension(self) -> int:
        return self.vectors.shape[1]

    @property
    def count(self) -> int:
        return self.vectors.shape[0]


def read_flat_index(path: Path) -> FlatIndex:
    """Read and check an IndexFlatIP file. Raises IndexFormatError if anything is off."""
    file_size = path.stat().st_size
    if file_size < HEADER_SIZE:
        raise IndexFormatError(f"{path}: {file_size} bytes is too short for an index header")
    with path.open("rb") as file:
        header = file.read(HEADER_SIZE)

    type_code = header[:4]
    dimension, count, placeholder_1, placeholder_2 = struct.unpack_from("<iqqq", header, 4)
    is_trained = header[32]
    (metric,) = struct.unpack_from("<i", header, 33)
    (floats,) = struct.unpack_from("<Q", header, 37)

    if type_code != TYPE_CODE:
        raise IndexFormatError(
            f"{path}: index type {type_code!r} is not supported; only {TYPE_CODE!r} "
            "(flat, inner product) is"
        )
    if metric != METRIC_INNER_PRODUCT:
        raise IndexFormatError(f"{path}: metric {metric} is not inner product (0)")
    if (placeholder_1, placeholder_2, is_trained) != (PLACEHOLDER, PLACEHOLDER, 1):
        raise IndexFormatError(f"{path}: unexpected header fields; not a flat index file")
    if dimension <= 0 or count < 0:
        raise IndexFormatError(f"{path}: invalid size {dimension} or count {count}")
    if floats != dimension * count:
        raise IndexFormatError(
            f"{path}: header says {floats} numbers, but {count} vectors of size {dimension} "
            f"need {dimension * count}"
        )
    expected_size = HEADER_SIZE + 4 * floats
    if file_size != expected_size:
        raise IndexFormatError(
            f"{path}: file is {file_size} bytes; the header implies exactly {expected_size}"
        )

    vectors = np.fromfile(path, dtype="<f4", count=floats, offset=HEADER_SIZE)
    vectors = vectors.astype(np.float32, copy=False).reshape(count, dimension)
    vectors.setflags(write=False)
    return FlatIndex(path=path, vectors=vectors)
