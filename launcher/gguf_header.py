"""gguf_header.py — Minimal, dependency-free reader for GGUF file metadata.

Reads the key/value section of a GGUF header (format versions 2 and 3) without
loading tensor data, so it is cheap even for very large models.  Used by ingest
to detect properties that only the GGUF itself records (e.g. an embedded MTP head).
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import BinaryIO

_MAGIC = b"GGUF"

# GGUF value type id → struct format for fixed-size scalars.
_SCALAR_FORMATS: dict[int, str] = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
}
_STRING = 8
_ARRAY = 9

Scalar = int | float | bool | str


def _read(f: BinaryIO, fmt: str) -> int | float | bool:
    size = struct.calcsize(fmt)
    data = f.read(size)
    if len(data) != size:
        raise ValueError("truncated GGUF header")
    return struct.unpack(fmt, data)[0]


def _read_string(f: BinaryIO) -> str:
    length = int(_read(f, "<Q"))
    data = f.read(length)
    if len(data) != length:
        raise ValueError("truncated GGUF header")
    return data.decode("utf-8", errors="replace")


def _skip_array(f: BinaryIO) -> None:
    elem_type = int(_read(f, "<I"))
    count = int(_read(f, "<Q"))
    if elem_type in _SCALAR_FORMATS:
        f.seek(struct.calcsize(_SCALAR_FORMATS[elem_type]) * count, 1)
    elif elem_type == _STRING:
        for _ in range(count):
            f.seek(int(_read(f, "<Q")), 1)
    elif elem_type == _ARRAY:
        for _ in range(count):
            _skip_array(f)
    else:
        raise ValueError(f"unknown GGUF array element type {elem_type}")


def read_gguf_metadata(path: Path) -> dict[str, Scalar]:
    """Return the scalar key/value metadata of the GGUF file at *path*.

    Array values (tokenizer vocabularies, etc.) are skipped.  For a split model,
    pass the first shard — it carries the model metadata.  Raises ValueError for
    files that are not valid GGUF.
    """
    metadata: dict[str, Scalar] = {}
    with open(path, "rb", buffering=1 << 20) as f:
        if f.read(4) != _MAGIC:
            raise ValueError(f"not a GGUF file: {path}")
        version = int(_read(f, "<I"))
        if version < 2:
            raise ValueError(f"unsupported GGUF version {version}: {path}")
        _read(f, "<Q")  # tensor count
        kv_count = int(_read(f, "<Q"))
        for _ in range(kv_count):
            key = _read_string(f)
            value_type = int(_read(f, "<I"))
            if value_type in _SCALAR_FORMATS:
                metadata[key] = _read(f, _SCALAR_FORMATS[value_type])
            elif value_type == _STRING:
                metadata[key] = _read_string(f)
            elif value_type == _ARRAY:
                _skip_array(f)
            else:
                raise ValueError(f"unknown GGUF value type {value_type} for {key!r}: {path}")
    return metadata


def arch_int(metadata: dict[str, Scalar], key: str) -> int:
    """Integer value of the architecture-scoped key ``<arch>.<key>`` (0 when absent)."""
    arch = metadata.get("general.architecture", "")
    return int(metadata.get(f"{arch}.{key}", 0) or 0)


def embedded_mtp_layers(metadata: dict[str, Scalar]) -> int:
    """Number of multi-token-prediction layers built into the model (0 = none)."""
    return arch_int(metadata, "nextn_predict_layers")
