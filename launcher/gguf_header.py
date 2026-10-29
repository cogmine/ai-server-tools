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


def _open_header(f: BinaryIO, path: Path) -> tuple[int, int]:
    """Check the magic and version; return (tensor count, key/value count)."""
    if f.read(4) != _MAGIC:
        raise ValueError(f"not a GGUF file: {path}")
    version = int(_read(f, "<I"))
    if version < 2:
        raise ValueError(f"unsupported GGUF version {version}: {path}")
    return int(_read(f, "<Q")), int(_read(f, "<Q"))


def _read_kv(f: BinaryIO, kv_count: int, path: Path):
    """Yield (key, value type, file offset of the value, scalar value or None for arrays)."""
    for _ in range(kv_count):
        key = _read_string(f)
        value_type = int(_read(f, "<I"))
        offset = f.tell()
        if value_type in _SCALAR_FORMATS:
            yield key, value_type, offset, _read(f, _SCALAR_FORMATS[value_type])
        elif value_type == _STRING:
            yield key, value_type, offset, _read_string(f)
        elif value_type == _ARRAY:
            _skip_array(f)
            yield key, value_type, offset, None
        else:
            raise ValueError(f"unknown GGUF value type {value_type} for {key!r}: {path}")


def read_gguf_metadata(path: Path) -> dict[str, Scalar]:
    """Return the scalar key/value metadata of the GGUF file at *path*.

    Array values (tokenizer vocabularies, etc.) are skipped.  For a split model,
    pass the first shard — it carries the model metadata.  Raises ValueError for
    files that are not valid GGUF.
    """
    with open(path, "rb", buffering=1 << 20) as f:
        _, kv_count = _open_header(f, path)
        return {key: value for key, vtype, _, value in _read_kv(f, kv_count, path) if vtype != _ARRAY}


def read_gguf_tensor_names(path: Path) -> list[str]:
    """Names of the tensors stored in the GGUF file at *path* (one shard of a split set)."""
    with open(path, "rb", buffering=1 << 20) as f:
        tensor_count, kv_count = _open_header(f, path)
        for _ in _read_kv(f, kv_count, path):
            pass
        names: list[str] = []
        for _ in range(tensor_count):
            names.append(_read_string(f))
            n_dims = int(_read(f, "<I"))
            f.seek(8 * n_dims + 4 + 8, 1)  # dims, ggml type, data offset
        return names


def split_position(metadata: dict[str, Scalar]) -> tuple[int, int]:
    """(1-based shard number, shard count) recorded in a split shard's header; (1, 1) when unsplit."""
    return int(metadata.get("split.no", 0)) + 1, int(metadata.get("split.count", 1))


def stamp_split_position(path: Path, shard: int, count: int) -> None:
    """Rewrite the split.no / split.count header values in place (the file is otherwise unchanged).

    Lets a table file published as a standalone GGUF (e.g. Unsloth's Engram tables)
    load as shard *shard* of a *count*-shard set: llama.cpp checks both values.
    """
    with open(path, "r+b") as f:
        _, kv_count = _open_header(f, path)
        targets = {"split.no": shard - 1, "split.count": count}
        found = {
            key: (vtype, offset)
            for key, vtype, offset, _ in _read_kv(f, kv_count, path) if key in targets
        }
        if set(found) != set(targets):
            raise ValueError(f"{path} has no split.no / split.count to stamp")
        for key, (vtype, offset) in found.items():
            f.seek(offset)
            f.write(struct.pack(_SCALAR_FORMATS[vtype], targets[key]))


def arch_int(metadata: dict[str, Scalar], key: str) -> int:
    """Integer value of the architecture-scoped key ``<arch>.<key>`` (0 when absent)."""
    arch = metadata.get("general.architecture", "")
    return int(metadata.get(f"{arch}.{key}", 0) or 0)


def embedded_mtp_layers(metadata: dict[str, Scalar]) -> int:
    """Number of multi-token-prediction layers built into the model (0 = none)."""
    return arch_int(metadata, "nextn_predict_layers")
