"""
storage.py — Storage locations and the files that make up a model on disk.

Every place models are stored (NAS, local drives) is a configured Location.
A model's catalog entry has exactly one path; copies of a model are separate
entries (same id and family) at the same path *relative to* another location's
root, so the layout under every root mirrors the others.

copy_model(), fill_missing() and DeletionPlan are the only operations that write or
delete model files; catalog bookkeeping for them lives in launch.py.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from launcher.config import CONFIG
from launcher.models import AIModel, gguf_shard_paths


@dataclass(frozen=True)
class Location:
    """A configured storage location (config.toml [[locations]])."""
    label: str              # short name for the Location column, e.g. "NAS"
    root: Path
    description: str = ""   # longer name for destination pickers

    @property
    def title(self) -> str:
        return self.description or self.label

    @property
    def online(self) -> bool:
        """False while the location is unavailable (e.g. NAS not mounted)."""
        return self.root.is_dir()

    def disk_info(self) -> tuple[float, float]:
        """(total_gb, free_gb) of the filesystem holding the root; (0, 0) when offline."""
        try:
            usage = shutil.disk_usage(self.root)
        except OSError:
            return 0.0, 0.0
        return usage.total / 1024**3, usage.free / 1024**3

    def contains(self, path: Path) -> bool:
        return path.is_relative_to(self.root)


LOCATIONS: list[Location] = [
    Location(label=loc.label, root=loc.root, description=loc.description)
    for loc in CONFIG.locations
]


def location_of(path: "str | Path", locations: "Iterable[Location] | None" = None) -> Location | None:
    """The location whose root contains *path* (the innermost root when roots nest), or None."""
    p = Path(path)
    matches = [loc for loc in (LOCATIONS if locations is None else locations) if loc.contains(p)]
    return max(matches, key=lambda loc: len(loc.root.parts), default=None)


def mirrored_path(path: "str | Path", dest: Location) -> Path:
    """*path* moved to the same relative position under *dest*'s root.

    Raises ValueError when *path* is not inside a configured location.
    """
    src = location_of(path)
    if src is None:
        raise ValueError(f"{path} is not inside a configured storage location")
    return dest.root / Path(path).relative_to(src.root)


def relocated_entry(entry: dict, dest: Location) -> dict:
    """A snapshot of JSON catalog *entry* for its copy at the mirrored path under *dest*."""
    relocated = json.loads(json.dumps(entry))  # deep copy: nested status / params dicts
    relocated["path"] = str(mirrored_path(entry["path"], dest))
    for key in ("mmproj_path", "mtp_path"):
        if relocated.get(key):
            relocated[key] = str(mirrored_path(relocated[key], dest))
    return relocated


# ---------------------------------------------------------------------------
# Files making up a model
# ---------------------------------------------------------------------------

# HF config / tokenizer files that describe a model; copied beside GGUF / converted
# outputs so ingest can read the model's architecture, context and chat template.
METADATA_FILES: tuple[str, ...] = (
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "generation_config.json",
    "vocab.json",
    "merges.txt",
    # transformers >= 4.55 ships the chat template as a standalone file rather
    # than embedding it in tokenizer_config.json; .json is the older multimodal form.
    "chat_template.jinja",
    "chat_template.json",
)

_WEIGHT_SUFFIXES = frozenset({".gguf", ".safetensors", ".bin", ".pt", ".pth", ".ckpt"})


def model_metadata_dir(path: Path) -> Path:
    """Directory holding the HF config/tokenizer files that describe the model at *path*.

    An HF model directory describes itself.  A GGUF file's config.json (placed by
    copy-configs or a conversion) sits beside it, or in the bundle root one level up
    for the per-quant-subdir layout (bundle_root/QUANTNAME/file.gguf).
    """
    if path.is_dir():
        return path
    return next(
        (d for d in (path.parent, path.parent.parent) if (d / "config.json").is_file()),
        path.parent,
    )


def model_files(model: AIModel) -> list[Path]:
    """Every path that makes up *model* on disk.

    A directory model is its directory.  A file model is its file (every shard of a
    split GGUF), its companion files (mmproj, MTP head) and the metadata files
    describing it.
    """
    path = Path(model.path)
    if path.is_dir():
        return [path]
    meta_dir = model_metadata_dir(path)
    return [
        *gguf_shard_paths(path),
        *(Path(c) for c in model.companion_paths),
        *(f for f in (meta_dir / name for name in METADATA_FILES) if f.is_file()),
    ]


def model_size_gb(model: AIModel) -> float:
    """Actual on-disk footprint of *model* (all files copy_model() transfers), excluding .git."""
    total = 0
    for p in model_files(model):
        if p.is_dir():
            total += sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and ".git" not in f.parts)
        elif p.is_file():
            total += p.stat().st_size
    return total / 1024**3


def _files_in(path: Path) -> list[Path]:
    """*path* itself when a file; every file under it (excluding .git) when a directory."""
    if path.is_dir():
        return [f for f in path.rglob("*") if f.is_file() and ".git" not in f.parts]
    return [path] if path.is_file() else []


def missing_copy_files(model: AIModel, copies: Iterable[AIModel]) -> dict[Path, Path]:
    """Files another copy of *model* has that *model*'s own copy lacks: {missing path: source}.

    Copies drift when files are added to one location only (e.g. copy-configs, or
    copies made before metadata files were copied with the model).  Only online
    locations are compared; the first copy holding a file is its source.
    """
    dest = location_of(model.path)
    if dest is None or not dest.online:
        return {}
    missing: dict[Path, Path] = {}
    for copy in copies:
        src = location_of(copy.path)
        if src is None or not src.online or src.root == dest.root:
            continue
        for f in (f for p in model_files(copy) for f in _files_in(p)):
            if src.contains(f):
                target = mirrored_path(f, dest)
                if target not in missing and not target.exists():
                    missing[target] = f
    return missing


def fill_missing(target: Path, source: Path) -> None:
    """Copy *source* to the absent *target*, never overwriting.

    Written to a temporary name and renamed into place, so an interrupted copy never
    leaves a truncated file that would look complete.
    """
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    shutil.copy2(source, partial)
    os.replace(partial, target)


def copy_model(model: AIModel, dest: Location) -> "subprocess.Popen[bytes]":
    """Start an rsync copying *model*'s files to their mirrored paths under *dest*.

    rsync's "/./" anchor keeps each file's path relative to its location root, so
    shards, companions and metadata land in the same layout as the source.  Re-running
    copies only what is missing.  stdout carries --info=progress2 output.
    """
    src = location_of(model.path)
    if src is None:
        raise ValueError(f"{model.path} is not inside a configured storage location")
    if src.root == dest.root:
        raise ValueError(f"{model.name} is already stored on {dest.label}")
    sources: list[str] = []
    for p in model_files(model):
        if not src.contains(p):
            raise ValueError(f"{p} is outside {src.label}; cannot mirror it to {dest.label}")
        anchored = f"{src.root}/./{p.relative_to(src.root)}"
        sources.append(anchored + "/" if p.is_dir() else anchored)
    dest.root.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        ["rsync", "-ahR", "--info=progress2", "--exclude=.git", *sources, f"{dest.root}/"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeletionPlan:
    """What deleting a model removes: shown for confirmation, then executed as-is."""
    entries: tuple[AIModel, ...]   # catalog entries removed (all sharing the model's path)
    files:   tuple[Path, ...]      # individual files removed
    dirs:    tuple[Path, ...]      # directories removed with all their contents
    location: Location

    @property
    def paths(self) -> tuple[Path, ...]:
        return (*self.dirs, *self.files)

    def execute(self) -> None:
        for d in self.dirs:
            if d.exists():
                shutil.rmtree(d)
        for f in self.files:
            f.unlink(missing_ok=True)
        # Prune directories left empty, up to (never including) the location root.
        for p in self.paths:
            parent = p.parent
            while parent != self.location.root and self.location.contains(parent):
                try:
                    parent.rmdir()
                except OSError:
                    break  # not empty (or already gone)
                parent = parent.parent


def plan_deletion(model: AIModel, catalog: "list[AIModel]") -> DeletionPlan:
    """Work out which files and catalog entries deleting *model* removes.

    Entries sharing the model's path (several formats in one directory) go with it.
    Companion files still referenced by other entries are kept.  A file model's
    directory is removed entirely once it holds no other weights and no other
    entry points inside it — unless it is a top-level directory of the location.
    Raises ValueError for paths outside the configured locations.
    """
    location = location_of(model.path)
    if location is None:
        raise ValueError(f"{model.path} is not inside a configured storage location")
    path = Path(model.path)
    entries = tuple(m for m in catalog if m.path == model.path)
    others = [m for m in catalog if m.path != model.path]
    if path.is_dir():
        return DeletionPlan(entries, (), (path,), location)

    referenced = {Path(p) for m in others for p in (m.path, *m.companion_paths)}
    removed = set(gguf_shard_paths(path)) | {
        Path(c) for m in entries for c in m.companion_paths if Path(c) not in referenced
    }
    directory = path.parent
    remaining = [
        f for f in directory.iterdir()
        if f not in removed and not f.name.startswith(".")
        and (f.is_dir() or f.suffix.lower() in _WEIGHT_SUFFIXES)
    ] if directory.is_dir() else []
    nested_entry = any(Path(m.path).is_relative_to(directory) for m in others)
    top_level = len(directory.relative_to(location.root).parts) < 2
    if not remaining and not nested_entry and not top_level:
        outside = tuple(sorted(f for f in removed if not f.is_relative_to(directory)))
        return DeletionPlan(entries, outside, (directory,), location)
    return DeletionPlan(entries, tuple(sorted(removed)), (), location)
