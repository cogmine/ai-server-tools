"""ingest.py — Tools for populating and enriching models_local.json from disk.

Subcommands:
  copy-configs <scan_root> [--dry-run]
      Scan scan_root (and the same directory under other storage locations) for GGUF bundle
      directories that lack a config.json.  For each, search for a matching
      original (non-GGUF) model and copy its config/tokenizer files across.
      Run this before ingesting a new GGUF bundle so that 'ingest' can read
      accurate context-window and architecture metadata from it.

  copy-mmproj <scan_root> [--dry-run]
      Scan scan_root for GGUF bundle directories that lack an mmproj-*.gguf
      file.  For each, search for a matching original model directory and copy
      its mmproj file into the bundle root.  Run this before ingesting (or
      refreshing) vision GGUFs so that mmproj_path is populated automatically.

  ingest <scan_root> [--ai-type text_gen|image_gen] [--all] [--dry-run]
      Scan scan_root for models not already catalogued in models_local.json
      and add them.  Without --all, stops after the first new model found
      (useful for testing detection before committing to a bulk import).
      With --all, ingests every uncatalogued model in the directory.

      For image_gen, each .safetensors file in the directory becomes its own
      catalog entry.  Format is inferred from the filename (fp8/e4m3fn → fp8,
      flux → bf16, everything else → fp16).

  ingest --path <path> [--ai-type text_gen|image_gen] [--dry-run]
      Add a single specific model by absolute path.

Usage:
    python -m launcher.ingest copy-configs /mnt/models/text-generation [--dry-run]
    python -m launcher.ingest copy-mmproj /mnt/models/text-generation [--dry-run]
    python -m launcher.ingest ingest /mnt/models/text-generation [--dry-run]
    python -m launcher.ingest ingest /mnt/models/text-generation --all [--dry-run]
    python -m launcher.ingest ingest --path /mnt/models/text-generation/MyModel [--dry-run]
    python -m launcher.ingest ingest /mnt/models/text-to-image/checkpoints --ai-type image_gen --all
    python -m launcher.ingest ingest /mnt/models/text-to-image/flux --ai-type image_gen --all
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from launcher.gguf_header import arch_int, embedded_mtp_layers, read_gguf_metadata
from launcher.models import GGUF_QUANTS_LONGEST_FIRST, gguf_shard_paths, gguf_split_info
from launcher.storage import LOCATIONS, METADATA_FILES, location_of, model_metadata_dir, relocated_entry

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_CATALOG_PATH = Path(__file__).parent / "user_data" / "models_local.json"



# ---------------------------------------------------------------------------
# Catalog I/O
# ---------------------------------------------------------------------------

def _load_catalog(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _known_path_formats(catalog: dict[str, Any]) -> set[tuple[str, str]]:
    """Return the set of (path, format) pairs already present in the catalog."""
    return {(m["path"], m.get("format", "")) for m in catalog.get("models", [])}


def _save_catalog(path: Path, catalog: dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(catalog, f, indent=2)
        f.write("\n")


# ---------------------------------------------------------------------------
# Model directory scanning
# ---------------------------------------------------------------------------

def _looks_like_model_dir(path: Path) -> bool:
    """True if the directory contains model weight files or a config.json."""
    if (path / "config.json").exists():
        return True
    for ext in (".safetensors", ".bin", ".gguf"):
        if any(path.glob(f"*{ext}")):
            return True
    return False


# GGUF files shipped inside a bundle that are not standalone models:
# multimodal projectors and multi-token-prediction draft heads.
_COMPANION_GGUF_PREFIXES = ("mmproj", "mtp-")


def _is_companion_gguf(path: Path) -> bool:
    return path.stem.lower().startswith(_COMPANION_GGUF_PREFIXES)


def _iter_candidates(scan_dir: Path):
    """Yield (path, format) candidate pairs from scan_dir in sorted order.

    Yields:
      - (gguf_file, "gguf") for each .gguf file inside a GGUF bundle directory.
        Scans two levels deep: flat bundles (entry/*.gguf) and per-quant-subdir
        bundles (entry/<quant>/*.gguf), so each quant variant becomes its own
        catalog entry.
      - (hf_dir, fmt) for each format detected in an HF model directory.
        Directories with both safetensors and pytorch_bin weights yield one pair
        per format so each becomes a separate catalog entry.
      - (gguf_file, "gguf") for top-level .gguf files (unusual but handled).
    """
    if not scan_dir.exists():
        print(f"[ingest] Directory not found: {scan_dir}", file=sys.stderr)
        return

    for entry in sorted(scan_dir.iterdir()):
        if entry.name.startswith("."):
            continue
        if entry.is_file():
            if entry.suffix.lower() == ".gguf" and not _is_companion_gguf(entry):
                info = gguf_split_info(entry)
                if info is None or info[1] == 1:
                    yield entry, "gguf"
        elif entry.is_dir():
            # Check for GGUF files one level deep (flat bundle: entry/*.gguf,
            # or per-quant subdirs: entry/<quant>/*.gguf).  Avoids rglob, which
            # would recursively traverse large HF model directories on slow storage.
            gguf_files = sorted(entry.glob("*.gguf"))
            if not gguf_files:
                for subdir in sorted(entry.iterdir()):
                    if subdir.is_dir() and not subdir.name.startswith("."):
                        gguf_files.extend(sorted(subdir.glob("*.gguf")))
            if gguf_files:
                for gguf_file in _primary_gguf_files(gguf_files):
                    if not _is_companion_gguf(gguf_file):
                        yield gguf_file, "gguf"
            elif _looks_like_model_dir(entry):
                cfg = _read_hf_config(entry)
                for fmt in _detect_all_hf_formats(entry, cfg):
                    yield entry, fmt
            # else: skip directories that don't look like model dirs


def find_new_model(
    scan_dir: Path, known: set[tuple[str, str]]
) -> tuple[Path, str] | None:
    """Return the first candidate (path, format) not already present in the catalog."""
    for candidate, fmt in _iter_candidates(scan_dir):
        if (str(candidate), fmt) not in known:
            return candidate, fmt
    return None


def find_all_new_models(
    scan_dir: Path, known: set[tuple[str, str]]
) -> list[tuple[Path, str]]:
    """Return all candidate (path, format) pairs not already present in the catalog."""
    return [
        (c, f) for c, f in _iter_candidates(scan_dir)
        if (str(c), f) not in known
    ]


# ---------------------------------------------------------------------------
# Size helpers
# ---------------------------------------------------------------------------

def _file_size_gb(path: Path) -> float:
    try:
        return round(path.stat().st_size / 1_000_000_000, 1)
    except OSError:
        return 0.0


# Minimum byte size for a file to be treated as a real model weight shard.
# Safetensors/bin stubs and placeholder files are typically a few hundred bytes;
# any genuine weight slice will be far larger than 1 MB.
_MIN_WEIGHT_FILE_BYTES = 1_000_000

# Extensions that constitute model weight files.  Only these are counted for
# size_gb — this excludes .git, config JSON, README, licence files, etc.
_MODEL_WEIGHT_SUFFIXES: frozenset[str] = frozenset({
    ".safetensors",
    ".bin",
    ".gguf",
    ".pt",
    ".pth",
    ".ggml",
})

# Maps ModelFormat value → glob patterns for the files that hold the weights.
# Used to count only the relevant files when a directory has mixed content.
# For safetensors formats this is a simple extension glob; for pytorch_bin we
# use HF shard naming patterns (pytorch_model*.bin, model*.bin) so that
# unrelated .bin files (GGML artefacts, quantization helpers, etc.) in the
# same directory are excluded from the size calculation.
_FORMAT_WEIGHT_GLOBS: dict[str, list[str]] = {
    "safetensors_bf16": ["*.safetensors"],
    "safetensors_fp16": ["*.safetensors"],
    "safetensors_fp8":  ["*.safetensors"],
    "gptq":             ["*.safetensors"],
    "awq":              ["*.safetensors"],
    "exl2":             ["*.safetensors"],
    "int8_bnb":         ["*.safetensors"],
    # HF shard naming conventions only — excludes GGML/GGUF/custom .bin artefacts.
    "pytorch_bin":      ["pytorch_model*.bin", "model*.bin", "*.pt", "*.pth"],
}

# Fractional slack allowed between two index files claiming to describe the same
# weights.  Real duplicates differ only by tensor-packing overhead (parts per
# million); anything larger is two genuinely different sets.
_DUPLICATE_TOTAL_TOLERANCE = 0.01


def _duplicate_weight_sets(model_dir: Path) -> bool:
    """True if the dir ships one set of weights twice, as HF and consolidated shards.

    Mistral repos publish `model-*.safetensors` alongside a native
    `consolidated-*.safetensors` set; both index files then declare the same
    weights.  The declared totals are the reliable signal — file counts differ
    between the two layouts — but they rarely agree to the byte, since the two
    layouts pack tensors differently, so they are compared within a tolerance
    that stays far below the 2x being guarded against.  The HF set must also be
    complete on disk, since sizing from a partially-downloaded set would
    understate the model.
    """
    hf_index = model_dir / "model.safetensors.index.json"
    consolidated_index = model_dir / "consolidated.safetensors.index.json"
    if not (hf_index.exists() and consolidated_index.exists()):
        return False
    try:
        with open(hf_index) as f:
            hf = json.load(f)
        with open(consolidated_index) as f:
            consolidated = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    hf_total = (hf.get("metadata") or {}).get("total_size")
    consolidated_total = (consolidated.get("metadata") or {}).get("total_size")
    if not (isinstance(hf_total, int) and isinstance(consolidated_total, int)):
        return False
    if hf_total <= 0 or consolidated_total <= 0:
        return False
    largest = max(hf_total, consolidated_total)
    if abs(hf_total - consolidated_total) / largest > _DUPLICATE_TOTAL_TOLERANCE:
        return False

    shards = set((hf.get("weight_map") or {}).values())
    missing = [s for s in shards if not (model_dir / s).exists()]
    if missing:
        print(
            f"[ingest] Warning: {model_dir.name} carries duplicate weight sets but "
            f"{len(missing)} of {len(shards)} HF shards are missing — sizing from all "
            "weight files instead.",
            file=sys.stderr,
        )
        return False
    return True


def _weight_globs_for(model_dir: Path, fmt: str) -> list[str] | None:
    """Weight-file globs for *fmt*, narrowed to avoid double-counting duplicates.

    When a directory holds the same weights as both HF and consolidated shard
    sets, only the HF shards are counted so size_gb is not doubled.
    """
    globs = _FORMAT_WEIGHT_GLOBS.get(fmt)
    if globs == ["*.safetensors"] and _duplicate_weight_sets(model_dir):
        return ["model-*.safetensors", "model.safetensors"]
    return globs


def _dir_size_gb(path: Path, include_globs: list[str] | None = None) -> float:
    """Sum model weight file sizes under path; return GB rounded to 1 d.p.

    When include_globs is provided (from _FORMAT_WEIGHT_GLOBS), only files
    whose name matches at least one fnmatch glob pattern are counted.  This
    lets pytorch_bin count only HF shard files (pytorch_model*.bin, model*.bin)
    and exclude unrelated .bin artefacts (GGML files, custom quants, etc.).
    Without include_globs, all files with extensions in _MODEL_WEIGHT_SUFFIXES
    are counted.  Hidden directories (name starting with '.') are skipped.
    """
    total = 0
    dirs = [path]
    while dirs:
        current = dirs.pop()
        try:
            entries = list(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.name.startswith("."):
                continue          # skip .git and other hidden dirs/files
            if entry.is_dir():
                dirs.append(entry)
            elif entry.is_file():
                if include_globs is not None:
                    if not any(fnmatch.fnmatch(entry.name, g) for g in include_globs):
                        continue
                elif entry.suffix.lower() not in _MODEL_WEIGHT_SUFFIXES:
                    continue
                try:
                    total += entry.stat().st_size
                except OSError:
                    pass
    return round(total / 1_000_000_000, 1)


# ---------------------------------------------------------------------------
# GGUF helpers
# ---------------------------------------------------------------------------

def _extract_gguf_quant(filename: str) -> str:
    """Extract the GGUF quantisation label from a filename, or '' if not found.

    A filename may carry more than one recognised label — a Q8_0 quant of a BF16
    upcast is conventionally named '<model>-bf16.Q8_0.gguf'.  The trailing label
    is the real quantisation, so the rightmost match wins; ties at the same
    position (Q4_K matching inside Q4_K_M) are broken by the longer label.
    """
    upper = filename.upper()
    best_key = (-1, -1)
    best_value = ""
    for q in GGUF_QUANTS_LONGEST_FIRST:
        for match in re.finditer(
            r"[.\-_]" + re.escape(q.value) + r"(?:[.\-_]|$)", upper
        ):
            key = (match.start(), len(q.value))
            if key > best_key:
                best_key, best_value = key, q.value
    return best_value


def _quant_slug(quant: str) -> str:
    """'Q4_K_M' → 'q4km', 'Q8_0' → 'q8', 'IQ3_XXS' → 'iq3xxs'."""
    s = quant.lower().replace("_", "")
    # Base quants end in a digit after stripping underscores (q80, q40, q50).
    # Strip trailing '0' so Q8_0 → q8, matching the existing catalog convention.
    if re.match(r"^q\d+0$", s):
        s = s[:-1]
    return s


# llama.cpp split sets are one model; only the first shard becomes a candidate.
def _primary_gguf_files(gguf_files: list[Path]) -> list[Path]:
    """Drop non-first shards so each split set contributes a single candidate."""
    return [
        f for f in gguf_files
        if (info := gguf_split_info(f)) is None or info[1] == 1
    ]


def _gguf_size_gb(gguf_path: Path) -> float:
    """Size in GB of a GGUF model, summing every shard when it is a split set."""
    total_bytes = 0
    for shard in gguf_shard_paths(gguf_path):
        try:
            total_bytes += shard.stat().st_size
        except OSError:
            print(
                f"[ingest] Warning: missing GGUF shard {shard.name}", file=sys.stderr
            )
    return round(total_bytes / 1_000_000_000, 1)


# ---------------------------------------------------------------------------
# Name / ID helpers
# ---------------------------------------------------------------------------

def _dir_to_display_name(dirname: str) -> str:
    """Convert a directory/file name to a human-readable display string."""
    # Strip common format/provenance suffixes (case-insensitive comparison).
    # Chained suffixes (e.g. "...-BF16-GGUF") are stripped repeatedly until
    # none remain, so the quant tag doesn't linger and later get duplicated
    # by the quant_suffix appended in the caller.
    strip_suffixes = [
        "-GGUF", "_GGUF", "-BF16", "-FP16", "-FP8", "-INT8",
        "-EXL2", "-AWQ", "-GPTQ", "-hf",
        "-UD", "_UD",  # Unsloth Dynamic tag that prefixes the quant label
    ]
    while True:
        for suffix in strip_suffixes:
            if dirname.upper().endswith(suffix.upper()):
                dirname = dirname[: -len(suffix)]
                break
        else:
            break
    # Replace common separators with spaces, but preserve dots that are
    # part of a version number (digit.digit, e.g. V3.2, v0.1).
    name = re.sub(r"[-_]", " ", dirname)
    name = re.sub(r"(?<!\d)\.(?!\d)|(?<=\d)\.(?!\d)|(?<!\d)\.(?=\d)", " ", name)
    return re.sub(r" +", " ", name).strip()


def _slugify(s: str) -> str:
    """Produce a URL-safe lowercase slug from an arbitrary string."""
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


# ---------------------------------------------------------------------------
# HF directory detection
# ---------------------------------------------------------------------------

def _read_hf_config(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[ingest] Warning: could not read config.json: {exc}", file=sys.stderr)
    return {}


def _detect_all_hf_formats(model_dir: Path, cfg: dict[str, Any]) -> list[str]:
    """Return all distinct weight formats present in an HF model directory.

    Quantised checkpoints are always single-format.  Unquantised directories
    may contain both safetensors and pytorch_bin weights (e.g. a HF repo that
    ships both formats).  In that case both format strings are returned, with
    safetensors first.
    """
    quant_cfg = cfg.get("quantization_config") or {}
    # quant_type covers ExLlama/GPTQ/AWQ; quant_method covers HF-style tags and tool-specific
    # quantization schemes (vLLM fp8, bitsandbytes, modelopt).
    quant_type   = str(quant_cfg.get("quant_type",   "")).lower()
    quant_method = str(quant_cfg.get("quant_method", "")).lower()
    quant_algo   = str(quant_cfg.get("quant_algo",   "")).lower()  # modelopt uses this

    if quant_type == "gptq":
        return ["gptq"]
    if quant_type in ("awq", "gemm", "gemv"):
        return ["awq"]
    if quant_type == "exl2":
        return ["exl2"]
    # bitsandbytes int8: some models use quant_type, others use quant_method
    if quant_type in ("nf4", "int8", "bitsandbytes", "bnb") or quant_method in ("bitsandbytes", "bnb"):
        load_8bit = quant_cfg.get("_load_in_8bit") or quant_cfg.get("load_in_8bit")
        if load_8bit:
            return ["int8_bnb"]
    # FP8: vLLM-style (quant_method=fp8), type-level (quant_type=fp8), or ModelOpt-quantised
    # (quant_method=modelopt, quant_algo=fp8)
    if quant_method == "fp8" or quant_type == "fp8" or quant_algo == "fp8":
        return ["safetensors_fp8"]

    # torch_dtype may be at the top level, or nested inside text_config for multimodal models
    torch_dtype = str(
        cfg.get("torch_dtype")
        or (cfg.get("text_config") or {}).get("dtype", "")
    ).lower()
    if "float8" in torch_dtype or "fp8" in torch_dtype:
        return ["safetensors_fp8"]

    # Only count safetensors files that are genuine weight shards (not stub/placeholder
    # files that some repos ship alongside the real pytorch_bin weights).
    has_safetensors = any(
        e.stat().st_size >= _MIN_WEIGHT_FILE_BYTES
        for e in model_dir.glob("*.safetensors")
    )
    has_bin = any(model_dir.glob("pytorch_model*.bin")) or any(model_dir.glob("model*.bin"))

    # Detect GPTQ checkpoints that predate the quantization_config field in config.json.
    # Filenames containing '4bit' or '8bit' (adjacent to a word separator or boundary)
    # are a reliable signal for older GPTQ-quantised safetensors.  Only applied when no
    # pytorch_bin shards are present — if both exist the directory is treated as genuinely
    # mixed-format and both entries will be created.
    if has_safetensors and not quant_cfg and not has_bin:
        if any(
            re.search(r"(?:^|[-_.])(4bit|8bit|gptq)(?:[-_.]|$)", e.stem, re.IGNORECASE)
            for e in model_dir.glob("*.safetensors")
            if e.stat().st_size >= _MIN_WEIGHT_FILE_BYTES
        ):
            return ["gptq"]

    formats: list[str] = []

    if has_safetensors:
        if torch_dtype == "bfloat16":
            formats.append("safetensors_bf16")
        elif torch_dtype == "float16":
            formats.append("safetensors_fp16")
        elif torch_dtype == "float32":
            print(
                "[ingest] Warning: torch_dtype=float32 (full-precision weights). "
                "Format recorded as safetensors_bf16 for server compatibility.",
                file=sys.stderr,
            )
            formats.append("safetensors_bf16")
        else:
            formats.append("safetensors_bf16")  # dtype unspecified; BF16 is most common

    if has_bin:
        formats.append("pytorch_bin")

    return formats if formats else ["safetensors_bf16"]  # final fallback


def _detect_hf_format(model_dir: Path, cfg: dict[str, Any]) -> str:
    """Determine the primary ModelFormat value string for an HF model directory."""
    return _detect_all_hf_formats(model_dir, cfg)[0]


def _detect_ctx(cfg: dict[str, Any]) -> int:
    """Extract the native context window from an HF config.json.

    Checks the top-level config first, then falls back to text_config for
    multimodal models where language config is nested (e.g. Qwen3.5-MoE).

    When max_position_embeddings is present but model_max_length is strictly
    larger (and ≤ 2M tokens), model_max_length is preferred — this covers
    models like Cohere Command R+ where max_position_embeddings holds only the
    RoPE base (e.g. 8192) while model_max_length reflects the actual extended
    training context (e.g. 131072).
    """
    _CTX_KEYS = (
        "max_position_embeddings",
        "max_seq_len",
        "seq_length",
        "context_length",
        "n_positions",
    )
    _MAX_PLAUSIBLE = 2_000_000  # guard against sentinel values like 999999999

    def _best(section: dict[str, Any]) -> int:
        ctx = 0
        for key in _CTX_KEYS:
            val = section.get(key)
            if isinstance(val, int) and val > 0:
                ctx = val
                break
        if ctx > 0:
            ml = section.get("model_max_length")
            if isinstance(ml, int) and ctx < ml <= _MAX_PLAUSIBLE:
                ctx = ml
        return ctx

    ctx = _best(cfg)
    if ctx > 0:
        return ctx
    # Multimodal fallback: language config nested under text_config
    return _best(cfg.get("text_config") or {})


def _params_m_from_name(name: str) -> int:
    """Return parameter count in millions inferred from a model name / file stem.

    Accepts occurrences of a number followed immediately by 'b'/'B' (billions)
    or 'm'/'M' (millions) that is not part of a longer alphanumeric token.
    Returns 0 when no candidates are found, or when no single best candidate
    can be determined.

    Matches are ranked by the character immediately preceding the number:
      - rank 0 (best): separator character ('-', '_', '.') or start of string
      - rank 1: letter character (weaker but still valid — e.g. 'Alpacino30b')

    When multiple matches share the same best rank the result is ambiguous and
    0 is returned.  A single rank-0 match wins over any number of rank-1 ones.

    Examples that match:
      '123b'→123000, '70B'→70000, '1.5B'→1500, '350m'→350, 'Alpacino30b'→30000
      'NVIDIA-Nemotron-3-Super-120B-A12B'→120000  (120B rank-0 beats A12B rank-1)
      'Qwen3.5-122B-A10B'→122000                  (122B rank-0 beats A10B rank-1)

    Non-matches: 'model-10b-32k-8b' (two rank-0 matches → ambiguous),
                 '4.0bpw-quant' ('bpw' suffix blocked by lookahead).
    """
    _SEP = frozenset("-_.")

    def _rank(pos: int) -> int:
        if pos == 0:
            return 0
        return 0 if name[pos - 1] in _SEP else 1

    hits = [
        (_rank(m.start()), m)
        for m in re.finditer(r"(?<!\d)(\d+(?:\.\d+)?)([bBmM])(?![a-zA-Z0-9])", name)
    ]
    if not hits:
        return 0
    best_rank = min(r for r, _ in hits)
    best = [m for r, m in hits if r == best_rank]
    if len(best) != 1:
        return 0
    value_str, suffix = best[0].groups()
    value = float(value_str)
    if suffix in ("b", "B"):
        value *= 1000.0
    return round(value)


def _detect_params_m(cfg: dict[str, Any], name: str = "") -> int:
    """Return parameter count in millions from config.json, falling back to name.

    Reads `num_parameters` from config.json for an exact count.  When that key
    is absent (common for GGUF models without a bundled config), applies
    `_params_m_from_name` to `name` (directory name or file stem) as a
    best-effort fallback.
    """
    raw = cfg.get("num_parameters")
    if isinstance(raw, (int, float)) and raw > 0:
        return round(raw / 1e6)
    if name:
        return _params_m_from_name(name)
    return 0


def _detect_arch(cfg: dict[str, Any]) -> str:
    """Return the first architecture class name from config.json, or ''."""
    archs = cfg.get("architectures", [])
    return archs[0] if archs else ""


def _detect_moe(cfg: dict[str, Any]) -> tuple[int, int]:
    """Return (moe_experts, moe_active) from config.json; (0, 0) for dense models."""
    text_cfg: dict[str, Any] = cfg.get("text_config") or cfg
    # Handle HF naming (num_local_experts), DeepSeek naming (n_routed_experts),
    # and Qwen3 naming (num_experts — also nested under text_config).
    n_experts = (
        text_cfg.get("num_local_experts")
        or text_cfg.get("n_routed_experts")
        or text_cfg.get("num_experts")
    )
    n_active  = text_cfg.get("num_experts_per_tok")
    if isinstance(n_experts, int) and n_experts > 1:
        return n_experts, (n_active if isinstance(n_active, int) else 0)
    return 0, 0


def _detect_vision(cfg: dict[str, Any]) -> bool:
    """Return True if config.json indicates a vision/multimodal model."""
    return bool(cfg.get("vision_config"))


# config.json keys declaring multi-token-prediction layers (vLLM's names; the first
# is Qwen3.5 / Qwen4Exp, the second DeepSeek-V3, GLM, Qwen3-Next, Nemotron-H, …).
_MTP_LAYER_KEYS = ("mtp_num_hidden_layers", "num_nextn_predict_layers")


def _detect_hf_mtp(model_dir: Path, cfg: dict[str, Any]) -> bool:
    """Return True if the checkpoint declares MTP layers *and* ships their weights.

    Quantized, abliterated or converted copies often keep the config keys but drop
    the weights, so the weight names are checked too: either an ``mtp.*`` module or
    a layer index past ``num_hidden_layers`` (DeepSeek-style NextN layers).
    """
    text_cfg: dict[str, Any] = cfg.get("text_config") or {}
    def _get(key: str) -> Any:
        return text_cfg.get(key, cfg.get(key))
    if not any(int(_get(k) or 0) > 0 for k in _MTP_LAYER_KEYS):
        return False
    n_layers = int(_get("num_hidden_layers") or 0)
    nextn_prefix = f"layers.{n_layers}."
    return any(
        name.startswith("mtp.") or ".mtp." in name or (n_layers and nextn_prefix in name)
        for name in _hf_weight_names(model_dir)
    )


def _hf_weight_names(model_dir: Path) -> list[str]:
    """Tensor names of an HF checkpoint: from its safetensors index, else the shard headers."""
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        try:
            return list(json.loads(index.read_text(encoding="utf-8")).get("weight_map", {}))
        except (OSError, json.JSONDecodeError):
            return []
    names: list[str] = []
    for shard in sorted(model_dir.glob("*.safetensors")):
        try:
            with open(shard, "rb") as f:
                header_len = int.from_bytes(f.read(8), "little")
                names += [k for k in json.loads(f.read(header_len)) if k != "__metadata__"]
        except (OSError, ValueError):
            continue
    return names


def _detect_trust_remote_code(cfg: dict[str, Any]) -> bool:
    """Return True if config.json signals that --trust-remote-code is required.

    Models that define custom code via auto_map require trust_remote_code so
    that transformers will load the custom class definitions.
    """
    return bool(cfg.get("auto_map"))


def _detect_fp8_min_sm(cfg: dict[str, Any]) -> int:
    """Return the minimum sm level required by an FP8 quantization scheme, or 0.

    Block-wise FP8 dequantizes at runtime and works on sm_86 (RTX 3090).
    Static / non-dequantize FP8 requires sm_89+ (Ada Lovelace and newer).
    """
    qcfg = cfg.get("quantization_config") or {}
    if qcfg.get("quant_method") != "fp8":
        return 0
    deq   = qcfg.get("dequantize", True)
    act   = qcfg.get("activation_scheme", "")
    block = qcfg.get("weight_block_size")
    if not deq and act == "static":
        return 89   # static FP8 — requires sm_89+
    if block:
        return 0    # block-wise FP8 — sm_86 OK
    return 0


_FORMAT_SHORT: dict[str, str] = {
    "safetensors_bf16": "bf16",
    "safetensors_fp16": "fp16",
    "safetensors_fp8":  "fp8",
    "int8_bnb":         "int8",
    "pytorch_bin":      "bin",
    "awq":              "awq",
    "gptq":             "gptq",
    "exl2":             "exl2",
}


# ---------------------------------------------------------------------------
# Ctx inference from sibling catalog entries (GGUF)
# ---------------------------------------------------------------------------

def _ctx_from_gguf_siblings(gguf_path: Path, catalog: dict[str, Any]) -> int:
    """For a GGUF file, try to borrow ctx from a sibling already in the catalog.

    Looks for catalog entries whose path shares the same parent directory
    (same GGUF bundle) and already has a known ctx > 0.
    """
    gguf_dir = str(gguf_path.parent)
    for entry in catalog.get("models", []):
        if (
            entry.get("format") == "gguf"
            and str(Path(entry["path"]).parent) == gguf_dir
            and int(entry.get("ctx", 0)) > 0
        ):
            return int(entry["ctx"])
    return 0


# ---------------------------------------------------------------------------
# copy-configs helpers
# ---------------------------------------------------------------------------

def _all_roots(scan_root: Path) -> list[Path]:
    """Return scan_root plus the same leaf directory under every other storage location.

    For example, if scan_root is /mnt/nas/ai_models/text-generation, also
    includes ~/fast_models/text-generation, /mnt/sda1/models/text-generation,
    etc. from the configured locations.
    """
    roots: list[Path] = []
    if scan_root.exists():
        roots.append(scan_root)
    subdir = scan_root.name  # e.g. "text-generation"
    for loc in LOCATIONS:
        candidate = loc.root / subdir
        if candidate.exists() and candidate not in roots:
            roots.append(candidate)
    return roots


def _normalise_for_match(name: str) -> str:
    """Reduce a directory name to a normalised token string for fuzzy matching.

    Steps:
    - Lowercase
    - Strip known vendor prefixes (e.g. "TheDrummer_", "nvidia.")
    - Strip known format/variant suffixes (-GGUF, -hf, etc.)
    - Strip GGUF quant suffixes (.Q8_0, -Q4_K_M, IQ3_XXS, etc.)
    - Strip trailing version tags (-v0.1, -v2, etc.)
    - Replace all non-alphanumeric characters with spaces, collapse whitespace
    """
    s = name.lower()
    # Remove a single leading vendor prefix of form "word_" or "word."
    s = re.sub(r"^[a-z0-9]+[_.]", "", s)
    # Strip format/provenance suffixes
    s = re.sub(
        r"[-_](gguf|bf16|fp16|fp8|int8|exl2|awq|gptq|hf|instruct|chat)$",
        "", s, flags=re.IGNORECASE,
    )
    # Strip GGUF quant label suffixes (.Q8_0, -Q4_K_M, .IQ3_XXS, etc.)
    s = re.sub(r"[._-][iq]\w+$", "", s, flags=re.IGNORECASE)
    # Strip trailing version tags (-v0.1, -v2, _v3, etc.)
    s = re.sub(r"[-_]v\d[\d.]*$", "", s, flags=re.IGNORECASE)
    # Normalise separators to spaces
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return s.strip()


def _names_match(gguf_norm: str, entry_norm: str) -> bool:
    """True if the two normalised names refer to the same model.

    Accepts exact match OR one being a prefix of the other (handles cases
    where the GGUF bundle name has extra version/quant tokens).
    """
    if gguf_norm == entry_norm:
        return True
    # Allow one to be a prefix of the other (split on spaces to avoid partial-word matches)
    gguf_tokens = gguf_norm.split()
    entry_tokens = entry_norm.split()
    shorter, longer = (
        (gguf_tokens, entry_tokens) if len(gguf_tokens) <= len(entry_tokens)
        else (entry_tokens, gguf_tokens)
    )
    return longer[:len(shorter)] == shorter


def _find_original_model(gguf_dir_name: str, all_roots: list[Path]) -> Path | None:
    """Search all roots for the original (non-GGUF) model directory.

    First tries an exact suffix-stripped name match (fast path), then falls
    back to normalised token prefix matching against all candidate dirs.
    """
    # --- Pass 1: exact suffix-stripped names (fast path) ---
    exact_candidates: list[str] = []
    stripped = re.sub(r"[-_]GGUF$", "", gguf_dir_name, flags=re.IGNORECASE)
    if stripped and stripped != gguf_dir_name:
        exact_candidates.append(stripped)
    stripped2 = re.sub(r"\.[IQ]\w+$", "", gguf_dir_name, flags=re.IGNORECASE)
    if stripped2 and stripped2 != gguf_dir_name and stripped2 not in exact_candidates:
        exact_candidates.append(stripped2)

    for candidate_name in exact_candidates:
        for root in all_roots:
            candidate = root / candidate_name
            if candidate.is_dir() and _looks_like_model_dir(candidate):
                return candidate

    # --- Pass 2: normalised token match against all dirs in all roots ---
    gguf_norm = _normalise_for_match(gguf_dir_name)
    if not gguf_norm:
        return None

    for root in all_roots:
        try:
            entries = sorted(root.iterdir())
        except OSError:
            continue
        for entry in entries:
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            # Skip GGUF bundle dirs and the entry itself
            if "gguf" in entry.name.lower() or entry.name == gguf_dir_name:
                continue
            entry_norm = _normalise_for_match(entry.name)
            if entry_norm and _names_match(gguf_norm, entry_norm) and _looks_like_model_dir(entry):
                return entry

    return None


def _copy_missing_configs(src_dir: Path, dst_dir: Path, dry_run: bool) -> list[str]:
    """Copy recognised config files present in src_dir but absent in dst_dir.

    Returns a list of filenames that were (or would be, in dry-run) copied.
    """
    copied: list[str] = []
    for filename in METADATA_FILES:
        src = src_dir / filename
        dst = dst_dir / filename
        if src.exists() and not dst.exists():
            if dry_run:
                print(f"    would copy  {filename}")
            else:
                shutil.copy2(src, dst)
                print(f"    copied      {filename}")
            copied.append(filename)
    return copied


def copy_model_metadata(src_dir: Path, dst_dir: Path) -> list[str]:
    """Copy standard model config/tokenizer files that are missing from *dst_dir*."""
    return _copy_missing_configs(src_dir, dst_dir, dry_run=False)


def do_copy_configs(scan_root: Path, dry_run: bool) -> None:
    """Copy config files from original model dirs into their GGUF bundle directories.

    Scans scan_root and the same leaf directory under every other storage location.
    For each directory that contains .gguf files and lacks a config.json, derives
    the original model name, locates it in any root, and copies the recognised
    config/tokenizer files.
    """
    all_roots = _all_roots(scan_root)
    if not all_roots:
        print(f"[copy-configs] No accessible roots found under {scan_root}.", file=sys.stderr)
        return

    processed = 0
    seen: set[str] = set()  # deduplicate across roots (same bundle may appear in multiple)
    for root in all_roots:
        for entry in sorted(root.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            if entry.name in seen:
                continue
            # Mirror _iter_candidates: check flat layout (entry/*.gguf) and
            # one level of per-quant subdirs (entry/<quant>/*.gguf).
            gguf_files = list(entry.glob("*.gguf"))
            if not gguf_files:
                for subdir in sorted(entry.iterdir()):
                    if subdir.is_dir() and not subdir.name.startswith("."):
                        gguf_files.extend(subdir.glob("*.gguf"))
                        if gguf_files:
                            break  # one hit is enough to confirm it's a GGUF bundle
            if not gguf_files:
                continue  # not a GGUF bundle dir
            seen.add(entry.name)

            if (entry / "config.json").exists():
                print(f"  {entry.name}: already has config.json — skipping")
                continue

            original = _find_original_model(entry.name, all_roots)
            if original is None:
                print(f"  {entry.name}: original model not found — skipping")
                continue

            print(f"  {entry.name}  ←  {original}")
            copied = _copy_missing_configs(original, entry, dry_run)
            if not copied:
                print("    (no config files found in source — nothing copied)")
            processed += 1

    if processed == 0:
        print(
            "[copy-configs] Nothing to do — all GGUF directories either already have "
            "config.json or no matching original was found."
        )
    elif dry_run:
        print(f"\n[copy-configs] dry-run complete — {processed} director(ies) would be updated.")
    else:
        print(f"\n[copy-configs] Done — {processed} director(ies) updated.")


def _find_mmproj(bundle_root: Path) -> Path | None:
    """Return the first mmproj-*.gguf file found in bundle_root, or None."""
    matches = sorted(bundle_root.glob("mmproj-*.gguf"))
    return matches[0] if matches else None


def _find_mtp(bundle_root: Path, quant: str) -> Path | None:
    """Return the mtp-*.gguf draft head in bundle_root, preferring one tagged with *quant*."""
    matches = sorted(bundle_root.glob("mtp-*.gguf"))
    for m in matches:
        if quant and _extract_gguf_quant(m.name) == quant:
            return m
    return matches[0] if matches else None


def do_copy_mmproj(scan_root: Path, dry_run: bool) -> None:
    """Copy mmproj files from original model dirs into their GGUF bundle directories.

    Scans scan_root and the same directory under other storage locations for GGUF bundle directories that
    lack an mmproj-*.gguf file.  For each, searches for a matching original
    model directory and copies its mmproj file into the bundle root.
    """
    all_roots = _all_roots(scan_root)
    if not all_roots:
        print(f"[copy-mmproj] No accessible roots found under {scan_root}.", file=sys.stderr)
        return

    processed = 0
    seen: set[str] = set()
    for root in all_roots:
        for entry in sorted(root.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            if entry.name in seen:
                continue
            # Confirm it is a GGUF bundle (flat or per-quant-subdir layout).
            gguf_files = list(entry.glob("*.gguf"))
            if not gguf_files:
                for subdir in sorted(entry.iterdir()):
                    if subdir.is_dir() and not subdir.name.startswith("."):
                        gguf_files.extend(subdir.glob("*.gguf"))
                        if gguf_files:
                            break
            if not gguf_files:
                continue
            seen.add(entry.name)

            if _find_mmproj(entry):
                print(f"  {entry.name}: already has mmproj — skipping")
                continue

            original = _find_original_model(entry.name, all_roots)
            if original is None:
                print(f"  {entry.name}: original model not found — skipping")
                continue

            mmproj_found = sorted(original.glob("mmproj-*.gguf"))
            if not mmproj_found:
                # Not a vision model — silently skip.
                continue

            print(f"  {entry.name}  ←  {original}")
            for src in mmproj_found:
                dst = entry / src.name
                if dry_run:
                    print(f"    would copy  {src.name}")
                else:
                    shutil.copy2(src, dst)
                    print(f"    copied      {src.name}")
            processed += 1

    if processed == 0:
        print(
            "[copy-mmproj] Nothing to do — all GGUF directories either already have "
            "an mmproj file, no matching original was found, or no originals contain mmproj files."
        )
    elif dry_run:
        print(f"\n[copy-mmproj] dry-run complete — {processed} director(ies) would be updated.")
    else:
        print(f"\n[copy-mmproj] Done — {processed} director(ies) updated.")


# ---------------------------------------------------------------------------
# Entry builders
# ---------------------------------------------------------------------------

def detect_gguf(gguf_path: Path, catalog: dict[str, Any]) -> dict[str, Any]:
    """Build a catalog entry dict for a GGUF file.

    If the parent directory contains a config.json (e.g. copied from the
    original model by the copy-configs subcommand), it is used to populate
    ctx and note accurately.  Otherwise ctx and MoE counts come from the GGUF
    header (ctx falling back to a sibling catalog entry), and vision is inferred
    from an mmproj file in the bundle.
    """
    quant = _extract_gguf_quant(gguf_path.name)

    # Display name: prefer parent dir name (minus -GGUF suffix) over file stem.
    # Strip the split-shard suffix and quant label first so that names like
    # "Model-GGUF-Q5_K_S" reduce to "Model" rather than keeping the tags, which
    # would then be duplicated by the quant_suffix appended below.
    parent = gguf_path.parent
    split_info = gguf_split_info(gguf_path)
    base_name = (
        parent.name if "gguf" in parent.name.lower()
        else (split_info[0] if split_info else gguf_path.stem)
    )
    # Bundle names may carry the quant label and a GGUF tag mid-name, followed by
    # a variant tag ("Model-GGUF-Q8_0-MTP"), so both are removed wherever they occur.
    tags = [re.escape(quant)] if quant else []
    tags.append("GGUF")
    base_name = re.sub(
        r"[.\-_](?:" + "|".join(tags) + r")(?=[.\-_]|$)", "", base_name, flags=re.IGNORECASE
    )
    display_name = _dir_to_display_name(base_name)

    quant_suffix = f"-{_quant_slug(quant)}" if quant else ""
    entry_id = f"{_slugify(display_name)}{quant_suffix}"

    # Collision guard: if this ID already belongs to a different path (e.g. a
    # safetensors_bf16 dir whose format slug matches this quant slug), append
    # "-gguf" to disambiguate rather than silently creating a duplicate key.
    _id_to_path = {m["id"]: m["path"] for m in catalog.get("models", [])}
    if entry_id in _id_to_path and _id_to_path[entry_id] != str(gguf_path):
        entry_id = f"{entry_id}-gguf"

    try:
        header = read_gguf_metadata(gguf_path)
    except (OSError, ValueError) as exc:
        print(f"[ingest] Warning: could not read GGUF header: {exc}", file=sys.stderr)
        header = {}

    # Use config.json from the GGUF bundle (placed by copy-configs or a conversion).
    cfg = _read_hf_config(model_metadata_dir(gguf_path))
    if cfg:
        ctx           = _detect_ctx(cfg)
        params_m      = _detect_params_m(cfg, gguf_path.stem)
        arch          = _detect_arch(cfg)
        moe_experts, moe_active = _detect_moe(cfg)
        vision        = _detect_vision(cfg)
        trust_rc      = _detect_trust_remote_code(cfg)
    else:
        # Without config.json the GGUF header is the best source; arch stays empty because
        # the catalog records HF architecture class names, which GGUF does not carry.
        ctx           = arch_int(header, "context_length") or _ctx_from_gguf_siblings(gguf_path, catalog)
        params_m      = _detect_params_m({}, gguf_path.stem)
        arch          = ""
        moe_experts   = arch_int(header, "expert_count")
        moe_active    = arch_int(header, "expert_used_count")
        vision        = _find_mmproj(parent) is not None
        trust_rc      = False

    # Only search for an mmproj file when this is a vision model — non-vision GGUFs
    # never have one, and glob() on slow storage is expensive at catalog-refresh scale.
    mmproj_path = (_find_mmproj(parent) or _find_mmproj(parent.parent)) if vision else None
    mtp_path = _find_mtp(parent, quant) or _find_mtp(parent.parent, quant)
    mtp_embedded = embedded_mtp_layers(header) > 0

    entry: dict[str, Any] = {
        "id": entry_id,
        "name": display_name,
        "path": str(gguf_path),
        "format": "gguf",
        "ai_types": ["text_gen"],
        "ctx": ctx,
        "size_gb": _gguf_size_gb(gguf_path),
        "server_status": {},
    }
    if quant:         entry["quant"]        = quant
    if params_m:      entry["params_m"]     = params_m
    if arch:          entry["arch"]         = arch
    if moe_experts:   entry["moe_experts"]  = moe_experts
    if moe_active:    entry["moe_active"]   = moe_active
    if vision:        entry["vision"]       = True
    if mmproj_path:   entry["mmproj_path"]  = str(mmproj_path)
    if mtp_path:      entry["mtp_path"]     = str(mtp_path)
    if mtp_embedded:  entry["mtp_embedded"] = True
    if trust_rc:      entry["trust_remote_code"] = True
    return entry


def detect_hf_dir(
    model_dir: Path, catalog: dict[str, Any], fmt: str | None = None
) -> dict[str, Any]:
    """Build a catalog entry dict for an HF model directory.

    When *fmt* is provided it is used directly; otherwise the format is
    auto-detected via ``_detect_hf_format``.  Pass an explicit format when
    ingesting one of several formats from a mixed-content directory so that
    the correct weight files are sized and the correct ID suffix is applied.
    """
    cfg           = _read_hf_config(model_dir)
    if fmt is None:
        fmt = _detect_hf_format(model_dir, cfg)
    ctx           = _detect_ctx(cfg)
    params_m      = _detect_params_m(cfg, model_dir.name)
    arch          = _detect_arch(cfg)
    moe_experts, moe_active = _detect_moe(cfg)
    vision        = _detect_vision(cfg)
    mtp_embedded  = _detect_hf_mtp(model_dir, cfg)
    trust_rc      = _detect_trust_remote_code(cfg)
    fp8_min_sm    = _detect_fp8_min_sm(cfg)
    display_name  = _dir_to_display_name(model_dir.name)
    fmt_short     = _FORMAT_SHORT.get(fmt, fmt)
    entry_id      = f"{_slugify(display_name)}-{fmt_short}"

    # Collision guard: if entry_id already belongs to a different path, fall back to
    # the raw directory name (preserving distinguishing suffixes like -hf, -fp16, etc.).
    _id_to_path = {m["id"]: m["path"] for m in catalog.get("models", [])}
    if entry_id in _id_to_path and _id_to_path[entry_id] != str(model_dir):
        entry_id = f"{_slugify(model_dir.name)}-{fmt_short}"

    print(
        "[ingest] Computing directory size (may take a moment on slow storage) …",
        file=sys.stderr,
    )
    size_globs = _weight_globs_for(model_dir, fmt)
    size_gb = _dir_size_gb(model_dir, include_globs=size_globs)

    entry: dict[str, Any] = {
        "id": entry_id,
        "name": display_name,
        "path": str(model_dir),
        "format": fmt,
        "ai_types": ["text_gen"],
        "ctx": ctx,
        "size_gb": size_gb,
        "server_status": {},
    }
    if params_m:    entry["params_m"]    = params_m
    if arch:        entry["arch"]        = arch
    if moe_experts: entry["moe_experts"] = moe_experts
    if moe_active:  entry["moe_active"]  = moe_active
    if vision:      entry["vision"]      = True
    if mtp_embedded: entry["mtp_embedded"] = True
    if trust_rc:    entry["trust_remote_code"] = True
    if fp8_min_sm:  entry["min_sm"]      = fp8_min_sm
    return entry


def detect_text_gen(
    path: Path, catalog: dict[str, Any], fmt: str | None = None
) -> dict[str, Any]:
    """Dispatch to the appropriate detector based on path type.

    *fmt* is forwarded to ``detect_hf_dir`` for directory paths and ignored
    for GGUF files (the format is always "gguf" for those).
    """
    if path.is_file() and path.suffix.lower() == ".gguf":
        return detect_gguf(path, catalog)
    if path.is_dir():
        gguf_files = sorted(path.glob("*.gguf"))
        if gguf_files and fmt is None:
            file_list = ", ".join(f.name for f in gguf_files[:3])
            if len(gguf_files) > 3:
                file_list += f", … ({len(gguf_files)} total)"
            raise ValueError(
                f"Directory contains GGUF files — target the specific .gguf file instead: {file_list}"
            )
        return detect_hf_dir(path, catalog, fmt=fmt)
    raise ValueError(f"Unrecognised model path (not a .gguf file or directory): {path}")


# ---------------------------------------------------------------------------
# Image generation model detection
# ---------------------------------------------------------------------------

def _detect_imagegen_format(filename: str) -> str:
    """Infer ModelFormat value string for a single-file image gen model from its filename."""
    name = filename.lower()
    if "fp8" in name or "e4m3fn" in name:
        return "safetensors_fp8"
    if "bf16" in name or name.startswith("flux"):
        return "safetensors_bf16"
    return "safetensors_fp16"  # SDXL / SD1.5 convention


def _detect_imagegen_comfyui_subdir(path: Path) -> str:
    """Infer the ComfyUI models/ subdirectory for an image gen model.

    FLUX diffusion model UNets go in diffusion_models/; standard checkpoints
    (SDXL, SD1.5) go in checkpoints/.  Detection uses the parent directory name
    and filename, covering both the standard layout and ad-hoc paths.
    """
    name = path.name.lower()
    parent = path.parent.name.lower()
    if "flux" in name or "flux" in parent or "diffusion" in parent or "unet" in parent:
        return "diffusion_models"
    return "checkpoints"


def _iter_image_gen_candidates(scan_dir: Path):
    """Yield (path, format) pairs for .safetensors files in scan_dir."""
    if not scan_dir.exists():
        print(f"[ingest] Directory not found: {scan_dir}", file=sys.stderr)
        return
    for f in sorted(scan_dir.iterdir()):
        if f.is_file() and f.suffix.lower() == ".safetensors":
            yield f, _detect_imagegen_format(f.name)


def detect_image_gen(path: Path, catalog: dict[str, Any]) -> dict[str, Any]:
    """Build a catalog entry dict for a single-file image gen model (.safetensors)."""
    if not path.is_file():
        raise ValueError(f"Expected a .safetensors file, got: {path}")
    fmt = _detect_imagegen_format(path.name)
    fmt_short = _FORMAT_SHORT.get(fmt, fmt)
    name = _dir_to_display_name(path.stem)
    entry_id = f"{_slugify(name)}-{fmt_short}"
    comfyui_subdir = _detect_imagegen_comfyui_subdir(path)
    entry: dict[str, Any] = {
        "id":            entry_id,
        "name":          name,
        "path":          str(path),
        "format":        fmt,
        "ai_types":      ["image_gen"],
        "ctx":           0,
        "size_gb":       _file_size_gb(path),
        "server_status": {},
    }
    # Only write comfyui_subdir when it differs from the default ("checkpoints")
    # to keep catalog entries compact.
    if comfyui_subdir != "checkpoints":
        entry["comfyui_subdir"] = comfyui_subdir
    return entry


# ---------------------------------------------------------------------------
# Entry construction
# ---------------------------------------------------------------------------

def _copy_source(path: Path, fmt: str | None, catalog: dict[str, Any]) -> dict[str, Any] | None:
    """The catalog entry *path* is a copy of: same relative path (and format) at another location."""
    loc = location_of(path)
    if loc is None:
        return None
    rel = path.relative_to(loc.root)
    for entry in catalog.get("models", []):
        other = location_of(entry["path"])
        if other is None or other.root == loc.root:
            continue
        if Path(entry["path"]).relative_to(other.root) == rel and fmt in (None, entry["format"]):
            return entry
    return None


def build_entry(
    path: Path, catalog: dict[str, Any], ai_type: str = "text_gen", fmt: str | None = None,
) -> dict[str, Any]:
    """Catalog entry for the model at *path*.

    A copy of a catalogued model (same relative path at another location) becomes
    a snapshot of that entry, sharing its id and family.  Anything else is detected
    from disk and starts a new family named after it.  Raises ValueError when
    detection fails.
    """
    original = _copy_source(path, fmt, catalog)
    if original is not None:
        loc = location_of(path)
        assert loc is not None
        return relocated_entry(original, loc)
    if ai_type == "image_gen":
        entry = detect_image_gen(path, catalog)
    else:
        entry = detect_text_gen(path, catalog, fmt=fmt)
    entry["family"] = _slugify(entry["name"])
    return entry


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# Fields that refresh can update — all are auto-detected from disk / config.json.
# User-curated fields (note, name, format, id, path, family, ctx_override,
# favorite, server_status, server_params, min_sm, hw_note, comfyui_subdir,
# quant, trust_remote_code) are never touched by refresh.
_REFRESHABLE_FIELDS: frozenset[str] = frozenset({
    "arch", "ctx", "moe_active", "moe_experts", "mmproj_path", "mtp_embedded", "mtp_path", "params_m", "size_gb", "vision",
})


def _refresh_one(
    entry: dict[str, Any], dry_run: bool
) -> tuple[dict[str, Any], bool]:
    """Re-detect auto-derived fields for one existing catalog entry.

    Returns (updated_entry, changed).  Only _REFRESHABLE_FIELDS are ever
    modified; all user-curated fields are preserved verbatim.
    """
    path = Path(entry["path"])
    if not path.exists():
        print(f"[refresh] Skipping (path not found): {path}", file=sys.stderr)
        return entry, False

    ai_types = entry.get("ai_types", ["text_gen"])
    ai_type = ai_types[0] if ai_types else "text_gen"

    try:
        if ai_type == "image_gen":
            fresh = detect_image_gen(path, {})
        else:
            # Pass the stored format for directory entries so that mixed-format
            # directories don't trigger auto-detection and the size is computed
            # for the correct set of weight files.
            stored_fmt = entry.get("format") if path.is_dir() else None
            fresh = detect_text_gen(path, {}, fmt=stored_fmt)
    except ValueError as exc:
        print(f"[refresh] Skipping (detection failed): {path}: {exc}", file=sys.stderr)
        return entry, False

    # Re-compute size_gb using the stored (user-curated) format rather than the
    # freshly-detected one.  If the user has corrected a format mismatch, the size
    # should reflect the corrected format's weight files, not the auto-detected ones.
    if ai_type != "image_gen" and path.is_dir():
        stored_fmt = entry.get("format", "")
        size_globs = _weight_globs_for(path, stored_fmt)
        fresh["size_gb"] = _dir_size_gb(path, include_globs=size_globs)

    # Normalise to canonical Python types for comparison
    def _norm(field: str, raw: Any) -> Any:
        if field == "params_m":              return int(raw)   if raw else 0
        if field == "size_gb":               return float(raw) if raw else 0.0
        if field in ("arch",):               return str(raw) if raw else ""
        if field in ("mmproj_path", "mtp_path"): return str(raw) if raw else ""
        if field == "ctx":                   return int(raw) if raw else 32_768
        if field in ("moe_experts", "moe_active"): return int(raw) if raw else 0
        if field in ("vision", "mtp_embedded"): return bool(raw)
        return raw

    # Fields omitted from JSON when default — delete key rather than write falsy value
    _omit_when_default = frozenset({"params_m", "arch", "moe_experts", "moe_active", "vision", "mmproj_path", "mtp_path", "mtp_embedded"})

    changes: list[str] = []
    updated = dict(entry)

    for field in sorted(_REFRESHABLE_FIELDS):
        # If ctx detection returned 0 (key absent from config — e.g. ALiBi models like
        # BLOOM that have no max_position_embeddings), preserve the stored value rather
        # than overwriting a manually set ctx with the fallback default.
        if field == "ctx" and fresh.get("ctx", 0) == 0:
            continue
        # If params_m detection returned 0 (name has no B/M suffix and config lacks
        # num_parameters), preserve any manually-set value rather than deleting it.
        if field == "params_m" and fresh.get("params_m", 0) == 0:
            continue
        fresh_val = _norm(field, fresh.get(field))
        old_val   = _norm(field, entry.get(field))
        if fresh_val == old_val:
            continue
        changes.append(f"  {field}: {old_val!r} -> {fresh_val!r}")
        if field in _omit_when_default and not fresh_val:
            updated.pop(field, None)
        else:
            updated[field] = fresh_val

    if changes:
        print(f"[refresh] {entry['id']}  ({Path(entry['path']).name}):")
        for c in changes:
            print(c)
    else:
        print(f"[refresh] {entry['id']}  ({Path(entry['path']).name}): no changes")

    return updated, bool(changes)


def cmd_refresh(args: "argparse.Namespace") -> None:
    """Subcommand: re-detect auto-derived fields for existing catalog entries."""
    catalog = _load_catalog(_CATALOG_PATH)
    models: list[dict[str, Any]] = catalog.get("models", [])

    if args.path:
        target_path = str(Path(args.path))
        to_refresh = [e for e in models if e["path"] == target_path]
        if not to_refresh:
            print(f"[refresh] No catalog entry found for path: {target_path}", file=sys.stderr)
            sys.exit(1)
    else:
        to_refresh = models

    index_by_id = {e["id"]: i for i, e in enumerate(models)}
    updated_models = list(models)
    total_changed = 0

    print(
        f"[refresh] Refreshing {len(to_refresh)} entr(ies) "
        "(size recompute may be slow over NAS) …\n",
        file=sys.stderr,
    )

    for entry in to_refresh:
        updated_entry, changed = _refresh_one(entry, dry_run=False)
        if changed:
            total_changed += 1
            if not args.dry_run:
                updated_models[index_by_id[entry["id"]]] = updated_entry

    print()
    if args.dry_run:
        print(f"[refresh] --dry-run: {total_changed} entr(ies) would be updated.", file=sys.stderr)
    elif total_changed:
        catalog["models"] = updated_models
        _save_catalog(_CATALOG_PATH, catalog)
        print(f"[refresh] Updated {total_changed} entr(ies).", file=sys.stderr)
    else:
        print("[refresh] All entries are up-to-date.", file=sys.stderr)


def _add_dry_run_arg(p: "argparse.ArgumentParser") -> None:
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without modifying any files.",
    )


def _ingest_one(
    path: Path,
    catalog: dict[str, Any],
    dry_run: bool,
    ai_type: str = "text_gen",
    fmt: str | None = None,
) -> bool:
    """Detect and optionally append a single model path.  Returns True on success.

    *fmt* is forwarded to ``detect_text_gen`` for HF directories; ignored for
    GGUF files and image-gen entries.
    """
    try:
        entry = build_entry(path, catalog, ai_type, fmt)
    except ValueError as exc:
        print(f"[ingest] Skipping {path}: {exc}", file=sys.stderr)
        return False

    print("\nDetected entry:")
    print(json.dumps(entry, indent=2))
    print()

    if dry_run:
        return True

    catalog.setdefault("models", []).append(entry)
    _save_catalog(_CATALOG_PATH, catalog)
    print(f"[ingest] Appended '{entry['id']}' to {_CATALOG_PATH}", file=sys.stderr)
    return True


def ingest_model(
    path: Path,
    *,
    fmt: str | None = None,
    ai_type: str = "text_gen",
    catalog_path: Path = _CATALOG_PATH,
    entry_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Detect and append one explicit model path, returning its catalog entry.

    If the same path and format are already catalogued, returns that entry
    without writing a duplicate.
    """
    catalog = _load_catalog(catalog_path)
    if fmt is not None:
        for entry in catalog.get("models", []):
            if entry.get("path") == str(path) and entry.get("format") == fmt:
                return entry

    entry = build_entry(path, catalog, ai_type, fmt)
    if entry_overrides:
        entry.update(entry_overrides)

    for existing in catalog.get("models", []):
        if (
            existing.get("path") == entry["path"]
            and existing.get("format") == entry["format"]
        ):
            return existing

    catalog.setdefault("models", []).append(entry)
    _save_catalog(catalog_path, catalog)
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tools for populating and enriching models_local.json from disk.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Subcommands:\n"
            "  ingest         Add uncatalogued models (scan or specific path)\n"
            "  copy-configs   Copy HF config files into GGUF bundle directories\n"
            "  copy-mmproj    Copy mmproj files into GGUF bundle directories\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", metavar="SUBCOMMAND")
    subparsers.required = True

    ingest_p = subparsers.add_parser(
        "ingest",
        help="Add uncatalogued models to the catalog.",
        description=(
            "Add models to models_local.json.  Either scan a directory for\n"
            "uncatalogued models (optionally adding all at once), or target\n"
            "a single model by absolute path."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ingest_p.add_argument(
        "scan_root",
        nargs="?",
        metavar="SCAN_ROOT",
        help="Directory to scan for new models. Required unless --path is given.",
    )
    ingest_p.add_argument(
        "--all",
        action="store_true",
        dest="ingest_all",
        help="Ingest every uncatalogued model found in scan_root (default: first one only).",
    )
    ingest_p.add_argument(
        "--path",
        metavar="PATH",
        help="Absolute path to a specific model file or directory to ingest (skips scan).",
    )
    ingest_p.add_argument(
        "--ai-type",
        choices=["text_gen", "image_gen"],
        default="text_gen",
        dest="ai_type",
        help=(
            "AI type for the ingested models (default: text_gen). "
            "Use image_gen for diffusion checkpoints and FLUX models."
        ),
    )
    _add_dry_run_arg(ingest_p)

    refresh_p = subparsers.add_parser(
        "refresh",
        help="Re-detect auto-derived fields (ctx, size_gb, params_m, arch, moe, vision) for existing entries.",
        description=(
            "Re-runs detection on catalogued model paths and updates auto-derived fields\n"
            "without touching user-curated fields (id, name, format, family,\n"
            "ctx_override, favorite, server_status, server_params, min_sm, hw_note, etc.)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    refresh_p.add_argument(
        "--path",
        metavar="PATH",
        help="Refresh only the entry with this exact model path (default: all entries).",
    )
    _add_dry_run_arg(refresh_p)

    copy_p = subparsers.add_parser(
        "copy-configs",
        help=(
            "Copy HF config/tokenizer files from original model dirs into their "
            "GGUF bundle directories."
        ),
    )
    copy_p.add_argument(
        "scan_root",
        metavar="SCAN_ROOT",
        help="Primary directory to scan (e.g. /mnt/models/text-generation).",
    )
    _add_dry_run_arg(copy_p)

    mmproj_p = subparsers.add_parser(
        "copy-mmproj",
        help=(
            "Copy mmproj files from original model dirs into their "
            "GGUF bundle directories."
        ),
    )
    mmproj_p.add_argument(
        "scan_root",
        metavar="SCAN_ROOT",
        help="Primary directory to scan (e.g. /mnt/models/text-generation).",
    )
    _add_dry_run_arg(mmproj_p)

    args = parser.parse_args()

    if args.command == "refresh":
        cmd_refresh(args)
        return

    if args.command == "copy-configs":
        scan_root = Path(args.scan_root)
        all_roots = _all_roots(scan_root)
        root_list = "\n  ".join(str(r) for r in all_roots)
        print(f"[copy-configs] Scanning roots:\n  {root_list}")
        do_copy_configs(scan_root, args.dry_run)
        return

    if args.command == "copy-mmproj":
        scan_root = Path(args.scan_root)
        all_roots = _all_roots(scan_root)
        root_list = "\n  ".join(str(r) for r in all_roots)
        print(f"[copy-mmproj] Scanning roots:\n  {root_list}")
        do_copy_mmproj(scan_root, args.dry_run)
        return

    # ---- ingest ------------------------------------------------------------
    if args.path and args.scan_root:
        parser.error("--path and scan_root are mutually exclusive")
    if not args.path and not args.scan_root:
        parser.error("one of scan_root or --path is required")
    if args.path and args.ingest_all:
        parser.error("--all has no effect with --path (a single path is always fully ingested)")

    catalog = _load_catalog(_CATALOG_PATH)
    ai_type: str = args.ai_type

    # --- Mode: specific path ------------------------------------------------
    if args.path:
        target = Path(args.path)
        if not target.exists():
            print(f"[ingest] Path not found: {target}", file=sys.stderr)
            sys.exit(1)

        if target.is_dir() and ai_type == "text_gen":
            # Directories may have multiple weight formats — detect and ingest all
            # uncatalogued ones.  GGUF bundle directories are not valid targets here;
            # the caller should target the specific .gguf file instead.
            gguf_files = sorted(target.glob("*.gguf"))
            if gguf_files:
                file_list = ", ".join(f.name for f in gguf_files[:3])
                if len(gguf_files) > 3:
                    file_list += f", … ({len(gguf_files)} total)"
                print(
                    f"[ingest] Directory contains GGUF files — "
                    f"target the specific .gguf file instead: {file_list}",
                    file=sys.stderr,
                )
                sys.exit(1)
            known = _known_path_formats(catalog)
            cfg = _read_hf_config(target)
            all_fmts = _detect_all_hf_formats(target, cfg)
            new_fmts = [f for f in all_fmts if (str(target), f) not in known]
            if not new_fmts:
                print(f"[ingest] Already fully catalogued: {target}")
                return
            for fmt in new_fmts:
                print(f"[ingest] Target: {target} (format: {fmt})", file=sys.stderr)
                _ingest_one(target, catalog, args.dry_run, ai_type, fmt=fmt)
        else:
            # Single file (GGUF) or image gen — original single-format behaviour.
            known_paths = {m["path"] for m in catalog.get("models", [])}
            if str(target) in known_paths:
                print(f"[ingest] Already catalogued: {target}")
                return
            print(f"[ingest] Target: {target}", file=sys.stderr)
            _ingest_one(target, catalog, args.dry_run, ai_type)

        if args.dry_run:
            print("[ingest] --dry-run: catalog not modified.", file=sys.stderr)
        return

    # --- Mode: scan (one or all) --------------------------------------------
    known = _known_path_formats(catalog)
    scan_root = Path(args.scan_root)
    print(f"[ingest] Scanning {scan_root} …", file=sys.stderr)

    iter_fn = _iter_image_gen_candidates if ai_type == "image_gen" else _iter_candidates

    if args.ingest_all:
        new_pairs = [(p, f) for p, f in iter_fn(scan_root) if (str(p), f) not in known]
        if not new_pairs:
            print("[ingest] No new models found — everything in this directory is already catalogued.")
            return
        print(f"[ingest] Found {len(new_pairs)} uncatalogued model(s).", file=sys.stderr)
        added = 0
        for path, fmt in new_pairs:
            print(f"\n[ingest] Processing: {path}", file=sys.stderr)
            if _ingest_one(path, catalog, args.dry_run, ai_type, fmt=fmt if path.is_dir() else None):
                added += 1
        if args.dry_run:
            print(f"\n[ingest] --dry-run: {added} entr(ies) would be added.", file=sys.stderr)
        else:
            print(f"\n[ingest] Done — {added} entr(ies) added.", file=sys.stderr)
    else:
        result = next(
            ((p, f) for p, f in iter_fn(scan_root) if (str(p), f) not in known),
            None,
        )
        if result is None:
            print("[ingest] No new models found — all entries in this directory are already catalogued.")
            return
        path, fmt = result
        print(f"[ingest] Found: {path}", file=sys.stderr)
        _ingest_one(path, catalog, args.dry_run, ai_type, fmt=fmt if path.is_dir() else None)
        if args.dry_run:
            print("[ingest] --dry-run: catalog not modified.", file=sys.stderr)


if __name__ == "__main__":
    main()
