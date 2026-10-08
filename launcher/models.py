"""
models.py — Schema definitions: enums, dataclasses, and compatibility helpers.

Instances (the actual servers and models) live in data.py.

Navigation flow produced by these types:
  Level 1 — AIType selection
  Level 2 — AIModel filtered by chosen AIType
  Level 3 — ServerSoftware filtered by model.format ∈ server.supported_formats
                                    AND model.ai_types ∩ server.supported_ai_types ≠ ∅
  Launch  — merge server.default_params | model.server_params[server.id] → build command
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import re
from typing import Any, Iterable, Literal

from launcher.config import CONFIG


# ---------------------------------------------------------------------------
# Type enumerations
# ---------------------------------------------------------------------------

class AIType(str, Enum):
    """Capability category declared by both models and servers.
    A model can belong to multiple types (e.g. a multimodal model).
    """
    TEXT_GEN  = "text_gen"   # LLM chat / completion
    IMAGE_GEN = "image_gen"  # Diffusion / image synthesis
    EMBEDDING = "embedding"  # Text embedding / retrieval
    SPEECH    = "speech"     # STT / TTS


# Human-readable display label for each AI type — co-located with the enum so
# that adding a new AIType value and its label are a single-file change.
AI_TYPE_LABELS: dict[AIType, str] = {
    AIType.TEXT_GEN:  "Text Generation",
    AIType.IMAGE_GEN: "Image Generation",
    AIType.EMBEDDING: "Embeddings",
    AIType.SPEECH:    "Speech",
}

# Cache subdirectory name for each AI type.  Directory models are stored under
# <mount.root>/<AI_TYPE_CACHE_SUBDIR[type]>/<source_dir_name>/ on every cache
# mount, where source_dir_name is the basename of the model's source path.
# Single-file models (GGUF, single safetensors) use
# <mount.root>/<AI_TYPE_CACHE_SUBDIR[type]>/<model.id>/<filename>.
AI_TYPE_CACHE_SUBDIR: dict[AIType, str] = {
    AIType.TEXT_GEN:  "text-generation",
    AIType.IMAGE_GEN: "text-to-image",
    AIType.EMBEDDING: "embedding",
    AIType.SPEECH:    "speech",
}


class ModelFormat(str, Enum):
    """Storage/quantisation format. Gates server compatibility."""
    GGUF             = "gguf"             # GGUF quant — llama.cpp
    SAFETENSORS_BF16 = "safetensors_bf16" # HF Transformers BF16 safetensors — vLLM, tabbyAPI
    SAFETENSORS_FP16 = "safetensors_fp16" # HF Transformers FP16 safetensors — vLLM, tabbyAPI
    SAFETENSORS_FP8  = "safetensors_fp8"  # FP8 safetensors — vLLM only (dequants on sm_86)
    AWQ              = "awq"              # Activation-Aware Weight Quant (int4) — vLLM, sglang, ooba
    GPTQ             = "gptq"             # GPTQ quantised safetensors — vLLM, tabbyAPI, ooba
    INT8_BNB         = "int8_bnb"         # bitsandbytes INT8 pre-saved checkpoint — vLLM only
    EXL2             = "exl2"             # ExLlamaV2 quantised — tabbyAPI only
    PYTORCH_BIN      = "pytorch_bin"      # Legacy pytorch_model.bin shards — HF Transformers only


class GGUFQuant(str, Enum):
    """Standard GGUF quantisation types used in the 'quant' field of AIModel.

    Values match the strings embedded in GGUF filenames and reported by llama.cpp.
    Use these constants instead of bare strings to avoid typos.
    """
    # --- Unquantised ---------------------------------------------------------
    F32   = "F32"    # 32-bit float  (~4× model size vs BF16)
    F16   = "F16"    # 16-bit float
    BF16  = "BF16"   # BF16 (same size as F16, preferred for Ampere)

    # --- Legacy integer quants (rarely used for new models) -----------------
    Q4_0  = "Q4_0"   # 4-bit, small blocks, no extra scaling
    Q4_1  = "Q4_1"   # 4-bit, small blocks, with min/delta scaling
    Q5_0  = "Q5_0"   # 5-bit, small blocks
    Q5_1  = "Q5_1"   # 5-bit, small blocks, with min/delta scaling
    Q8_0  = "Q8_0"   # 8-bit, small blocks — near-lossless reference quant

    # --- K-quants (recommended mainstream options) --------------------------
    Q2_K   = "Q2_K"    # ~2.6 bpw  — very small, significant quality loss
    Q3_K_S = "Q3_K_S"  # ~3.4 bpw  — small variant
    Q3_K_M = "Q3_K_M"  # ~3.9 bpw  — medium variant
    Q3_K_L = "Q3_K_L"  # ~4.3 bpw  — large variant
    Q4_K_S = "Q4_K_S"  # ~4.4 bpw  — small variant
    Q4_K_M = "Q4_K_M"  # ~4.8 bpw  — medium variant (most popular)
    Q5_K_S = "Q5_K_S"  # ~5.5 bpw  — small variant
    Q5_K_M = "Q5_K_M"  # ~5.7 bpw  — medium variant
    Q6_K   = "Q6_K"    # ~6.6 bpw  — near-lossless for most architectures

    # --- I-quants (importance-matrix; better quality per bit than K-quants) -
    IQ1_S   = "IQ1_S"   # ~1.6 bpw  — extreme compression
    IQ1_M   = "IQ1_M"   # ~1.7 bpw
    IQ2_XXS = "IQ2_XXS" # ~2.1 bpw
    IQ2_XS  = "IQ2_XS"  # ~2.3 bpw
    IQ2_S   = "IQ2_S"   # ~2.5 bpw
    IQ2_M   = "IQ2_M"   # ~2.7 bpw
    IQ3_XXS = "IQ3_XXS" # ~3.1 bpw
    IQ3_XS  = "IQ3_XS"  # ~3.3 bpw
    IQ3_S   = "IQ3_S"   # ~3.5 bpw
    IQ3_M   = "IQ3_M"   # ~3.7 bpw
    IQ4_XS  = "IQ4_XS"  # ~4.3 bpw  — often better than Q4_K_M at same size
    IQ4_NL  = "IQ4_NL"  # ~4.5 bpw  — non-linear 4-bit (llama.cpp NL tables)

    # --- K-quant XL variants (Unsloth Dynamic extended-context) -------------
    # Longer names must be listed so GGUF_QUANTS_LONGEST_FIRST auto-sorts them
    # before their shorter K-quant prefixes (e.g. Q6_K_XL before Q6_K).
    Q2_K_XL = "Q2_K_XL"
    Q3_K_XL = "Q3_K_XL"
    Q4_K_XL = "Q4_K_XL"
    Q5_K_XL = "Q5_K_XL"
    Q6_K_XL = "Q6_K_XL"
    Q8_K    = "Q8_K"     # base 8-bit K-quant (Unsloth; not the same as Q8_0)
    Q8_K_XL = "Q8_K_XL"

    # --- K-quant Plus variants (Unsloth Dynamic) ----------------------------
    Q3_K_P = "Q3_K_P"
    Q4_K_P = "Q4_K_P"
    Q5_K_P = "Q5_K_P"
    Q6_K_P = "Q6_K_P"
    Q8_K_P = "Q8_K_P"

    # --- Architecture-specific custom quants --------------------------------
    MXFP4_MOE = "MXFP4_MOE"  # NVIDIA Microscaling FP4 (MoE; requires sm_89+)


# All GGUFQuant values sorted longest-first so that more-specific variants
# (e.g. "IQ2_XXS") always match before their shorter prefixes ("IQ2_XS", "IQ2_S")
# during filename scanning.  Derived from the enum; no manual maintenance needed.
GGUF_QUANTS_LONGEST_FIRST: tuple[GGUFQuant, ...] = tuple(
    sorted(GGUFQuant, key=lambda q: -len(q.value))
)


# llama.cpp splits large models into '<base>-00001-of-000NN.gguf' shards.  The
# first shard carries the metadata and pulls in the rest when loaded, so a split
# set is one model whose catalog path is the first shard.
_GGUF_SPLIT_RE = re.compile(r"^(?P<base>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})$")


def gguf_split_info(gguf_path: Path) -> tuple[str, int, int] | None:
    """Return (base_stem, index, total) for a llama.cpp split shard, else None."""
    match = _GGUF_SPLIT_RE.match(gguf_path.stem)
    if match is None:
        return None
    return match["base"], int(match["index"]), int(match["total"])


def gguf_shard_paths(gguf_path: Path) -> list[Path]:
    """Every file making up the model at *gguf_path*: all shards of a split set, else itself.

    Shard names are derived from the split pattern, so shards missing from disk
    are still listed — callers decide whether that is an error.
    """
    info = gguf_split_info(gguf_path)
    if info is None:
        return [gguf_path]
    base, _, total = info
    return [
        gguf_path.parent / f"{base}-{index:05d}-of-{total:05d}{gguf_path.suffix}"
        for index in range(1, total + 1)
    ]


class RuntimeQuant(str, Enum):
    """Runtime quantization mode applied by a server at model-load time.

    Only applicable to full-precision formats (BF16/FP16/BIN). Each
    ServerSoftware advertises which modes it supports via
    `supported_runtime_quants` and which input formats qualify via
    `quantizable_formats`.
    """
    BNB_INT8 = "bnb_int8"  # bitsandbytes 8-bit (INT8)  — ~half VRAM of BF16/FP16
    BNB_NF4  = "bnb_nf4"   # bitsandbytes 4-bit NF4      — ~quarter VRAM of BF16/FP16


# Human-readable display label for each runtime quant option.
RUNTIME_QUANT_LABEL: dict[RuntimeQuant, str] = {
    RuntimeQuant.BNB_INT8: "INT8 (bitsandbytes)",
    RuntimeQuant.BNB_NF4:  "INT4 NF4 (bitsandbytes)",
}

# Short label for compact display contexts (note bar hints, etc.).
RUNTIME_QUANT_SHORT: dict[RuntimeQuant, str] = {
    RuntimeQuant.BNB_INT8: "INT8",
    RuntimeQuant.BNB_NF4:  "NF4",
}

# Approximate VRAM multiplier relative to stored (BF16/FP16) size for each
# runtime quant mode.  Used for "would fit" hints in the model list.
# These are weight-only estimates; KV cache and activations add ~5–15% overhead.
RUNTIME_QUANT_VRAM_RATIO: dict[RuntimeQuant, float] = {
    RuntimeQuant.BNB_INT8: 0.50,   # INT8: ~1 byte/param vs 2 bytes for BF16
    RuntimeQuant.BNB_NF4:  0.25,   # NF4:  ~0.5 bytes/param (4-bit + double-quant)
}


def _quant_key(quant: "RuntimeQuant | None") -> str:
    """Return the JSON dict key used to store status for a given runtime quant.

    RuntimeQuant values map to their string values (e.g. 'bnb_int8').
    None maps to 'none' — full/stored precision, no additional quantization.
    """
    return quant.value if quant is not None else "none"


Status = Literal["good", "bad", "unknown"]


def _best_status(statuses: "Iterable[Status]") -> Status:
    """Collapse several results into one.  Priority: good > bad > unknown."""
    seen = set(statuses)
    if "good" in seen:
        return "good"
    if "bad" in seen:
        return "bad"
    return "unknown"

# Human-readable short name for each format (non-GGUF; GGUF uses quant string)
FORMAT_SHORT: dict[ModelFormat, str] = {
    ModelFormat.SAFETENSORS_BF16: "BF16",
    ModelFormat.SAFETENSORS_FP16: "FP16",
    ModelFormat.SAFETENSORS_FP8:  "FP8",
    ModelFormat.AWQ:              "AWQ",
    ModelFormat.GPTQ:             "GPTQ",
    ModelFormat.INT8_BNB:         "INT8",
    ModelFormat.EXL2:             "EXL2",
    ModelFormat.PYTORCH_BIN:      "BIN",
    ModelFormat.GGUF:             "GGUF",   # overridden by quant when set
}


def fmt_ctx(ctx: int) -> str:
    """131072 → '128K', 65536 → '64K', 32768 → '32K', 4096 → '4K', 0 → ''"""
    if ctx == 0:
        return ""
    if ctx >= 1_024:
        return f"{round(ctx / 1024)}K"
    return str(ctx)


# ---------------------------------------------------------------------------
# Cache mount
# ---------------------------------------------------------------------------


@dataclass
class CacheMount:
    """A local storage tier that holds a faster copy of a model.

    When a model has a cached_path set in the catalog, resolve_path()
    returns that stored path directly rather than scanning these mounts.
    Mounts are used here only to determine *where* to place a new cached
    copy and to display disk-space information in the UI.
    """
    label: str    # human-readable, e.g. "NVMe (fast_models)"
    root: Path    # filesystem root for this cache tier
    priority: int # 0 = highest priority


# ---------------------------------------------------------------------------
# AI Model
# ---------------------------------------------------------------------------

@dataclass
class AIModel:
    """Describes a single model artefact on disk — what it IS.

    How to run it on a specific server is expressed as param overrides in
    server_params, merged on top of ServerSoftware.default_params at launch time.
    Only override values that differ from the server defaults.

    Example server_params:
        {
            "vllm":  {"gpu_mem_util": 0.80, "tensor_parallel": 8, "extra_flags": ["--x"]},
            "tabby": {"model_name": "BeMag-123B-BF16"},
        }
    """
    id: str                        # unique slug; used as served-model-name / llama.cpp alias
    name: str                      # human display name, e.g. "BeMag 123B v0.1"
    path: str                      # canonical source path — any accessible location
    format: ModelFormat            # storage / quantisation format
    ai_types: frozenset[AIType]    # capabilities this model exposes
    ctx: int = 32_768              # model's native/trained max context length (tokens)
    ctx_override: int = 0         # launch override — passed to the server instead of ctx; 0 = use ctx
    cached_path: str = ""          # faster local copy; resolve_path() returns this when set and present
    size_gb: float = 0.0           # disk footprint in GB
    params_m: int = 0              # parameter count in millions (0 = unknown)
    moe_experts: int = 0           # routed expert count; 0 = dense model
    moe_active: int = 0            # experts active per token (0 when dense)
    vision: bool = False           # model accepts image/vision input
    mmproj_path: str = ""          # path to companion mmproj .gguf file (vision GGUF only)
    mtp_path: str = ""             # path to companion multi-token-prediction draft head .gguf (GGUF only)
    trust_remote_code: bool = False  # servers must pass --trust-remote-code
    quant: str = ""                # quant suffix for display, e.g. "Q4_K_M" (mainly GGUF)
    note: str = ""                 # user-defined note shown in the status bar
    arch: str = ""                 # HF architectures[0] class name, e.g. "MistralForCausalLM"
    favorite: bool = False         # pinned models float to the top of the sorted list
    min_sm: int = 0                # minimum CUDA compute capability required (e.g. 89 for sm_89+); 0 = no constraint
    hw_note: str = ""              # human-readable reason shown when hw_compat() is False
    # vLLM/SGLang tool-call parser name (e.g. "openai", "hermes", "llama3_json").
    # When set, builders emit --enable-auto-tool-choice / --tool-call-parser flags.
    # Uses vLLM parser names as canonical; SGLang names are translated automatically.
    tool_call_parser: str = ""
    # For image_gen models: which ComfyUI models/ subdirectory to stage into.
    # 'checkpoints' for standard SD/SDXL files; 'diffusion_models' for FLUX UNets.
    comfyui_subdir: str = "checkpoints"

    # Per-server test results: server_status[server_id][env_name][quant_key].
    # env_name is a registered Environment.name for that server; quant_key is
    # "none" (full/stored precision) or a RuntimeQuant.value such as "bnb_int8".
    # Missing keys = that (server, env, quant) combo is untested.
    server_status: dict[str, dict[str, dict[str, Status]]] = field(default_factory=dict)

    # Per-server parameter overrides, keyed by ServerSoftware.id.
    # Merged on top of ServerSoftware.default_params at launch time.
    server_params: dict[str, dict[str, Any]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Deserialisation
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AIModel":
        """Construct from a JSON-compatible dict (as loaded from models_local.json)."""
        return AIModel(
            id=d["id"],
            name=d["name"],
            path=d["path"],
            cached_path=d.get("cached_path", ""),
            format=ModelFormat(d["format"]),
            ai_types=frozenset(AIType(t) for t in d.get("ai_types", ["text_gen"])),
            ctx=int(d.get("ctx", 32_768)),
            ctx_override=int(d.get("ctx_override", 0)),
            size_gb=float(d.get("size_gb", 0.0)),
            params_m=int(d.get("params_m", 0)),
            moe_experts=int(d.get("moe_experts", 0)),
            moe_active=int(d.get("moe_active", 0)),
            vision=bool(d.get("vision", False)),
            mmproj_path=d.get("mmproj_path", ""),
            mtp_path=d.get("mtp_path", ""),
            trust_remote_code=bool(d.get("trust_remote_code", False)),
            quant=d.get("quant", ""),
            note=d.get("note", ""),
            arch=d.get("arch", ""),
            favorite=bool(d.get("favorite", False)),
            min_sm=int(d.get("min_sm", 0)),
            hw_note=d.get("hw_note", ""),
            tool_call_parser=d.get("tool_call_parser", ""),
            comfyui_subdir=d.get("comfyui_subdir", "checkpoints"),
            server_status=d.get("server_status", {}),
            server_params=d.get("server_params", {}),
        )

    # ------------------------------------------------------------------
    # Path / cache helpers
    # ------------------------------------------------------------------

    @property
    def cache_subdir(self) -> str:
        """Cache subdirectory name for this model's primary AI type.

        Directory models are stored at <mount.root>/<cache_subdir>/<source_dir_name>/.
        Single-file models are stored at <mount.root>/<cache_subdir>/<model.id>/.
        The primary type is the first AIType by enum declaration order.
        """
        for ai_type in AIType:
            if ai_type in self.ai_types:
                return AI_TYPE_CACHE_SUBDIR[ai_type]
        return "models"  # unreachable in practice

    def resolve_path(self) -> str:
        """Return the fastest available path: cached_path if present on disk, else path."""
        if self.cached_path:
            p = Path(self.cached_path)
            # An incomplete copy of a split GGUF would fail to load; fall back to the source.
            if all(shard.exists() for shard in gguf_shard_paths(p)):
                return self.cached_path
        return self.path

    @property
    def companion_paths(self) -> list[str]:
        """Source paths of companion files that must sit beside the model (mmproj, MTP head)."""
        return [p for p in (self.mmproj_path, self.mtp_path) if p]

    def _resolve_companion(self, source_path: str) -> str:
        """Return the companion's copy beside the cached model when the cache is in use, else source_path."""
        if not source_path:
            return ""
        resolved = self.resolve_path()
        if resolved != self.path:
            cached = Path(resolved).parent / Path(source_path).name
            if cached.is_file():
                return str(cached)
        return source_path

    def resolve_mmproj_path(self) -> str:
        return self._resolve_companion(self.mmproj_path)

    def resolve_mtp_path(self) -> str:
        return self._resolve_companion(self.mtp_path)

    def cached_in(self, mounts: "list[CacheMount]") -> "CacheMount | None":
        """Return the cache mount that contains cached_path, or None if not cached."""
        if not self.cached_path:
            return None
        p = Path(self.cached_path)
        for mount in sorted(mounts, key=lambda m: m.priority):
            try:
                p.relative_to(mount.root)
                return mount
            except ValueError:
                continue
        return None

    @property
    def effective_ctx(self) -> int:
        """Context length to pass to the server: ctx_override if set, otherwise ctx."""
        return self.ctx_override if self.ctx_override else self.ctx

    @property
    def format_tag(self) -> str:
        """Short format/quant string for the display label, e.g. 'Q4_K_M' or 'BF16'."""
        return self.quant if self.quant else FORMAT_SHORT.get(self.format, self.format.value)

    @property
    def display_label(self) -> str:
        """Full label rendered in the model list: name, format tag, size, ctx."""
        size  = f"{self.size_gb:.0f} GB" if self.size_gb else ""
        ctx   = fmt_ctx(self.ctx)
        right = "  ".join(filter(None, [size, f"{ctx} ctx"]))
        return f"{self.name}  ·  {self.format_tag}" + (f"  [{right}]" if right else "")

    def vram_fit_hints(
        self,
        total_gb: float,
        servers: "list[ServerSoftware]",
    ) -> list[tuple[RuntimeQuant, float]]:
        """Return runtime quants that would fit within total_gb, with estimated sizes.

        Only considers quants that at least one compatible server supports for
        this model's format.  Returned list is ordered by VRAM ratio (largest
        first, i.e. best quality first).

        Args:
            total_gb: total available VRAM in GB.
            servers:  full server list to consult for quant support.

        Returns:
            List of (RuntimeQuant, estimated_gb) pairs for quants that would fit.
            Empty when size_gb is unknown (0), or no quant would fit.
        """
        if not self.size_gb:
            return []
        compatible = [s for s in servers if self.compatible_with(s)]
        supported: set[RuntimeQuant] = set()
        for s in compatible:
            supported.update(s.available_runtime_quants(self.format))
        result: list[tuple[RuntimeQuant, float]] = []
        for rq in RuntimeQuant:  # enum declaration order = best quality first
            if rq not in supported:
                continue
            est = round(self.size_gb * RUNTIME_QUANT_VRAM_RATIO[rq], 0)
            if est <= total_gb:
                result.append((rq, est))
        return result

    def hw_compat(self, system_sm: int) -> bool:
        """True if this model can run on hardware with the given compute capability.

        Args:
            system_sm: minimum sm across all GPUs as an int (e.g. 86 for sm_8.6).
                       Pass 0 to indicate unknown — returns True (no gate).
        """
        return self.min_sm == 0 or system_sm == 0 or system_sm >= self.min_sm

    @property
    def hw_compat_note(self) -> str:
        """Human-readable incompatibility message. Returns hw_note if set,
        otherwise generates a default from min_sm."""
        if self.hw_note:
            return self.hw_note
        if self.min_sm:
            return f"Requires sm_{self.min_sm}+ GPU"
        return ""

    def compatible_with(self, server: "ServerSoftware") -> bool:
        """True if this model can be loaded and served by the given server."""
        return (
            self.format in server.supported_formats
            and bool(self.ai_types & server.supported_ai_types)
        )

    def status_for(self, server_id: str, env_name: str, quant: "RuntimeQuant | None" = None) -> Status:
        """Status for a specific (server, environment, quant) combo, defaulting to 'unknown'."""
        env_data = self.server_status.get(server_id, {}).get(env_name, {})
        return env_data.get(_quant_key(quant), "unknown")

    def env_best_status(self, server_id: str, env_name: str) -> Status:
        """Best status across all quant combos for one environment of a server."""
        return _best_status(self.server_status.get(server_id, {}).get(env_name, {}).values())

    def best_status_for(self, server_id: str) -> Status:
        """Best status across all environments and quant combos for this server."""
        return _best_status(
            s for env_data in self.server_status.get(server_id, {}).values() for s in env_data.values()
        )

    def composite_status(self, servers: "list[ServerSoftware]") -> tuple[str, str]:
        """Return (color, symbol) summarising status across all applicable servers.

        Symbol key:
          [dim]?[/]   — all untested
          [green]✓[/] — all confirmed good
          [red]✗[/]   — all confirmed bad
          ✓?          — ≥1 good, none bad (some untested)
          ✓✗          — mixed: ≥1 good and ≥1 bad
          ✗?          — ≥1 bad, none good (some untested)
        """
        applicable = [s for s in servers if self.compatible_with(s)]
        if not applicable:
            return "dim", "N/A"
        statuses = [self.best_status_for(s.id) for s in applicable]
        goods = statuses.count("good")
        bads  = statuses.count("bad")
        n     = len(statuses)
        if goods == 0 and bads == 0:  return "dim",    "?"
        if goods == n:                 return "green",  "✓"
        if bads  == n:                 return "red",    "✗"
        if goods > 0 and bads == 0:   return "green",  "✓?"
        if goods > 0:                  return "yellow", "✓✗"
        return                         "yellow", "✗?"


# ---------------------------------------------------------------------------
# Server Software
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VersionProbe:
    """Reads a server's version by running a command inside an environment.

    For builds without Python package metadata (e.g. llama.cpp).  `patterns` are
    regexes with named groups, tried in order (output formats change across
    releases); `template` formats the first match, e.g. "b{build} ({commit})".
    """
    executable: str
    args: tuple[str, ...]
    patterns: tuple[str, ...]
    template: str

    def label(self, env: Any) -> str:
        """Formatted version for *env* (an environments.Environment), or ''."""
        output = env.command_output(self.executable, self.args)
        for pattern in self.patterns:
            if match := re.search(pattern, output):
                return self.template.format(**match.groupdict())
        return ""

@dataclass
class ServerSoftware:
    """Describes a server process — supported formats/types and default launch params.

    default_params is the baseline for every launch on this server.
    A model's server_params[self.id] dict is merged on top at launch time.
    Only values that differ from these defaults need to appear in server_params.
    """
    id: str                                     # unique slug, e.g. "vllm", "llamacpp", "tabby"
    label: str                                  # display text in the UI
    supported_formats: frozenset[ModelFormat]   # model formats this server can load
    supported_ai_types: frozenset[AIType]       # AI types this server can expose
    default_params: dict[str, Any] = field(default_factory=dict)
    # Formats for which this server can apply runtime quantization (full-precision only).
    quantizable_formats: frozenset[ModelFormat] = field(default_factory=frozenset)
    # Ordered tuple of runtime quant options available for quantizable formats.
    supported_runtime_quants: tuple[RuntimeQuant, ...] = field(default_factory=tuple)
    # Launch-time environment / process setup — set in data.py, consumed by launch.py.
    # The interpreter / binaries come from the Environment chosen at launch
    # (see environments.py); these fields describe what the server needs on top.
    env_kind: str = "conda"              # Environment.kind registered for this server
    # Python distributions whose installed versions identify an environment in the
    # UI (e.g. ("vllm",) → "vllm 0.26.0").  Empty = no version shown.
    version_packages: tuple[str, ...] = ()
    # Command-based version detection for environments without package metadata
    # (native builds).  None = no probe.
    version_probe: VersionProbe | None = None
    strip_pythonpath: bool = False       # pop PYTHONPATH from env (prevents src-tree shadowing)
    cuda_home: str = ""                  # override CUDA_HOME in env when non-empty
    # Glob (relative to the environment's site-packages) locating a pip-bundled CUDA
    # toolkit, e.g. "nvidia/cu1[0-9]*" for split-wheel cu13 installs.  When it matches,
    # that root becomes CUDA_HOME and its lib/ is prepended to LD_LIBRARY_PATH (runtime
    # .so loading) and LIBRARY_PATH (link-time -l resolution for JIT compilers).
    bundled_cuda_glob: str = ""
    # Symlinks ensured inside the bundled CUDA lib/ dir before launch: {link_name: target_glob}.
    # Self-heals split-wheel CUDA toolkits that ship libfoo.so.NN but omit the libfoo.so linker name.
    lib_symlinks: dict[str, str] = field(default_factory=dict)
    extra_env: dict[str, str] = field(default_factory=dict)  # additional env vars merged at launch
    uses_staging_dir: bool = False       # create a symlink under model_dir before launch
    # Maps ModelFormat → suffix appended to the staging symlink name.
    # e.g. {ModelFormat.GGUF: ".gguf"} makes ooba's format scanner recognise the file.
    staging_format_suffixes: dict[ModelFormat, str] = field(default_factory=dict)
    # When True, stage ALL compatible models at launch (not just the selected one).
    # Use for servers where model selection happens per-request (e.g. ComfyUI API),
    # rather than at launch time.
    stage_all_models: bool = False
    # When set, names an AIModel field whose value is used as a subdirectory within
    # the staging base (model_dir).  e.g. "comfyui_subdir" → model.comfyui_subdir.
    # Allows per-model routing into different server-side subdirectories (e.g.
    # diffusion_models/ vs checkpoints/ in ComfyUI) without any branching in launch.py.
    staging_subdir_attr: str | None = None
    # When True, the staged symlink is named after the source file (Path(resolve_path()).name)
    # rather than model.id + format suffix.  Use for servers (e.g. ComfyUI) where the
    # client selects models by original filename, not by catalog ID.
    staging_use_source_filename: bool = False
    launch_cwd: str | None = None        # working directory for the spawned process
    # GPU selection constraints — used by GpuSelectScreen.
    # max_gpus: 0 = no limit (all GPUs selectable, multi-select);
    #           1 = single GPU only (radio-button selection).
    max_gpus: int = 0
    # gpu_count_param: when set, the GPU count is injected as param_overrides[gpu_count_param]
    # so that parallelism settings (e.g. pipeline_parallel for vLLM) stay in sync with the
    # number of GPUs actually selected.  Set to None for servers that handle any GPU count
    # automatically (tabbyAPI --gpu-split-auto, llama.cpp layer distribution).
    gpu_count_param: str | None = None
    # concurrency_param: names the default_params key holding the number of requests the
    # server processes concurrently (llama.cpp "parallel" slots).  When set, GpuSelectScreen
    # lets the user adjust it per launch and remembers the value per model.
    # The model's effective_ctx stays the per-request context; each builder translates.
    concurrency_param: str | None = None
    def available_runtime_quants(self, fmt: ModelFormat) -> tuple[RuntimeQuant, ...]:
        """Return ordered runtime quant options for the given input format.

        Returns an empty tuple when no runtime quantization is available for
        this server+format combination (already-quantized formats, or servers
        that never support runtime quantization).
        """
        if fmt not in self.quantizable_formats:
            return ()
        return self.supported_runtime_quants

    def env_version_label(self, env: Any) -> str:
        """Installed version(s) of this server in *env*, e.g. 'vllm 0.26.0' ('' if unknown)."""
        parts = [env.version_label(self.version_packages)]
        if self.version_probe is not None:
            parts.append(self.version_probe.label(env))
        return "  ·  ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Lookup helpers  (pass in the lists from data.py)
# ---------------------------------------------------------------------------

def models_for_type(models: list[AIModel], ai_type: AIType) -> list[AIModel]:
    """All models that support the given AI type."""
    return [m for m in models if ai_type in m.ai_types]


def servers_for_model(servers: list[ServerSoftware], model: AIModel) -> list[ServerSoftware]:
    """All servers that can run the given model (format + type compatibility)."""
    return [s for s in servers if model.compatible_with(s)]


def servers_for_type(servers: list[ServerSoftware], ai_type: AIType) -> list[ServerSoftware]:
    """All servers that support the given AI type."""
    return [s for s in servers if ai_type in s.supported_ai_types]


def uses_hub_flow(servers: list[ServerSoftware], ai_type: AIType) -> bool:
    """True when every server for this AI type uses stage_all_models.

    These server types don't require per-launch model selection, so the
    launcher shows a hub with separate Launch / Manage Models paths instead
    of going straight to ModelSelectScreen.
    """
    matching = servers_for_type(servers, ai_type)
    return bool(matching) and all(s.stage_all_models for s in matching)


# ---------------------------------------------------------------------------
# Quantization Tool
# ---------------------------------------------------------------------------

@dataclass
class QuantizationTool:
    """Describes an offline quantization / conversion tool.

    Each tool converts models from one or more input formats to a single
    output format.  The quant_options tuple lists the option identifiers
    that the tool accepts (e.g. GGUFQuant values or bpw strings for EXL2).
    quant_option_labels provides a human-readable display string for each.

    Instance data (the actual registered tools) lives in data.py alongside
    SERVERS.  Only tool-agnostic logic belongs here.

    Dependency management fields (populated in data.py):
      env_server_id     — ServerSoftware.id whose *default* Environment the tool runs from
                          (e.g. "llamacpp" for llama-quantize, "tabby" for exllamav2).
      avail_executable  — executable name that must exist in that environment's bin dir.
      avail_module      — Python module that must be importable in that environment.
      pip_package       — pip package installed into that environment for auto-install;
                          "" = not auto-installable.
      install_note      — shown to the user when the tool is unavailable and cannot be auto-installed.
    """
    id: str                                  # unique slug, e.g. "llama-quantize", "exl2-convert"
    label: str                               # display label, e.g. "llama.cpp (GGUF requantize)"
    input_formats: frozenset[ModelFormat]    # model formats this tool accepts as input
    output_format: ModelFormat               # the format it produces
    quant_options: tuple[str, ...]           # ordered list of option identifiers
    quant_option_labels: dict[str, str]      # human-readable label per option identifier
    env_server_id: str = ""                  # server whose default environment provides the tool
    cuda_home: str = ""                      # CUDA_HOME override injected at run time (if non-empty)
    avail_executable: str = ""               # executable in the env's bin dir; must exist
    avail_module: str = ""                   # module importable in the env's python
    pip_package: str = ""                    # auto-install package; "" = not auto-installable
    install_note: str = ""                   # shown when unavailable and not auto-installable


def tools_for_model(tools: "list[QuantizationTool]", model: AIModel) -> "list[QuantizationTool]":
    """All quantization tools that can accept the given model's format as input."""
    return [t for t in tools if model.format in t.input_formats]


def available_ai_types(models: list[AIModel]) -> list[AIType]:
    """AI types that have at least one model, in canonical display order."""
    _ORDER = [AIType.TEXT_GEN, AIType.IMAGE_GEN, AIType.EMBEDDING, AIType.SPEECH]
    present = {t for m in models for t in m.ai_types}
    return [t for t in _ORDER if t in present]


