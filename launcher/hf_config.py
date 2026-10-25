"""
hf_config.py — Launch-planning facts read from a model's HF config.json.

Used by the parallelism planner (launch.compute_parallelism) and by "auto" launch
parameters (data.py) that depend on model dimensions, e.g. whether vLLM needs
expert parallelism, and by quantization tools that need to know a checkpoint's
quantization layout.  Results are cached per model path.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from launcher.models import AIModel


def _compressed_tensors_block(quant: dict[str, Any]) -> list[int]:
    """Weight block shape of a compressed-tensors block scheme (e.g. FP8_BLOCK [128, 128])."""
    for group in (quant.get("config_groups") or {}).values():
        block = (group.get("weights") or {}).get("block_structure")
        if block:
            return list(block)
    return []


@functools.lru_cache(maxsize=64)
def read_parallelism_fields(model_path: str) -> dict[str, Any] | None:
    """Read parallelism-relevant fields from a model's HF config.json.

    Multimodal checkpoints nest the language model's fields under
    ``text_config``; those take precedence over the top level.
    Returns None if config.json is absent or lacks num_attention_heads.
    """
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.is_file():
        return None
    try:
        with open(cfg_path, encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    text = raw.get("text_config") or {}
    def field(key: str, default: int = 0) -> int:
        return int(text.get(key, raw.get(key, default)) or 0)
    n_heads = field("num_attention_heads")
    if not n_heads:
        return None
    architectures = raw.get("architectures", [])
    quant = raw.get("quantization_config") or text.get("quantization_config") or {}
    block = quant.get("weight_block_size") or _compressed_tensors_block(quant)
    return {
        "num_attention_heads":    n_heads,
        "num_key_value_heads":    field("num_key_value_heads", n_heads),
        "intermediate_size":      field("intermediate_size"),
        "num_hidden_layers":      field("num_hidden_layers"),
        # Linear-attention (Gated DeltaNet) heads in hybrid models such as
        # Qwen3-Next / Qwen3.5 / Qwen4Exp — also sharded across TP ranks.
        "linear_num_key_heads":   field("linear_num_key_heads"),
        "linear_num_value_heads": field("linear_num_value_heads"),
        # Routed-expert width, sharded across TP ranks unless experts are distributed (EP).
        "moe_intermediate_size":  field("moe_intermediate_size"),
        # Block-quantization tile (e.g. FP8 [128, 128]); every TP shard must align to it.
        "weight_block_size":      tuple(int(b) for b in block),
        "architecture":           architectures[0] if architectures else "",
    }


def expert_parallel_needed(model: "AIModel | None", tp: int) -> tuple[bool, str]:
    """Whether MoE experts must be distributed whole across *tp* ranks (vLLM --enable-expert-parallel).

    Expert parallelism only takes effect with tensor parallelism (TP > 1).  It is
    required when the experts cannot be TP-sharded: the expert width does not divide
    by TP, or a shard does not align to the checkpoint's quantization block (e.g.
    Qwen3.8-Flash-Next FP8: 640 / 8 = 80, not a multiple of 128).
    """
    if model is None or not model.moe_experts:
        return False, "dense model"
    if tp <= 1:
        return False, "no effect without tensor parallelism"
    cfg = read_parallelism_fields(model.path)
    width = cfg["moe_intermediate_size"] if cfg else 0
    if not width:
        return False, "expert width unknown"
    if width % tp:
        return True, f"expert width {width} does not divide by TP{tp}"
    shard = width // tp
    for block in cfg["weight_block_size"] if cfg else ():
        if shard % block:
            return True, f"expert shard {width}/TP{tp} = {shard} not aligned to quantization block {block}"
    return False, f"experts shard evenly at TP{tp}"


@functools.lru_cache(maxsize=64)
def fp8_block_size(model_path: str) -> tuple[int, int] | None:
    """Block shape of a standard block-FP8 checkpoint, else None.

    Standard = quantization_config quant_method "fp8" with a weight_block_size,
    and each FP8 weight paired with a "<module>.weight_scale_inv" tensor (DeepSeek
    V3, Qwen, MiniMax, GLM FP8 releases) — the layout generic dequantizers
    handle.  Checkpoints with their own layout (e.g. DeepSeek-V4's FP4 experts
    with ".scale" tensors) return None.  Reads config.json and the safetensors
    index only.
    """
    path = Path(model_path)
    try:
        cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
        quant = cfg.get("quantization_config") or (cfg.get("text_config") or {}).get("quantization_config") or {}
        block = quant.get("weight_block_size")
        if quant.get("quant_method") != "fp8" or not block or len(block) != 2:
            return None
        index = path / "model.safetensors.index.json"
        if index.is_file():
            names = json.loads(index.read_text(encoding="utf-8"))["weight_map"].keys()
            if not any(n.endswith(".weight_scale_inv") for n in names):
                return None
    except (OSError, ValueError, KeyError):
        return None
    return int(block[0]), int(block[1])
