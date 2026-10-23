"""
launch.py — Unified command builder and process launcher.

build_cmd(server, env, model) → list[str]
    Merges server.default_params with model.server_params[server.id],
    then dispatches to the server-specific builder, which takes its
    interpreter / binaries from the chosen Environment.

compute_parallelism(gpu_count, model) → ParallelismPlan
    Reads the model's HF config.json to determine valid TP×PP splits for the
    given GPU count.  Prefers smallest TP (largest PP) for PCIe topology.
    Reduces GPU count when model dimensions are incompatible.

launch(server, env, model) → tuple[subprocess.Popen[bytes], Path, int]
    Builds the command, sets up the environment, spawns the process, and
    returns the Popen handle, the session log path, and the PTY master fd.
    stdout/stderr are connected to a PTY slave so the child process sees a
    real terminal (preserving ANSI colours). The caller owns master_fd and
    must close it after the tee thread finishes.

Adding a new server type:
  1. Write a _build_<name>(server, env, model, params) → list[str] function.
  2. Register it in _BUILDERS.
  3. Set any launch-time fields (env_kind, strip_pythonpath, cuda_home, uses_staging_dir,
     staging_format_suffixes, launch_cwd) on the ServerSoftware entry in data.py.
"""
from __future__ import annotations

import fcntl
import json
import math
import os
import pty
import shlex
import shutil
import struct
import subprocess
import termios
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from launcher.data import AI_MODELS
from launcher.chat_template import patched_template
from launcher.hf_config import read_parallelism_fields
from launcher.environments import ENVIRONMENTS, Environment
from launcher.gpu_power import PowerLimitStep
from launcher.models import (
    AIModel, GGUF_QUANTS_LONGEST_FIRST, ModelFormat,
    QuantizationTool, RopeScaling, RuntimeQuant, ServerSoftware, _quant_key,
    gguf_shard_paths, gguf_split_info,
)
from launcher.params import AutoBoolParam, ParamSpec
from launcher.config import CONFIG
from launcher.storage import (
    LOCATIONS, DeletionPlan, Location, location_of, model_metadata_dir, relocated_entry,
)


_LOGS_DIR = Path(__file__).parent.parent / "logs"
_CATALOG_PATH = Path(__file__).parent / "user_data" / "models_local.json"

# Install/app paths are resolved from CONFIG (launcher/user_data/config.toml) at import
# time; interpreters and binaries come from the Environment chosen at launch.

# Translation from canonical (vLLM) tool-call parser names to SGLang names.
# Only entries where the names differ need to appear here; matching names
# pass through unchanged.
_SGLANG_TOOL_PARSER_MAP: dict[str, str] = {
    "openai":       "gpt-oss",
    "llama3_json":  "llama3",
    "llama4_json":  "llama3",       # SGLang uses llama3 for all Llama models
    "llama4_pythonic": "pythonic",
    "deepseek_v3":  "deepseekv3",
    "deepseek_v31": "deepseekv31",
    "deepseek_v32": "deepseekv32",
    "hermes":       "hermes",       # same — listed for documentation
    "qwen3_xml":    "qwen3_coder",
    "minimax_m2":   "minimax-m2",
}

# Translation from canonical (vLLM) reasoning parser names to SGLang names; matching
# names (qwen3, glm45, kimi_k2, step3, step3p5) pass through unchanged.
_SGLANG_REASONING_PARSER_MAP: dict[str, str] = {
    "deepseek_r1":             "deepseek-r1",
    "deepseek_v3":             "deepseek-v3",
    "openai_gptoss":           "gpt-oss",
    "minimax_m2":              "minimax",
    "minimax_m2_append_think": "minimax-append-think",
    "nemotron_v3":             "nano_v3",
}


@dataclass(frozen=True)
class LaunchJob:
    """A confirmed server launch: process factory plus what was selected to produce it."""

    factory: Callable[[], tuple[subprocess.Popen[bytes], Path, int]]
    model: "AIModel | None"
    server: ServerSoftware
    env: Environment


@dataclass(frozen=True)
class QuantizeJob:
    """A conversion/quantization process plus the information needed to finalize it."""

    factory: Callable[[], tuple[subprocess.Popen[bytes], Path, int]]
    model: AIModel
    tool: QuantizationTool
    quant_option: str
    output_path: Path


_EXL2_WORK_ARTIFACTS = (
    "cal_data.safetensors",
    "hidden_states.safetensors",
    "job_new.json",
    "measurement.json",
    "out_tensor",
)


def _remove_exl2_work_artifacts(output_path: Path) -> None:
    for name in _EXL2_WORK_ARTIFACTS:
        artifact = output_path / name
        if artifact.is_dir():
            shutil.rmtree(artifact)
        elif artifact.exists():
            artifact.unlink()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _merged(server: ServerSoftware, model: "AIModel | None", param_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge server defaults with the model's per-server overrides, then param_overrides."""
    p = dict(server.default_params)
    if model is not None:
        p.update(model.server_params.get(server.id, {}))
    if param_overrides:
        p.update(param_overrides)
    return p


def _entry_key(entry: dict) -> tuple[str, str]:
    """Catalog key of a JSON entry — matches AIModel.key."""
    return entry["path"], entry["format"]


def _load_catalog() -> dict:
    with open(_CATALOG_PATH, encoding="utf-8") as f:
        return json.load(f)


def _save_catalog(catalog: dict) -> None:
    with open(_CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _patch_catalog(model: AIModel, apply: "Callable[[dict], None]") -> None:
    """Load models_local.json, find *model*'s entry (by key), call apply(entry), then save.

    Raises ValueError if the model is not found in the catalog.
    """
    catalog = _load_catalog()
    for entry in catalog["models"]:
        if _entry_key(entry) == model.key:
            apply(entry)
            break
    else:
        raise ValueError(f"Could not find {model.path} ({model.format.value}) in models_local.json")
    _save_catalog(catalog)


# ---------------------------------------------------------------------------
# Parallelism computation (vLLM / SGLang)
# ---------------------------------------------------------------------------

@dataclass
class ParallelismPlan:
    """Computed TP×PP parallelism for a multi-GPU server launch."""
    tp: int            # tensor_parallel_size
    pp: int            # pipeline_parallel_size
    gpu_count: int     # effective GPU count (may be < user-selected count)
    note: str = ""     # human-readable explanation when adjustments were made


# Architectures validated to correctly scope MLP / attention parallelism
# assertions to the TP group (get_tensor_model_parallel_world_size) rather
# than the total process count.  For these, only standard TP divisibility
# checks are needed.
#
# Unlisted or unknown architectures get an extra safety check:
# intermediate_size must also divide evenly by the total GPU count, because
# some model implementations assert ``intermediate_size % world_size == 0``
# where world_size is the global process count (TP × PP).
_TP_SCOPED_ARCHS: frozenset[str] = frozenset({
    "BloomForCausalLM",
    "CohereForCausalLM",
    "DeepseekV2ForCausalLM",
    "DeepseekV3ForCausalLM",
    "DeepseekV4ForCausalLM",
    "GPT2LMHeadModel",
    "GPTBigCodeForCausalLM",
    "GPTJForCausalLM",
    "GPTNeoForCausalLM",
    "GPTNeoXForCausalLM",
    "Gemma2ForCausalLM",
    "KimiK25ForConditionalGeneration",
    "LagunaForCausalLM",
    "LLaMAForCausalLM",
    "Llama4ForConditionalGeneration",
    "LlamaForCausalLM",
    "Ministral3ForCausalLM",
    "Mistral3ForConditionalGeneration",
    "MiniMaxM2ForCausalLM",
    "MistralForCausalLM",
    "MixtralForCausalLM",
    "OPTForCausalLM",
    "Phi3ForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen3MoeForCausalLM",
    "Qwen3NextForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
    "SarvamMLAForCausalLM",
    "XGLMForCausalLM",
})

# Architectures that cannot use pipeline parallelism at all; the planner uses
# pure TP (PP=1) for them, even if that leaves some selected GPUs unused.
#   Qwen4Exp — its N-gram PLE embedding needs the raw input_ids, which
#              non-first pipeline ranks never receive (vLLM rejects PP > 1).
_NO_PP_ARCHS: frozenset[str] = frozenset({
    "Qwen4ExpForCausalLM",
    "Qwen4ExpForConditionalGeneration",
})


def _sorted_divisors(n: int) -> list[int]:
    """Return the positive divisors of *n* in ascending order."""
    small: list[int] = []
    large: list[int] = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            small.append(i)
            if i != n // i:
                large.append(n // i)
    return small + large[::-1]


def compute_parallelism(gpu_count: int, model: "AIModel | None") -> ParallelismPlan:
    """Compute optimal (TP, PP) for vLLM/SGLang given a GPU count and model.

    Strategy for PCIe topology (no NVLink):
      - Prefer smallest TP to minimise cross-GPU tensor traffic.
      - TP must divide num_attention_heads, the linear-attention head counts
        (hybrid models) and intermediate_size.  num_key_value_heads must be
        divisible by TP, or TP a multiple of it (servers replicate KV heads).
      - For architectures not in ``_TP_SCOPED_ARCHS``, intermediate_size must
        also be divisible by the *total* GPU count.  Some model
        implementations assert ``intermediate_size % world_size == 0`` using
        the global process count rather than the TP group size; this check
        prevents those assertion failures by reducing the GPU count to the
        largest compatible value.
      - Architectures in ``_NO_PP_ARCHS`` get the largest valid TP with PP=1.
    """
    if model is None or gpu_count <= 1:
        return ParallelismPlan(tp=1, pp=max(gpu_count, 1), gpu_count=max(gpu_count, 1))

    cfg = read_parallelism_fields(model.path)
    if cfg is None:
        return ParallelismPlan(tp=1, pp=gpu_count, gpu_count=gpu_count)

    n_kv_heads   = cfg["num_key_value_heads"]
    intermediate = cfg["intermediate_size"]
    n_layers     = cfg["num_hidden_layers"]

    # Dimensions that TP must evenly divide.
    tp_dims = [
        d for d in (
            cfg["num_attention_heads"], intermediate,
            cfg["linear_num_key_heads"], cfg["linear_num_value_heads"],
        ) if d > 0
    ]

    def _tp_ok(tp: int) -> bool:
        kv_ok = n_kv_heads <= 0 or n_kv_heads % tp == 0 or tp % n_kv_heads == 0
        return kv_ok and all(d % tp == 0 for d in tp_dims)

    arch = model.arch or cfg.get("architecture", "")

    if arch in _NO_PP_ARCHS:
        tp = max(t for t in range(1, gpu_count + 1) if _tp_ok(t))
        note = "" if tp == gpu_count else f"{arch} does not support pipeline parallelism — TP only"
        return ParallelismPlan(tp=tp, pp=1, gpu_count=tp, note=note)

    # For architectures not in _TP_SCOPED_ARCHS, check intermediate_size %
    # total_gpus.  Fetch the arch from the catalog model or from config.json.
    check_global = (arch not in _TP_SCOPED_ARCHS) and intermediate > 0

    def _find_plan(n: int) -> ParallelismPlan | None:
        # Safety: some model implementations assert intermediate_size %
        # world_size == 0 using the global process count rather than the
        # TP group size.  Skip GPU counts that would trigger such failures
        # for architectures not known to be safe.
        if check_global and intermediate % n != 0:
            return None
        for tp in _sorted_divisors(n):
            pp = n // tp
            if _tp_ok(tp) and (n_layers == 0 or n_layers >= pp):
                return ParallelismPlan(tp=tp, pp=pp, gpu_count=n)
        return None

    # Try the requested GPU count first.
    plan = _find_plan(gpu_count)
    if plan is not None:
        return plan

    # Reduce until we find a compatible count.
    for n in range(gpu_count - 1, 0, -1):
        plan = _find_plan(n)
        if plan is not None:
            plan.note = f"no valid TP×PP split for {gpu_count} GPUs with this model's dimensions"
            return plan

    return ParallelismPlan(tp=1, pp=1, gpu_count=1)


# ---------------------------------------------------------------------------
# Per-server command builders
# ---------------------------------------------------------------------------

def _chat_template_args(server: ServerSoftware, p: dict) -> list[str]:
    """Override flag for the patched chat template (see chat_template.py), if any."""
    return [server.chat_template_flag, p["_chat_template"]] if p.get("_chat_template") else []


def _build_vllm(server: ServerSoftware, env: Environment, model: AIModel, p: dict) -> list[str]:
    cmd = [
        str(env.python), "-m", "vllm.entrypoints.openai.api_server",
        "--model",                        p["_model_path"],
        "--served-model-name",            model.id,
        "--tensor-parallel-size",         str(p["tensor_parallel"]),
        "--pipeline-parallel-size",       str(p["pipeline_parallel"]),
        "--gpu-memory-utilization",       str(p["gpu_mem_util"]),
        "--max-model-len",                "-1" if p.get("auto_fit_ctx") else str(server.launch_ctx(model)),
        "--host",                         "0.0.0.0",
        "--port",                         str(p["port"]),
    ]
    # Executor backend: omit to let vLLM choose (mp for single-node); only pass when
    # explicitly set (e.g. "ray" for multi-node or "mp" to force it).
    if p.get("executor_backend"):
        cmd += ["--distributed-executor-backend", p["executor_backend"]]
    # dtype: "auto" (default) respects the model's config.json torch_dtype; only pass
    # the flag when a specific dtype is requested per-model.
    if p.get("dtype") and p["dtype"] != "auto":
        cmd += ["--dtype", p["dtype"]]
    # Concurrency cap: 0 means use vLLM's own default (256).
    if p.get("max_num_seqs"):
        cmd += ["--max-num-seqs", str(p["max_num_seqs"])]
    if p.get("chunked_prefill"):
        cmd += ["--enable-chunked-prefill"]
    if p.get("expert_parallel"):
        cmd += ["--enable-expert-parallel"]
    if p.get("mtp") and model.mtp_embedded:
        spec = {"method": "mtp", "num_speculative_tokens": int(p["mtp_tokens"])}
        cmd += ["--speculative-config", json.dumps(spec, separators=(",", ":"))]
    if p.get("enforce_eager"):
        cmd += ["--enforce-eager"]
    # Only pass --trust-remote-code for models that actually require custom code.
    if model.trust_remote_code:
        cmd += ["--trust-remote-code"]
    if not p.get("log_requests"):
        cmd += ["--no-enable-log-requests"]
    rq = p.get("_runtime_quant")
    if rq == RuntimeQuant.BNB_INT8:
        cmd += ["--quantization", "bitsandbytes", "--load-format", "bitsandbytes"]
    if model.tool_call_parser:
        cmd += ["--enable-auto-tool-choice", "--tool-call-parser", model.tool_call_parser]
    if model.reasoning_parser:
        cmd += ["--reasoning-parser", model.reasoning_parser]
    if server.applies_rope_scaling(model):
        cmd += ["--hf-overrides", json.dumps(_hf_rope_override(model))]
    cmd += _chat_template_args(server, p)
    cmd.extend(p.get("extra_flags", []))
    return cmd


def _hf_rope_override(model: AIModel) -> dict[str, Any]:
    """config.json override enabling model.rope_scaling (vLLM --hf-overrides, SGLang --json-model-override-args).

    Built from the model's own RoPE block: servers replace the block wholesale, so
    architecture fields such as mrope_section, partial_rotary_factor and rope_theta
    must be carried over.  Multimodal checkpoints nest it under text_config;
    transformers ≥ 5 names it rope_parameters, older configs rope_scaling.
    """
    rs = model.rope_scaling
    assert rs is not None
    root = Path(model.path)
    cfg_path = (root if root.is_dir() else root.parent) / "config.json"
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        cfg = {}
    nested = isinstance(cfg.get("text_config"), dict)
    section = cfg["text_config"] if nested else cfg
    key = "rope_parameters" if "rope_parameters" in section else "rope_scaling"
    rope = dict(section.get(key) or {})
    if "type" in rope:
        rope["type"] = rs.method  # legacy spelling of rope_type
    rope.update(
        rope_type=rs.method,
        factor=rs.factor,
        original_max_position_embeddings=model.rope_original_ctx,
    )
    return {"text_config": {key: rope}} if nested else {key: rope}


def _llamacpp_rope_args(model: AIModel) -> list[str]:
    """llama-server flags applying model.rope_scaling (also forwarded by ooba)."""
    rs = model.rope_scaling
    assert rs is not None
    return [
        "--rope-scaling",  rs.method,
        "--rope-scale",    f"{rs.factor:g}",
        "--yarn-orig-ctx", str(model.rope_original_ctx),
    ]


def _build_llamacpp(server: ServerSoftware, env: Environment, model: AIModel, p: dict) -> list[str]:
    cmd = [
        str(env.executable("llama-server")),
        *llamacpp_model_args(server, model, p),
        "--alias",        model.id,
        "--host",         "0.0.0.0",
        "--port",         str(p["port"]),
        "--log-prefix",
        "--log-timestamps",
    ]
    if p["threads"]:
        cmd += ["--threads", str(p["threads"])]
    if p["cache_ram_mib"] is not None:
        cmd += ["--cache-ram", str(p["cache_ram_mib"])]
    if model.mmproj_path:
        cmd += ["--mmproj", model.mmproj_path]
    cmd += _mtp_args(model, p)
    cmd += _chat_template_args(server, p)
    cmd.extend(p.get("extra_flags", []))
    return cmd


def _mtp_args(model: AIModel, p: dict) -> list[str]:
    """Speculative-decoding flags using the model's MTP head (llama-server and ooba share them).

    Param "mtp" arrives resolved (see the server's AutoBoolParam): auto is on only
    for a sidecar head (mtp_path); on also uses an embedded head; off disables.
    Embedded heads are opt-in because many GGUFs keep NextN layers that a given
    llama.cpp build cannot run for that architecture.  An embedded head runs on the
    main model's own weights, so no draft file is passed.
    """
    if not p.get("mtp"):
        return []
    if model.mtp_path:
        return ["--spec-type", "draft-mtp", "--model-draft", model.mtp_path]
    if model.mtp_embedded:
        return ["--spec-type", "draft-mtp"]
    return []


def llamacpp_model_args(server: ServerSoftware, model: AIModel, p: dict) -> list[str]:
    """Model, placement and context arguments shared by llama-server and llama-fit-params.

    With fixed slots llama.cpp divides --ctx-size evenly between them, so it is
    emitted as the per-request context × parallel to give every slot the full context.
    """
    args = [
        "--model",        p["_model_path"],
        "--n-gpu-layers", str(p["n_gpu_layers"]),
        "--split-mode",   p["split_mode"],
        "--ctx-size",     str(server.launch_ctx(model) * p["parallel"]),
        "--parallel",     str(p["parallel"]),
    ]
    if p["kv_cache_type"]:
        args += ["--cache-type-k", p["kv_cache_type"], "--cache-type-v", p["kv_cache_type"]]
    if p.get("ubatch"):
        args += ["--ubatch-size", str(p["ubatch"])]
    if server.applies_rope_scaling(model):
        args += _llamacpp_rope_args(model)
    return args


def _build_tabby(server: ServerSoftware, env: Environment, model: AIModel, p: dict) -> list[str]:
    # model_dir is a staging directory; launch() creates a symlink named model.id
    # there pointing to the model's path before this builder runs.
    cmd = [
        str(env.python), str(CONFIG.tabby_main),
        "--model-dir",    p["model_dir"],
        "--model-name",   model.id,
        "--cache-mode",   p["cache_mode"],
        "--max-seq-len",  str(server.launch_ctx(model)),
        "--host",         "0.0.0.0",
        "--port",         str(p["port"]),
        "--disable-auth", str(p["disable_auth"]),
    ]
    cmd += _chat_template_args(server, p)
    cmd.extend(p.get("extra_flags", []))
    return cmd


def _build_ooba(server: ServerSoftware, env: Environment, model: AIModel, p: dict) -> list[str]:
    # ooba resolves user_data_dir relative to CWD when no flag is given; since the
    # launcher runs from ~/ai-server-tools, we must pass the absolute path explicitly.
    user_data_dir = str(CONFIG.ooba_main.parent / "user_data")
    model_ref = _staged_name(server, model)
    if _is_split_gguf(model):
        # Name shard 1 explicitly: ooba's metadata reader takes an unsorted glob's first hit.
        model_ref += "/" + Path(model.path).name
    cmd = [
        str(env.python), str(CONFIG.ooba_main),
        "--model",          model_ref,
        "--model-dir",      p["model_dir"],
        "--user-data-dir",  user_data_dir,
        "--ctx-size",       str(server.launch_ctx(model)),
        "--api",
        "--api-port",       str(p["api_port"]),
        "--listen",
        "--listen-host",    "0.0.0.0",
        "--listen-port",    str(p["webui_port"]),
    ]
    if p.get("loader"):
        cmd += ["--loader", p["loader"]]
    if p.get("nowebui"):
        cmd.append("--nowebui")
    if model.format == ModelFormat.GGUF:
        cmd += ["--gpu-layers", str(p["gpu_layers"])]
        if p.get("tensor_split"):
            cmd += ["--tensor-split", p["tensor_split"]]
        if model.mmproj_path:
            cmd += ["--mmproj", model.mmproj_path]
        cmd += _mtp_args(model, p)
    if server.applies_rope_scaling(model):
        # ooba has no RoPE options of its own; it forwards these to its llama-server.
        cmd += ["--extra-flags", shlex.join(_llamacpp_rope_args(model))]
    if p.get("cache_type"):
        cmd += ["--cache-type", p["cache_type"]]
    rq = p.get("_runtime_quant")
    if rq == RuntimeQuant.BNB_INT8:
        cmd.append("--load-in-8bit")
    elif rq == RuntimeQuant.BNB_NF4:
        cmd.append("--load-in-4bit")
    cmd += _chat_template_args(server, p)
    cmd.extend(p.get("extra_flags", []))
    return cmd


def _build_sglang(server: ServerSoftware, env: Environment, model: AIModel, p: dict) -> list[str]:
    cmd = [
        str(env.python), "-m", "sglang.launch_server",
        "--model-path",            p["_model_path"],
        "--served-model-name",     model.id,
        "--dtype",                 "bfloat16",
        "--tensor-parallel-size",  str(p["tensor_parallel"]),
        "--pipeline-parallel-size", str(p["pipeline_parallel"]),
        "--mem-fraction-static",   str(p["mem_fraction"]),
        "--context-length",        str(server.launch_ctx(model)),
        "--trust-remote-code",
        "--host",                  "0.0.0.0",
        "--port",                  str(p["port"]),
        "--log-level",             "warning",
    ]
    if model.format == ModelFormat.GGUF:
        cmd += ["--load-format", "gguf"]
    elif (
        model.format == ModelFormat.INT8_BNB
        or p.get("_runtime_quant") == RuntimeQuant.BNB_INT8
    ):
        cmd += ["--load-format", "bitsandbytes", "--quantization", "bitsandbytes"]
    if model.tool_call_parser:
        sglang_name = _SGLANG_TOOL_PARSER_MAP.get(model.tool_call_parser, model.tool_call_parser)
        cmd += ["--tool-call-parser", sglang_name]
    if model.reasoning_parser:
        sglang_name = _SGLANG_REASONING_PARSER_MAP.get(model.reasoning_parser, model.reasoning_parser)
        cmd += ["--reasoning-parser", sglang_name]
    if server.applies_rope_scaling(model):
        cmd += ["--json-model-override-args", json.dumps(_hf_rope_override(model))]
    cmd += _chat_template_args(server, p)
    cmd.extend(p.get("extra_flags", []))
    return cmd


def _build_comfyui(server: ServerSoftware, env: Environment, model: AIModel, p: dict) -> list[str]:
    # ComfyUI discovers models via ~/ComfyUI/extra_model_paths.yaml at startup;
    # no model path is passed at launch time.
    cmd = [
        str(env.python), str(CONFIG.comfyui_main),
        "--listen", "0.0.0.0",
        "--port",   str(p["port"]),
    ]
    cmd.extend(p.get("extra_flags", []))
    return cmd


_BUILDERS = {
    "vllm":     _build_vllm,
    "llamacpp": _build_llamacpp,
    "tabby":    _build_tabby,
    "ooba":     _build_ooba,
    "sglang":   _build_sglang,
    "comfyui":  _build_comfyui,
}


# ---------------------------------------------------------------------------
# Quantization tool availability
# ---------------------------------------------------------------------------

# Availability is checked once per (tool, environment) and cached; exl2-convert
# requires a subprocess call (to check importability without triggering JIT compilation).
_tool_avail_cache: dict[tuple[str, ...], bool] = {}


def quant_tool_env(tool: QuantizationTool) -> Environment | None:
    """The environment a quantization tool runs from: its server's default environment."""
    return ENVIRONMENTS.default_for(tool.env_server_id)


def _tool_env(tool: QuantizationTool) -> Environment:
    env = quant_tool_env(tool)
    if env is None:
        raise ValueError(f"No environment registered for server {tool.env_server_id!r} (needed by {tool.label})")
    return env


def quant_tool_available(tool: QuantizationTool) -> bool:
    """Return True if the tool's required binary or Python package is present.

    Result is cached after the first call — safe to call in hot paths.
    """
    env = quant_tool_env(tool)
    if env is None or not env.exists():
        return False
    key = (tool.id, str(env.location), tool.python_executable)
    if key not in _tool_avail_cache:
        _tool_avail_cache[key] = _check_tool_available(tool, env)
    return _tool_avail_cache[key]


def _check_tool_available(tool: QuantizationTool, env: Environment) -> bool:
    if tool.avail_executable and not env.executable(tool.avail_executable).is_file():
        return False
    if tool.avail_script and not (env.bin_dir.parent.parent / tool.avail_script).is_file():
        return False
    modules = tool.avail_modules or ((tool.avail_module,) if tool.avail_module else ())
    if modules:
        interpreter = tool.python_executable or str(env.python)
        try:
            r = subprocess.run(
                [
                    interpreter, "-c",
                    "import importlib.util, sys; "
                    f"modules = {modules!r}; "
                    "raise SystemExit(0 if all(importlib.util.find_spec(m) for m in modules) else 1)",
                ],
                capture_output=True, timeout=5,
            )
            return r.returncode == 0
        except Exception:
            return False
    return True


def install_quant_tool(tool: QuantizationTool) -> tuple[bool, str]:
    """pip-install the tool's package into its environment and return (success, log_path).

    Logs stdout/stderr to /tmp/launcher_install_<id>.log.
    Updates the availability cache on success.
    Returns (False, "") when the tool has no pip_package.
    """
    if not tool.pip_package:
        return False, ""
    env = _tool_env(tool)
    log_path = f"/tmp/launcher_install_{tool.id}.log"
    with open(log_path, "w") as f:
        result = subprocess.run(
            [str(env.python), "-m", "pip", "install", tool.pip_package],
            stdout=f, stderr=subprocess.STDOUT,
            timeout=300,
        )
    success = result.returncode == 0
    _tool_avail_cache[(tool.id, str(env.location), tool.python_executable)] = success
    return success, log_path


# ---------------------------------------------------------------------------
# Quantization command builders
# ---------------------------------------------------------------------------

def _strip_gguf_quant(stem: str) -> str:
    """Strip a known GGUF quant suffix from a filename stem.

    "BeMag-123B-v0.1.Q8_0"  →  "BeMag-123B-v0.1"
    "MyModel.IQ3_XXS"        →  "MyModel"
    "NoQuant"                →  "NoQuant"   (returned unchanged)
    """
    upper = stem.upper()
    for q in GGUF_QUANTS_LONGEST_FIRST:
        suffix = f".{q.value}"
        if upper.endswith(suffix):
            return stem[: len(stem) - len(suffix)]
    return stem


def compute_quant_output_path(
    tool: QuantizationTool,
    model: AIModel,
    quant_option: str,
    dest: Location,
) -> Path:
    """Derive the output *directory* for a quantization job.

    Every job produces its own uniquely-named directory so that removal
    is always consistent: delete the directory and its contents.

    GGUF output:
        <dest.root>/<model.type_subdir>/<stem_no_quant>-GGUF-<quant>/
        Existing GGUF inputs have their quant suffix removed; HF directory
        inputs keep their source directory name.

    EXL2 convert:
        <dest.root>/<model.type_subdir>/<source_dirname>-exl2-<bpw>bpw/
        e.g. ~/fast_models/text-generation/BeMag-123B-v0.1-exl2-4.0bpw/
    """
    src = Path(model.path)
    output_root = dest.root / model.type_subdir
    if tool.output_format == ModelFormat.GGUF:
        stem = _strip_gguf_quant(src.stem) if model.format == ModelFormat.GGUF else src.name
        return output_root / f"{stem}-GGUF-{quant_option}"
    else:
        return output_root / f"{src.name}-exl2-{quant_option}bpw"


def _build_llama_quantize(
    tool: QuantizationTool,
    env: Environment,
    model: AIModel,
    quant_option: str,
    output_dir: Path,
) -> list[str]:
    # output_dir is the per-job directory; the actual file goes inside it.
    src = Path(model.path)
    stem = _strip_gguf_quant(src.stem)
    output_file = output_dir / f"{stem}.{quant_option}.gguf"
    return [
        str(env.executable("llama-quantize")),
        model.path,
        str(output_file),
        quant_option,
    ]


def _build_exl2_convert(
    tool: QuantizationTool,
    env: Environment,
    model: AIModel,
    quant_option: str,
    output_path: Path,
) -> list[str]:
    script = CONFIG.exl2_convert_script
    if script.exists():
        cmd_prefix = [str(env.python), str(script)]
    else:
        cmd_prefix = [str(env.python), "-m", "exllamav2.conversion.convert_exl2"]
    return [
        *cmd_prefix,
        "-i", model.path,
        "-o", str(output_path),
        "-b", quant_option,
    ]


def _build_llama_hf_to_gguf(
    tool: QuantizationTool,
    env: Environment,
    model: AIModel,
    quant_option: str,
    output_dir: Path,
) -> list[str]:
    source_root = env.bin_dir.parent.parent
    script = source_root / "convert_hf_to_gguf.py"
    if not script.is_file():
        raise FileNotFoundError(f"llama.cpp converter script not found: {script}")
    source = Path(model.path)
    if not source.is_dir():
        raise ValueError(f"HF-to-GGUF conversion requires a model directory: {source}")
    output_file = output_dir / f"{source.name}.{quant_option}.gguf"
    return [
        tool.python_executable or str(env.python),
        str(script),
        "--outfile", str(output_file),
        "--outtype", quant_option.lower(),
        str(source),
    ]


_QUANT_BUILDERS: dict[str, Any] = {
    "llama-quantize": _build_llama_quantize,
    "llama-hf-to-gguf": _build_llama_hf_to_gguf,
    "exl2-convert":   _build_exl2_convert,
}


def build_quantize_cmd(
    tool: QuantizationTool,
    model: AIModel,
    quant_option: str,
    dest: Location,
    cuda_devices: str | None = None,
) -> tuple[list[str], Path]:
    """Return (argv, output_path), including the job's explicit CUDA environment."""
    output_path = compute_quant_output_path(tool, model, quant_option, dest)
    builder = _QUANT_BUILDERS.get(tool.id)
    if builder is None:
        raise ValueError(f"No command builder registered for quantization tool: {tool.id!r}")
    cmd = builder(tool, _tool_env(tool), model, quant_option, output_path)
    if not tool.uses_cuda:
        return cmd, output_path
    env_cmd = [
        "env",
        f"CUDA_VISIBLE_DEVICES={cuda_devices if cuda_devices is not None else CONFIG.resolved_cuda_devices}",
    ]
    if tool.cuda_home:
        env_cmd.append(f"CUDA_HOME={tool.cuda_home}")
    return [*env_cmd, *cmd], output_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def launch_params(
    server: ServerSoftware,
    model: "AIModel | None",
    runtime_quant: "RuntimeQuant | None" = None,
    param_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolved builder params: merged defaults/overrides with "auto" values resolved,
    plus the injected _model_path / _runtime_quant / _chat_template."""
    p = _merged(server, model, param_overrides)
    for spec, value, _reason in _resolve_auto(server, model, p):
        p[spec.key] = value
    if model is not None:
        p["_model_path"] = model.path
    p["_runtime_quant"] = runtime_quant
    template = patched_template(model) if model is not None and server.chat_template_flag else None
    p["_chat_template"] = str(template.path) if template is not None else ""  # written by launch()
    return p


def _resolve_auto(
    server: ServerSoftware, model: "AIModel | None", merged: dict[str, Any],
) -> list[tuple[AutoBoolParam, bool, str]]:
    return [
        (spec, *spec.resolve(model, merged))
        for spec in server.auto_params()
        if merged.get(spec.key) is None
    ]


def auto_param_states(
    server: ServerSoftware, model: "AIModel | None", param_overrides: dict[str, Any] | None = None,
) -> list[tuple[AutoBoolParam, bool, str]]:
    """(spec, resolved value, reason) for each parameter left at "auto" for this launch."""
    return _resolve_auto(server, model, _merged(server, model, param_overrides))


def gpu_param_overrides(
    server: ServerSoftware, model: "AIModel | None", gpu_count: int,
) -> tuple[dict[str, Any], "ParallelismPlan | None"]:
    """Parallelism params for launching on *gpu_count* GPUs, and the TP×PP plan when one applies.

    vLLM / SGLang (gpu_count_param "pipeline_parallel") get a planned TP×PP split
    (which may use fewer GPUs — see plan.gpu_count); other servers with a
    gpu_count_param receive the count itself.
    """
    if server.gpu_count_param == "pipeline_parallel" and model is not None:
        plan = compute_parallelism(gpu_count, model)
        return {"tensor_parallel": plan.tp, "pipeline_parallel": plan.pp}, plan
    if server.gpu_count_param:
        return {server.gpu_count_param: gpu_count}, None
    return {}, None


def build_cmd(server: ServerSoftware, env: Environment, model: "AIModel | None", runtime_quant: "RuntimeQuant | None" = None, param_overrides: dict[str, Any] | None = None) -> list[str]:
    """Return the full argv list for launching model on server.

    model may be None for servers that use stage_all_models (e.g. ComfyUI),
    where no single model is selected at launch time.

    param_overrides is merged on top of default_params and model.server_params;
    used by GpuSelectScreen to inject gpu_count_param (e.g. pipeline_parallel)
    and the concurrency_param.
    """
    p = launch_params(server, model, runtime_quant, param_overrides)
    builder = _BUILDERS.get(server.id)
    if builder is None:
        raise ValueError(f"No command builder registered for server: {server.id!r}")
    if server.stage_all_models:
        return builder(server, env, model, p)  # type: ignore[arg-type]  # builder ignores model
    assert model is not None, f"model is required for server {server.id!r}"
    return builder(server, env, model, p)


def format_cmd(cmd: list[str]) -> str:
    """Human-readable multi-line shell representation of a command."""
    return " \\\n  ".join(shlex.quote(c) for c in cmd)


def _bundled_cuda_root(server: ServerSoftware, env: Environment) -> Path | None:
    """The pip-bundled CUDA toolkit root inside *env*, when the server declares one."""
    if not server.bundled_cuda_glob:
        return None
    return env.find_package_path(server.bundled_cuda_glob)


def _ensure_lib_symlinks(lib_dir: Path, links: dict[str, str]) -> None:
    """Create missing linker-name symlinks, e.g. libcudart.so → libcudart.so.13.

    Self-heals split-wheel CUDA toolkits that ship libfoo.so.NN but omit the
    libfoo.so name the linker needs.  The shortest match (the soname) is the target.
    """
    for link_name, target_glob in links.items():
        link = lib_dir / link_name
        targets = sorted((t for t in lib_dir.glob(target_glob) if t != link), key=lambda t: len(t.name))
        if not targets or link.exists():
            continue
        if link.is_symlink():
            link.unlink()  # replace a broken symlink
        link.symlink_to(targets[0].name)


def _is_split_gguf(model: AIModel) -> bool:
    return model.format == ModelFormat.GGUF and len(gguf_shard_paths(Path(model.path))) > 1


def _staged_name(server: ServerSoftware, model: AIModel) -> str:
    """Name of the entry _stage_model() creates for *model* in the staging dir."""
    if _is_split_gguf(model):
        return model.id
    if server.staging_use_source_filename:
        return Path(model.path).name
    return model.id + server.staging_format_suffixes.get(model.format, "")


def _stage_model(server: ServerSoftware, model: AIModel, staging: Path) -> None:
    """Link *model* into *staging* under _staged_name(), replacing any previous entry.

    A split GGUF is staged as a directory of per-shard links that keep their
    original names: llama.cpp locates shards 2..N from the first shard's name and
    directory, so a single renamed link to shard 1 cannot load.
    """
    entry = staging / _staged_name(server, model)
    if entry.is_symlink():
        entry.unlink()
    elif entry.is_dir():
        # Only ever remove a dir we created (links only) — never real model data.
        if not all(child.is_symlink() for child in entry.iterdir()):
            raise FileExistsError(f"Staging path {entry} is a real directory; refusing to replace it")
        shutil.rmtree(entry)
    if not _is_split_gguf(model):
        entry.symlink_to(model.path)
        return
    entry.mkdir()
    for shard in gguf_shard_paths(Path(model.path)):
        (entry / shard.name).symlink_to(shard)


def _one_copy_each(models: list[AIModel]) -> list[AIModel]:
    """One entry per model: copies collapse to the first present on disk, in [[locations]] order."""
    def preference(m: AIModel) -> tuple[bool, int]:
        loc = location_of(m.path)
        return not Path(m.path).exists(), LOCATIONS.index(loc) if loc is not None else len(LOCATIONS)

    chosen: dict[tuple[str, str], AIModel] = {}
    for m in sorted(models, key=preference):
        chosen.setdefault((m.id, m.format.value), m)
    return list(chosen.values())


def _prepend_path(entry: str, var: str) -> str:
    """Prepend *entry* to the current value of a PATH-style environment variable."""
    existing = os.environ.get(var, "")
    return f"{entry}:{existing}" if existing else entry


def build_launch_command(
    server: ServerSoftware,
    env: Environment,
    model: "AIModel | None",
    runtime_quant: "RuntimeQuant | None" = None,
    cuda_devices: str | None = None,
    param_overrides: dict[str, Any] | None = None,
    power_step: PowerLimitStep | None = None,
) -> str:
    """Return the complete paste-ready shell command for a server launch.

    power_step: GPU power limits applied first (``&&`` — the server never starts
    with the limits unapplied).
    """
    env_cmd = ["env"]
    if server.strip_pythonpath:
        env_cmd += ["-u", "PYTHONPATH"]
    env_vars: dict[str, str] = {
        "CUDA_VISIBLE_DEVICES": cuda_devices if cuda_devices is not None else CONFIG.resolved_cuda_devices,
    }
    cuda_root = _bundled_cuda_root(server, env)
    cuda_home = str(cuda_root) if cuda_root is not None else server.cuda_home
    if cuda_home:
        env_vars["CUDA_HOME"] = cuda_home
    if cuda_root is not None:
        # Existing values are captured inline so the logged command reproduces exactly.
        lib_dir = str(cuda_root / "lib")
        env_vars["LD_LIBRARY_PATH"] = _prepend_path(lib_dir, "LD_LIBRARY_PATH")
        env_vars["LIBRARY_PATH"]    = _prepend_path(lib_dir, "LIBRARY_PATH")
    env_vars.update(server.extra_env)
    if model is not None and server.applies_rope_scaling(model):
        env_vars.update(server.rope_scaling_env)

    cmd = build_cmd(server, env, model, runtime_quant=runtime_quant, param_overrides=param_overrides)
    command = format_cmd([*env_cmd, *(f"{key}={value}" for key, value in env_vars.items()), *cmd])
    if server.launch_cwd:
        command = f"cd {shlex.quote(server.launch_cwd)} && \\\n  {command}"
    if power_step is not None:
        command = f"{power_step.command()} && \\\n  {command}"
    return command


def _spawn_with_pty(
    command: str,
    env: dict[str, str],
) -> "tuple[subprocess.Popen[bytes], int]":
    """Run a shell command in a PTY sized to the current terminal.

    Returns (proc, master_fd).  The caller is responsible for closing master_fd
    after the tee thread finishes reading from it.
    """
    master_fd, slave_fd = pty.openpty()
    # Match the PTY dimensions to the real terminal so width-sensitive output
    # (tqdm progress bars, torch-compile banners, etc.) renders correctly.
    try:
        term_cols, term_rows = shutil.get_terminal_size()
        fcntl.ioctl(
            slave_fd, termios.TIOCSWINSZ,
            struct.pack("HHHH", term_rows, term_cols, 0, 0),
        )
    except Exception:
        pass  # non-fatal — falls back to whatever tqdm/server defaults to
    # Wrap in bash so 'Segmentation fault (core dumped)' and other shell-level
    # signal diagnostics appear in the log.  Bash exits with 128+signum.
    bash_cmd = ["bash", "-c", command]
    proc = subprocess.Popen(
        bash_cmd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=slave_fd,
        stderr=slave_fd,
    )
    os.close(slave_fd)  # child has its own copy; close parent's reference
    return proc, master_fd


def launch(server: ServerSoftware, env: Environment, model: "AIModel | None", runtime_quant: "RuntimeQuant | None" = None, cuda_devices: str | None = None, param_overrides: dict[str, Any] | None = None, power_step: PowerLimitStep | None = None, command: str | None = None) -> tuple[subprocess.Popen[bytes], Path, int]:
    """Build the command and spawn the server process.

    env: the Environment (conda env / native build) providing the server's interpreter or binaries.
    cuda_devices: overrides CONFIG.resolved_cuda_devices when set (from GpuSelectScreen).
    param_overrides: merged on top of server/model params (e.g. pipeline_parallel → GPU count).
    power_step: GPU power limits applied before the server starts (from GpuSelectScreen).
    command: complete shell command override from ConfirmLaunch; executed verbatim when set.
    """
    proc_env = os.environ.copy()

    cuda_root = _bundled_cuda_root(server, env)
    if cuda_root is not None and server.lib_symlinks:
        _ensure_lib_symlinks(cuda_root / "lib", server.lib_symlinks)

    if server.uses_staging_dir:
        p = _merged(server, model)
        staging_base = Path(p["model_dir"])
        staging_base.mkdir(parents=True, exist_ok=True)
        # stage_all_models: create symlinks for every compatible model in the catalog.
        # Used by servers (e.g. ComfyUI) where the client picks the model per-request.
        # stage single model: used by servers that load exactly one model at startup.
        if server.stage_all_models:
            models_to_stage: list[AIModel] = _one_copy_each(
                [m for m in AI_MODELS if m.compatible_with(server)]
            )
        else:
            assert model is not None, f"model is required for server {server.id!r}"
            models_to_stage = [model]
        for m in models_to_stage:
            # staging_subdir_attr: route each model into the correct subdirectory.
            # e.g. ComfyUI uses model.comfyui_subdir ('checkpoints' or 'diffusion_models').
            if server.staging_subdir_attr:
                subdir = getattr(m, server.staging_subdir_attr, "")
                staging = staging_base / subdir if subdir else staging_base
                staging.mkdir(parents=True, exist_ok=True)
            else:
                staging = staging_base
            _stage_model(server, m, staging)

    template = patched_template(model) if model is not None and server.chat_template_flag else None
    if template is not None:
        template.write()  # referenced by the command's chat_template_flag

    _LOGS_DIR.mkdir(exist_ok=True)
    model_slug = model.id if model is not None else "no-model"
    log_path = _LOGS_DIR / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{server.id}_{model_slug}.log"
    if command is None:
        command = build_launch_command(
            server,
            env,
            model,
            runtime_quant=runtime_quant,
            cuda_devices=cuda_devices,
            param_overrides=param_overrides,
            power_step=power_step,
        )
    quant_str = f" + {runtime_quant.value}" if runtime_quant else ""
    if model is not None:
        model_line = f"# model:  {model.id}  ({model.format.value}{quant_str}, {model.size_gb} GB)\n"
        path_line  = f"# path:   {model.path}\n"
    else:
        model_line = "# model:  (all catalog models staged)\n"
        path_line  = ""
    versions = server.env_version_label(env)
    env_detail = f"{env.kind_label}, {versions}" if versions else env.kind_label
    template_line = (
        f"# tmpl:   {template.path}  (patched: {', '.join(p.id for p in template.patches)})\n"
        if template is not None else ""
    )
    header = (
        f"# launcher session log\n"
        f"# date:   {datetime.now().isoformat(timespec='seconds')}\n"
        f"{model_line}"
        f"# server: {server.id}\n"
        f"# env:    {env.name}  ({env_detail})\n"
        f"{path_line}"
        f"{template_line}"
        f"#\n"
        f"# command:\n"
        f"{command}\n"
        f"# {'─' * 72}\n"
    )
    log_path.write_text(header, encoding="utf-8")
    proc, master_fd = _spawn_with_pty(command, proc_env)
    return proc, log_path, master_fd


def quantize(
    tool: QuantizationTool,
    model: AIModel,
    quant_option: str,
    dest: Location,
    cuda_devices: str | None = None,
) -> tuple[subprocess.Popen[bytes], Path, int]:
    """Build the quantization command and spawn the process.

    Follows the same PTY + log pattern as launch() so the caller can use
    the same tee / SIGWINCH / exit-footer machinery in app.py.
    Returns (proc, log_path, master_fd).
    """
    cmd, output_path = build_quantize_cmd(tool, model, quant_option, dest, cuda_devices)

    # output_path is always a directory — create it before the process starts.
    output_path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()

    _LOGS_DIR.mkdir(exist_ok=True)
    safe_quant = quant_option.replace("/", "-")
    log_path = _LOGS_DIR / (
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        f"_quantize_{tool.id}_{model.id}_{safe_quant}.log"
    )
    header = (
        f"# launcher quantize log\n"
        f"# date:    {datetime.now().isoformat(timespec='seconds')}\n"
        f"# model:   {model.id}  ({model.format.value}, {model.size_gb} GB)\n"
        f"# tool:    {tool.id}\n"
        f"# option:  {quant_option}\n"
        f"# source:  {model.path}\n"
        f"# output:  {output_path}\n"
        f"#\n"
        f"# command:\n"
        f"{format_cmd(cmd)}\n"
        f"# {'─' * 72}\n"
    )
    log_path.write_text(header, encoding="utf-8")
    proc, master_fd = _spawn_with_pty(format_cmd(cmd), env)
    return proc, log_path, master_fd


def finalize_quantization(job: QuantizeJob) -> AIModel:
    """Copy model metadata, catalog a successful output, and return its model.

    The output inherits the source model's intrinsic catalog fields
    (AIModel.derived_entry_fields) on top of what ingest detects from disk.
    """
    from launcher.ingest import copy_model_metadata, ingest_model

    copy_model_metadata(model_metadata_dir(Path(job.model.path)), job.output_path)
    ingest_path = job.output_path
    if job.tool.output_format == ModelFormat.EXL2:
        _remove_exl2_work_artifacts(job.output_path)
    elif job.tool.output_format == ModelFormat.GGUF:
        gguf_files = sorted(job.output_path.glob("*.gguf"))
        if not gguf_files:
            raise FileNotFoundError(f"GGUF conversion produced no .gguf output in {job.output_path}")
        first_shard = next(
            (p for p in gguf_files if (info := gguf_split_info(p)) is not None and info[1] == 1),
            None,
        )
        ingest_path = first_shard or gguf_files[0]
        missing_shards = [p for p in gguf_shard_paths(ingest_path) if not p.is_file()]
        if missing_shards:
            raise FileNotFoundError(f"GGUF output is missing shard(s): {', '.join(map(str, missing_shards))}")

    entry_overrides = {
        **job.model.derived_entry_fields(),
        "id": job.output_path.name.lower().replace(".", "-"),
        "name": job.model.name,
        "quant": (
            f"{job.quant_option} bpw"
            if job.tool.output_format == ModelFormat.EXL2
            else job.quant_option.upper()
        ),
    }
    entry = ingest_model(
        ingest_path,
        fmt=job.tool.output_format.value,
        entry_overrides=entry_overrides,
    )
    for model in AI_MODELS:
        if model.path == entry["path"] and model.format.value == entry["format"]:
            return model

    model = AIModel.from_dict(entry)
    AI_MODELS.append(model)
    return model


# ---------------------------------------------------------------------------
# Copies and deletion — catalog bookkeeping (file operations live in storage.py)
# ---------------------------------------------------------------------------

def copies_of(model: AIModel) -> list[AIModel]:
    """Other catalog entries holding the same model at other paths."""
    return [m for m in AI_MODELS if m.is_copy_of(model)]


def register_copy(model: AIModel, dest: Location) -> list[AIModel]:
    """Add catalog entries for *model* copied to *dest* (after storage.copy_model() succeeded).

    Every entry sharing the model's path (several formats in one directory) gets a
    copy — a snapshot of all its fields at its mirrored path.  Entries already
    catalogued at *dest* are left as they are.  Returns the entries at *dest*.
    """
    catalog = _load_catalog()
    keys = {_entry_key(e) for e in catalog["models"]}
    copied: list[AIModel] = []
    for entry in [e for e in catalog["models"] if e["path"] == model.path]:
        relocated = relocated_entry(entry, dest)
        key = _entry_key(relocated)
        if key not in keys:
            catalog["models"].append(relocated)
            keys.add(key)
            AI_MODELS.append(AIModel.from_dict(relocated))
        copied.append(next(m for m in AI_MODELS if m.key == key))
    _save_catalog(catalog)
    return copied


def delete_model(plan: DeletionPlan) -> None:
    """Delete the files in *plan* (see storage.plan_deletion), then drop its catalog entries."""
    plan.execute()
    keys = {m.key for m in plan.entries}
    catalog = _load_catalog()
    catalog["models"] = [e for e in catalog["models"] if _entry_key(e) not in keys]
    _save_catalog(catalog)
    AI_MODELS[:] = [m for m in AI_MODELS if m.key not in keys]


def save_server_status(
    model: AIModel,
    server_id: str,
    env_name: str,
    status: str,
    quant: "RuntimeQuant | None" = None,
) -> None:
    """Persist a model/server/environment/quant status result back into models_local.json.

    Mutates the in-memory model immediately, then rewrites the server_status
    entry for that model in the JSON catalog so it survives the next launch.
    """
    qk = _quant_key(quant)

    # 1. Update in-memory object immediately.
    model.server_status.setdefault(server_id, {}).setdefault(env_name, {})[qk] = status  # type: ignore[assignment]

    # 2. Persist to JSON catalog.
    def _apply(entry: dict) -> None:
        env_dict = entry.setdefault("server_status", {}).setdefault(server_id, {}).setdefault(env_name, {})
        env_dict[qk] = status

    _patch_catalog(model, _apply)


def _rewrite_env_status(server_id: str, old_name: str, new_name: str | None) -> int:
    """Move (new_name set) or drop (None) every model's results for one environment.

    Applies to in-memory models and the JSON catalog in a single write.
    Returns the number of models whose results were affected.
    """
    def _move(status: dict) -> bool:
        per_env = status.get(server_id, {})
        if old_name not in per_env:
            return False
        results = per_env.pop(old_name)
        if new_name is not None:
            per_env[new_name] = results
        if not per_env:
            status.pop(server_id, None)
        return True

    for m in AI_MODELS:
        _move(m.server_status)
    with open(_CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)
    affected = sum(_move(entry.get("server_status", {})) for entry in catalog["models"])
    with open(_CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return affected


def count_env_status(server_id: str, env_name: str) -> int:
    """Number of models with recorded results for one environment of a server."""
    return sum(1 for m in AI_MODELS if env_name in m.server_status.get(server_id, {}))


def apply_env_rename(env: Environment, new_name: str) -> None:
    """Rename *env* (and registry entries sharing it) and migrate their recorded results.

    Call after the underlying environment has been renamed (env.rename_cmd()).
    """
    old_name = env.name
    for renamed in ENVIRONMENTS.rename(env, new_name):
        _rewrite_env_status(renamed.server_id, old_name, new_name)


def apply_env_removal(env: Environment) -> None:
    """Unregister *env* and drop the test results recorded against it."""
    ENVIRONMENTS.remove(env)
    _rewrite_env_status(env.server_id, env.name, None)


def save_ctx_override(model: AIModel, value: int) -> None:
    """Persist ctx_override for a model.  value=0 clears it (reverts to native ctx).

    Mutates the in-memory model immediately, then updates the JSON catalog.
    """
    model.ctx_override = value

    def _apply(entry: dict) -> None:
        if value:
            entry["ctx_override"] = value
        else:
            entry.pop("ctx_override", None)

    _patch_catalog(model, _apply)


def save_rope_scaling(model: AIModel, rope_scaling: RopeScaling | None) -> None:
    """Persist rope_scaling for a model.  None clears it (native context, unscaled).

    Mutates the in-memory model immediately, then updates the JSON catalog.
    """
    model.rope_scaling = rope_scaling

    def _apply(entry: dict) -> None:
        if rope_scaling is not None:
            entry["rope_scaling"] = rope_scaling.to_dict()
        else:
            entry.pop("rope_scaling", None)

    _patch_catalog(model, _apply)


def save_favorite(model: AIModel, favorite: bool) -> None:
    """Persist the favorite flag for a model.

    Mutates the in-memory model immediately, then updates the JSON catalog.
    favorite=False removes the key from JSON to keep the catalog tidy.
    """
    model.favorite = favorite

    def _apply(entry: dict) -> None:
        if favorite:
            entry["favorite"] = True
        else:
            entry.pop("favorite", None)

    _patch_catalog(model, _apply)


def save_server_param(model: AIModel, server: ServerSoftware, key: str, value: Any) -> None:
    """Persist one per-model server param override (e.g. the last-launched slot count).

    A value equal to the server default removes the override, keeping the catalog
    tidy (so saving the default is how an override is reset).  Mutates the in-memory
    model immediately, then updates the JSON catalog.
    """
    default = server.param(key).default_value()

    def _apply(server_params: dict) -> None:
        params = server_params.setdefault(server.id, {})
        if value == default:
            params.pop(key, None)
        else:
            params[key] = value
        if not params:
            server_params.pop(server.id)

    def _apply_entry(entry: dict) -> None:
        _apply(entry.setdefault("server_params", {}))
        if not entry["server_params"]:
            entry.pop("server_params")

    _apply(model.server_params)
    _patch_catalog(model, _apply_entry)


def save_model_field(model: AIModel, spec: ParamSpec, value: Any) -> None:
    """Persist one user-editable AIModel field (see models.MODEL_FIELDS).

    The catalog omits the key at its default.  Mutates the in-memory model
    immediately, then updates the JSON catalog.
    """
    setattr(model, spec.key, value)

    def _apply(entry: dict) -> None:
        if value == spec.default_value():
            entry.pop(spec.key, None)
        else:
            entry[spec.key] = value

    _patch_catalog(model, _apply)


def save_template_patch_enabled(model: AIModel, patch_id: str, enabled: bool) -> None:
    """Persist a model's opt-out of one chat template patch (enabled=False opts out).

    Mutates the in-memory model immediately, then updates the JSON catalog; the
    key is omitted when no patch is disabled.
    """
    disabled = set(model.disabled_template_patches)
    if enabled:
        disabled.discard(patch_id)
    else:
        disabled.add(patch_id)
    model.disabled_template_patches = frozenset(disabled)

    def _apply(entry: dict) -> None:
        if disabled:
            entry["disabled_template_patches"] = sorted(disabled)
        else:
            entry.pop("disabled_template_patches", None)

    _patch_catalog(model, _apply)
