"""
data.py — Generic server software registry, quantization tools and the model catalog.

Everything here is driven by config.toml and is safe to commit.
Your personal model inventory lives in models_local.py (gitignored).
Copy models_local.py.example → models_local.py to get started.

Adding a new server:
  1. Create a ServerSoftware() entry in SERVERS below: its launch parameters as
     typed ParamSpecs (params.py) plus any launch-time fields (env_kind,
     version_packages, strip_pythonpath, cuda_home, uses_staging_dir,
     staging_format_suffixes, launch_cwd) that the server requires.
  2. Create a matching builder function in launch.py and register it in _BUILDERS.
  3. Register at least one environment for it in user_data/environments.json.
"""
from __future__ import annotations

import json
from pathlib import Path

from typing import Any

from launcher.hf_config import expert_parallel_needed, fp8_block_size
from launcher.models import (
    AIModel, AIType, GGUFQuant, ModelFormat,
    HardwareRequirement, QuantizationTool, QuantOption, RuntimeQuant, RuntimeQuantOption,
    ServerSoftware, VersionProbe,
)
from launcher.params import (
    AutoBoolParam, BoolParam, ChoiceParam, FlagsParam, FloatParam, IntParam, TextParam,
)
from launcher.config import CONFIG


# ---------------------------------------------------------------------------
# Server software registry
# ---------------------------------------------------------------------------
# params declares every launch parameter (default, help, editor); their defaults are
# the baseline for every launch on that server.  A model's server_params[server.id]
# is merged on top at launch time; only values that differ from the defaults need
# to be specified there.  "auto" parameters are resolved per launch (launch_params).


def _port(server_key: str, fallback: int, key: str = "port", label: str = "Port") -> IntParam:
    return IntParam(key=key, label=label, default=CONFIG.server_ports.get(server_key, fallback),
                    minimum=1, maximum=65535)


_EXTRA_FLAGS = FlagsParam(
    key="extra_flags", label="Extra flags",
    help="Additional command-line arguments appended to the server command",
)


def _vllm_expert_parallel(model: AIModel | None, p: dict[str, Any]) -> tuple[bool, str]:
    return expert_parallel_needed(model, int(p["tensor_parallel"]))


def _sidecar_mtp(model: AIModel | None, p: dict[str, Any]) -> tuple[bool, str]:
    """llama.cpp-style MTP: auto uses a sidecar mtp-*.gguf head; embedded heads are opt-in."""
    if model is not None and model.mtp_path:
        return True, "sidecar MTP head"
    if model is not None and model.mtp_embedded:
        return False, "embedded head is opt-in (the build may not support it for this architecture)"
    return False, "no MTP head"


def _has_mtp_head(model: AIModel) -> bool:
    return bool(model.mtp_path or model.mtp_embedded)


def _gguf_mtp_head(model: AIModel) -> bool:
    """ooba applies MTP only through its llama.cpp loader (GGUF)."""
    return model.format == ModelFormat.GGUF and _has_mtp_head(model)


def _is_moe(model: AIModel) -> bool:
    return model.moe_experts > 0


def _has_embedded_mtp(model: AIModel) -> bool:
    return model.mtp_embedded


_GPU_SCREEN = "set from the GPU selection"

# Checkpoints described by an HF config.json, which vLLM / SGLang can override at launch.
_HF_CONFIG_FORMATS: frozenset[ModelFormat] = frozenset({
    ModelFormat.SAFETENSORS_BF16,
    ModelFormat.SAFETENSORS_FP16,
    ModelFormat.SAFETENSORS_FP8,
    ModelFormat.COMPRESSED_TENSORS,
    ModelFormat.AWQ,
    ModelFormat.GPTQ,
    ModelFormat.INT8_BNB,
    ModelFormat.PYTORCH_BIN,
})

# vLLM and SGLang quantize unquantized checkpoints in flight through bitsandbytes —
# always to 4-bit NF4 (their loaders call quantize_4bit; 8-bit only loads pre-quantized
# checkpoints).
_BNB_NF4_INFLIGHT = RuntimeQuantOption(
    RuntimeQuant.BNB_NF4, ("--quantization", "bitsandbytes", "--load-format", "bitsandbytes"),
)

# vLLM online quantization (--quantization <shorthand>): weights are quantized layer by
# layer as they load, so the full-precision model never has to fit.  Requirements come
# from vLLM 0.26's kernel checks; on sm_86 FP8 per-tensor, INT8 experts (MoE) and
# bitsandbytes were verified, the others fail as their requirement predicts.
_VLLM_RUNTIME_QUANTS: tuple[RuntimeQuantOption, ...] = (
    RuntimeQuantOption(
        RuntimeQuant.FP8_BLOCK, ("--quantization", "fp8_per_block"),
        requires=HardwareRequirement(89, "FP8 block kernels (Triton has no FP8 type below sm_89)"),
    ),
    RuntimeQuantOption(
        RuntimeQuant.FP8_CHANNEL, ("--quantization", "fp8_per_channel"),
        requires=HardwareRequirement(89, "per-token FP8 activations need FP8 tensor cores"),
    ),
    RuntimeQuantOption(
        RuntimeQuant.FP8, ("--quantization", "fp8"),
        note="below sm_89 it runs weight-only (Marlin W8A16)",
    ),
    RuntimeQuantOption(
        RuntimeQuant.MXFP8, ("--quantization", "mxfp8"),
        requires=HardwareRequirement(100, "native MX kernels; the Marlin fallback failed on sm_86"),
    ),
    RuntimeQuantOption(
        RuntimeQuant.INT8_EXPERTS, ("--quantization", "int8_per_channel_weight_only"),
        note="attention and shared layers stay BF16",
        applies_to=_is_moe,
    ),
    RuntimeQuantOption(
        RuntimeQuant.NVFP4_EXPERTS, ("--quantization", "nvfp4_per_token"),
        requires=HardwareRequirement(100, "Blackwell FlashInfer TRT-LLM MoE kernels only"),
        applies_to=_is_moe,
    ),
    _BNB_NF4_INFLIGHT,
)

SERVERS: list[ServerSoftware] = [

    ServerSoftware(
        id="vllm",
        label="vLLM",
        supported_formats=_HF_CONFIG_FORMATS,
        supported_ai_types=frozenset({AIType.TEXT_GEN}),
        params=(
            IntParam(key="tensor_parallel", label="Tensor parallel", default=1, managed_by=_GPU_SCREEN),
            IntParam(key="pipeline_parallel", label="Pipeline parallel",
                     default=CONFIG.pipeline_parallel, managed_by=_GPU_SCREEN),
            ChoiceParam(
                key="executor_backend", label="Executor backend", default="",
                choices=("", "mp", "ray"), labels=(("", "auto"),),
                help="auto lets vLLM choose (mp for single-node — simpler and more robust than ray)",
            ),
            ChoiceParam(
                key="dtype", label="dtype", default="auto",
                choices=("auto", "bfloat16", "float16", "float32"),
                help="auto honours the model's torch_dtype; override only when a checkpoint's "
                     "native dtype is undesirable (e.g. a float32 checkpoint that would OOM)",
            ),
            FloatParam(
                key="gpu_mem_util", label="GPU memory utilization", default=0.90,
                minimum=0.1, maximum=1.0,
                help="vLLM's default 0.90 leaves headroom; 0.97 was a frequent cause of "
                     "load-time / CUDA-graph OOM",
            ),
            IntParam(
                key="max_num_seqs", label="Max concurrent sequences", default=0, minimum=0,
                help="0 = vLLM's default (256); lower for very large / memory-tight models",
            ),
            BoolParam(
                key="enforce_eager", label="Enforce eager", default=False,
                help="Disable CUDA graphs (slower); only if graph capture causes OOM",
            ),
            BoolParam(
                key="chunked_prefill", label="Chunked prefill", default=False,
                help="off = omit the flag (vLLM enables chunked prefill itself when helpful)",
            ),
            AutoBoolParam(
                key="expert_parallel", label="Expert parallel", resolve=_vllm_expert_parallel,
                applies_to=_is_moe,
                help="--enable-expert-parallel: distribute whole experts across the TP ranks "
                     "instead of sharding each.  auto turns it on when the experts cannot be "
                     "TP-sharded (width or FP8 block alignment); it has no effect at TP=1",
            ),
            BoolParam(
                key="auto_fit_ctx", label="Auto-fit context", default=False,
                help="--max-model-len -1: start from the native context and lower it only as far "
                     "as needed to fit the KV cache (logged as \"Auto-fit max_model_len\")",
            ),
            BoolParam(
                key="mtp", label="MTP speculative decoding", default=False,
                applies_to=_has_embedded_mtp,
                help="Use the checkpoint's multi-token-prediction layers as a draft "
                     "(--speculative-config method mtp).  Speeds up decoding but its weights "
                     "and draft KV cost memory (the fitted context shrinks)",
            ),
            IntParam(
                key="mtp_tokens", label="MTP draft tokens", default=2, minimum=1, maximum=8,
                applies_to=_has_embedded_mtp,
                help="Tokens drafted per step.  Beyond the checkpoint's MTP layer count (usually 1) "
                     "vLLM reuses the layer, and each extra position is accepted less often — "
                     "check \"SpecDecoding metrics\" in the log",
            ),
            BoolParam(
                key="log_requests", label="Log requests", default=False,
                help="off = --no-enable-log-requests (quieter logs)",
            ),
            _port("vllm", 5000),
            _EXTRA_FLAGS,
        ),
        quantizable_formats=frozenset({
            ModelFormat.SAFETENSORS_BF16,
            ModelFormat.SAFETENSORS_FP16,
            ModelFormat.PYTORCH_BIN,
        }),
        runtime_quants=_VLLM_RUNTIME_QUANTS,
        # Prevent the vllm source tree (if present in PYTHONPATH) from shadowing
        # the installed package.  CWD /tmp ensures no local vllm/ dir is on sys.path.
        version_packages=("vllm",),
        strip_pythonpath=True,
        launch_cwd="/tmp",
        # vLLM 0.25+ uses cu13 split wheels: the CUDA toolkit lives inside the env's
        # Python package tree (nvidia/cu13/) and is not automatically on LD_LIBRARY_PATH.
        # Resolved per environment at launch: CUDA_HOME points nvcc / JIT compilers at
        # the bundled toolkit; LIBRARY_PATH + the libcudart.so symlink let flashinfer's
        # runtime JIT link against the bundled cudart (the wheel ships only libcudart.so.NN).
        bundled_cuda_glob="nvidia/cu1[0-9]*",
        lib_symlinks={"libcudart.so": "libcudart.so.[0-9]*"},
        # pipeline_parallel must equal the number of GPUs selected; GpuSelectScreen
        # will auto-inject param_overrides["pipeline_parallel"] = selected_gpu_count.
        gpu_count_param="pipeline_parallel",
        # RoPE scaling via --hf-overrides; vLLM refuses a max-model-len past the
        # config's own limit unless explicitly allowed.
        rope_scaling_formats=_HF_CONFIG_FORMATS,
        rope_scaling_env={"VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"},
        chat_template_flag="--chat-template",
    ),

    ServerSoftware(
        id="llamacpp",
        label="llama.cpp",
        supported_formats=frozenset({ModelFormat.GGUF}),
        supported_ai_types=frozenset({AIType.TEXT_GEN}),
        params=(
            IntParam(
                key="n_gpu_layers", label="GPU layers", default="auto", minimum=-1, keywords=("auto",),
                help="Layers offloaded to GPU — auto, a count, or -1 for all",
            ),
            ChoiceParam(
                key="split_mode", label="Split mode", default="layer", choices=("layer", "row", "none"),
                help="How the model is split across the visible GPUs",
            ),
            IntParam(
                key="parallel", label="Request slots", default=1, minimum=1,
                help="Concurrent request slots, each with the model's full context (--ctx-size is "
                     "emitted as context × slots).  Also adjustable per launch with +/- on the GPU screen",
            ),
            IntParam(
                key="threads", label="CPU threads", default=0, minimum=0,
                help="0 = omit --threads; llama.cpp then uses the physical core count, which is "
                     "optimal for CPU math (hyperthreads only add contention)",
            ),
            ChoiceParam(
                key="kv_cache_type", label="KV cache type", default="q8_0",
                choices=("", "f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl"),
                labels=(("", "llama.cpp default (f16)"),),
                help="K and V cache dtype.  q8_0 halves KV VRAM at negligible quality cost — often "
                     "what lets several long-context slots fit at all",
            ),
            IntParam(
                key="ubatch", label="Micro-batch size", default=None, minimum=1,
                none_label="llama.cpp default (512)",
                help="--ubatch-size.  Compute buffers scale with ubatch × total context; at very long "
                     "contexts (YaRN 1M × several slots) 256/128 is what lets it fit, at some "
                     "prompt-processing speed",
            ),
            IntParam(
                key="cache_ram_mib", label="Host prompt cache (MiB)", default=CONFIG.host_cache_ram_mib,
                minimum=0, none_label="llama.cpp default",
                help="--cache-ram ceiling (default from [memory] host_cache_ram in config.toml)",
            ),
            AutoBoolParam(
                key="mtp", label="MTP speculative decoding", resolve=_sidecar_mtp,
                applies_to=_has_mtp_head,
                help="auto uses a sidecar mtp-*.gguf head; on also uses the model's embedded head "
                     "(opt-in: llama.cpp may not implement MTP for the architecture); off disables",
            ),
            _port("llamacpp", 5000),
            _EXTRA_FLAGS,
        ),
        # llama.cpp is a native build: each environment is a build's bin/ directory.
        env_kind="native",
        # "version: 0.6.0-dev (build 11461, commit 4d756bc72)" (current) or
        # "version: 10109 (0cea36222)" (older builds) → "llama.cpp b11461 (4d756bc72)"
        version_probe=VersionProbe(
            executable="llama-server",
            args=("--version",),
            patterns=(
                r"\(build (?P<build>\d+), commit (?P<commit>\w+)\)",
                r"version: (?P<build>\d+) \((?P<commit>\w+)\)",
            ),
            template="llama.cpp b{build} ({commit})",
        ),
        concurrency_param="parallel",
        rope_scaling_formats=frozenset({ModelFormat.GGUF}),
        chat_template_flag="--chat-template-file",
    ),

    ServerSoftware(
        id="tabby",
        label="tabbyAPI / ExLlamaV2",
        supported_formats=frozenset({
            ModelFormat.SAFETENSORS_BF16,
            ModelFormat.SAFETENSORS_FP16,
            ModelFormat.PYTORCH_BIN,
            ModelFormat.GPTQ,
            ModelFormat.EXL2,
        }),
        supported_ai_types=frozenset({AIType.TEXT_GEN}),
        params=(
            TextParam(key="model_dir", label="Model staging directory",
                      default=str(CONFIG.tabby_model_dir), hidden=True),
            ChoiceParam(
                key="cache_mode", label="KV cache mode", default="Q8", choices=("FP16", "Q8", "Q6", "Q4"),
                help="Quantized KV cache (Q8 is fast on Ampere)",
            ),
            BoolParam(key="disable_auth", label="Disable auth", default=True,
                      help="Serve the API without an API key"),
            _port("tabby", 5000),
            _EXTRA_FLAGS,
        ),
        # tabbyAPI requires a models directory; launch.py creates a symlink named
        # model.id under model_dir pointing to the model path.
        uses_staging_dir=True,
        version_packages=("exllamav2", "exllamav3"),
        # flash-attn was built against cuda-12.5; CUDA_HOME ensures nvcc resolution.
        cuda_home=CONFIG.cuda_home_tabby,
        # Absolute paths work: tabby resolves the name against its templates/ dir.
        chat_template_flag="--prompt-template",
    ),

    ServerSoftware(
        id="ooba",
        label="text-generation-webui",
        supported_formats=frozenset({
            ModelFormat.GGUF,
            ModelFormat.SAFETENSORS_BF16,
            ModelFormat.SAFETENSORS_FP16,
            ModelFormat.AWQ,
            ModelFormat.GPTQ,
            ModelFormat.INT8_BNB,
            ModelFormat.EXL2,
            ModelFormat.PYTORCH_BIN,
        }),
        supported_ai_types=frozenset({AIType.TEXT_GEN}),
        params=(
            TextParam(key="model_dir", label="Model staging directory",
                      default=str(CONFIG.ooba_model_dir), hidden=True),
            ChoiceParam(
                key="loader", label="Loader", default="",
                choices=("", "Transformers", "llama.cpp", "ExLlamav3_HF", "ExLlamav3"),
                labels=(("", "auto"),),
                help="auto = ooba picks the loader from the file type",
            ),
            IntParam(key="gpu_layers", label="GPU layers (llama.cpp)", default=-1, minimum=-1,
                     help="-1 offloads all layers"),
            TextParam(key="tensor_split", label="Tensor split (llama.cpp)", default="", empty_label="none",
                      help="e.g. 1,1,1,1 for an equal 4-GPU split; blank = no flag"),
            TextParam(key="cache_type", label="KV cache type", default="", empty_label="default",
                      help="e.g. q8_0 (llama.cpp loader) or q8 (ExLlama)"),
            AutoBoolParam(
                key="mtp", label="MTP speculative decoding", resolve=_sidecar_mtp,
                applies_to=_gguf_mtp_head,
                help="llama.cpp loader: auto uses a sidecar mtp-*.gguf head; on also uses the "
                     "model's embedded head; off disables",
            ),
            _port("ooba", 5000, key="api_port", label="API port"),
            _port("ooba_webui", 7860, key="webui_port", label="WebUI port"),
            BoolParam(key="nowebui", label="API only (no WebUI)", default=False),
            _EXTRA_FLAGS,
        ),
        quantizable_formats=frozenset({
            ModelFormat.SAFETENSORS_BF16,
            ModelFormat.SAFETENSORS_FP16,
            ModelFormat.PYTORCH_BIN,
        }),
        runtime_quants=(
            RuntimeQuantOption(RuntimeQuant.BNB_INT8, ("--load-in-8bit",)),
            RuntimeQuantOption(RuntimeQuant.BNB_NF4, ("--load-in-4bit",)),
        ),
        # ooba's server.py uses relative paths for Gradio's allowed_paths, which
        # Gradio resolves against CWD — must run from the TGW installation directory.
        launch_cwd=str(CONFIG.ooba_main.parent),
        # ooba requires a models directory; GGUF links get a .gguf suffix so its
        # format scanner can identify them automatically.
        uses_staging_dir=True,
        staging_format_suffixes={ModelFormat.GGUF: ".gguf"},
        # Only its llama.cpp loader can apply RoPE scaling (forwarded via --extra-flags).
        rope_scaling_formats=frozenset({ModelFormat.GGUF}),
        chat_template_flag="--chat-template-file",
    ),

    ServerSoftware(
        id="comfyui",
        label="ComfyUI",
        supported_formats=frozenset({
            ModelFormat.SAFETENSORS_BF16,
            ModelFormat.SAFETENSORS_FP16,
            ModelFormat.SAFETENSORS_FP8,
        }),
        supported_ai_types=frozenset({AIType.IMAGE_GEN}),
        params=(
            # Base under which per-model subdirs are created: each model is staged into
            # model_dir/<model.comfyui_subdir>/<source_filename> (staging_use_source_filename
            # keeps the original filename so workflows can reference models by their names).
            TextParam(key="model_dir", label="Model staging directory",
                      default=str(CONFIG.comfyui_models_dir), hidden=True),
            _port("comfyui", 8188),
            _EXTRA_FLAGS,
        ),
        # ComfyUI doesn't accept a model path at launch; all catalog image_gen models
        # are staged as symlinks so ComfyUI discovers them at startup.
        uses_staging_dir=True,
        stage_all_models=True,
        # Symlinks use the original source filename so ComfyUI workflows can reference
        # models by their natural names (e.g. flux1-dev.safetensors, ae.safetensors).
        staging_use_source_filename=True,
        # Read model.comfyui_subdir to route each model into the correct subdirectory
        # (checkpoints/ for SDXL, diffusion_models/ for FLUX UNets, etc.).
        staging_subdir_attr="comfyui_subdir",
        # ComfyUI-MultiGPU (DisTorch) distributes model layers across multiple GPUs,
        # allowing large models (e.g. FLUX) to use more VRAM than a single GPU provides.
        # Compute (latent space) still runs on the primary GPU; other GPUs act as VRAM
        # donors.  All selected GPUs are exposed via CUDA_VISIBLE_DEVICES.
        max_gpus=0,
    ),

    ServerSoftware(
        id="sglang",
        label="SGLang",
        supported_formats=_HF_CONFIG_FORMATS | {ModelFormat.GGUF},
        supported_ai_types=frozenset({AIType.TEXT_GEN}),
        params=(
            IntParam(key="tensor_parallel", label="Tensor parallel", default=1, managed_by=_GPU_SCREEN),
            IntParam(key="pipeline_parallel", label="Pipeline parallel",
                     default=CONFIG.pipeline_parallel, managed_by=_GPU_SCREEN),
            FloatParam(key="mem_fraction", label="Static memory fraction", default=0.9,
                       minimum=0.1, maximum=1.0, help="--mem-fraction-static"),
            _port("sglang", 5000),
            _EXTRA_FLAGS,
        ),
        quantizable_formats=frozenset({
            ModelFormat.SAFETENSORS_BF16,
            ModelFormat.SAFETENSORS_FP16,
            ModelFormat.PYTORCH_BIN,
        }),
        runtime_quants=(
            # SGLang materializes the full-precision weights before quantizing them:
            # it frees memory for KV cache afterwards but cannot make a model fit.
            RuntimeQuantOption(
                RuntimeQuant.FP8, ("--quantization", "fp8"),
                requires=HardwareRequirement(89, "SGLang's online FP8 uses Triton FP8 kernels (fails on sm_86)"),
                note="loads full precision first — needs the unquantized footprint to load",
                full_precision_load=True,
            ),
            _BNB_NF4_INFLIGHT,
        ),
        version_packages=("sglang",),
        # flashinfer JIT-compiles CUDA kernels; CUDA_HOME points to the nvcc
        # compatible with gcc-13 (cuda-12.5 vs the bundled cuda-12.1 headers).
        cuda_home=CONFIG.cuda_home_sglang,
        # Same parallelism constraint as vLLM: pipeline_parallel must match GPU count.
        gpu_count_param="pipeline_parallel",
        # RoPE scaling via --json-model-override-args (HF-config checkpoints only).
        rope_scaling_formats=_HF_CONFIG_FORMATS,
        rope_scaling_env={"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1"},
        chat_template_flag="--chat-template",
    ),

]


# ---------------------------------------------------------------------------
# Quantization tool registry
# ---------------------------------------------------------------------------
# These tools perform offline model conversion / requantization.
# input_formats gates which source formats each tool accepts.
# quant_options lists what each can produce (QuantOption), in picker order.


def _options(labels: dict[str, str]) -> tuple[QuantOption, ...]:
    """QuantOptions without hardware requirements, from {id: label}."""
    return tuple(QuantOption(option_id, label) for option_id, label in labels.items())


def _is_fp8_source(model: AIModel) -> bool:
    return model.format == ModelFormat.SAFETENSORS_FP8


def _llmc_accepts(model: AIModel) -> bool:
    """16-bit checkpoints, and standard block-FP8 ones (dequantized while converting)."""
    return not _is_fp8_source(model) or fp8_block_size(model.path) is not None


def _llmc_tool(
    id: str, label: str, output_format: ModelFormat, options: tuple[QuantOption, ...],
) -> QuantizationTool:
    """An llm-compressor tool (launcher/scripts/llmc-quantize).

    Quantization options are compressed-tensors preset scheme names; only data-free
    schemes (weight-only, or dynamic activations) apply, since nothing is calibrated.
    It processes safetensors shard by shard on the selected GPUs without
    instantiating the model, so any architecture and size works; block-FP8 sources
    are dequantized on the fly.  llm-compressor pins its own transformers /
    compressed-tensors versions, so it runs from a dedicated environment group
    rather than a server's environment.
    """
    return QuantizationTool(
        id=id,
        label=label,
        input_formats=frozenset({
            ModelFormat.SAFETENSORS_BF16, ModelFormat.SAFETENSORS_FP16, ModelFormat.SAFETENSORS_FP8,
        }),
        accepts=_llmc_accepts,
        output_format=output_format,
        quant_options=options,
        env_group="llm-compressor",
        avail_module="llmcompressor",
        pip_package="llmcompressor",
        install_note=(
            "llm-compressor needs its own environment (it pins transformers and "
            "compressed-tensors versions that conflict with vLLM): create a conda env "
            "(e.g. `conda create -n llm-compressor python=3.11`), register it under "
            "\"llm-compressor\" in user_data/environments.json, then pick this tool "
            "again to install the llmcompressor package."
        ),
    )


# Hardware needed to *serve* llm-compressor outputs (vLLM 0.26 kernel checks).
_SM89_FP8 = HardwareRequirement(89, "FP8 activations need FP8 tensor cores")
_SM90_W4AFP8 = HardwareRequirement(90, "vLLM's W4A8-FP8 kernel targets Hopper (sm_90); no fallback below")
_SM100_MX = HardwareRequirement(100, "native MX kernels; vLLM's Marlin fallback failed on sm_86")

QUANTIZATION_TOOLS: list[QuantizationTool] = [

    QuantizationTool(
        id="llama-quantize",
        label="llama.cpp (GGUF re-quantize)",
        input_formats=frozenset({ModelFormat.GGUF}),
        output_format=ModelFormat.GGUF,
        quant_options=_options({
            GGUFQuant.Q2_K.value:   "Q2_K   (~2.6 bpw)",
            GGUFQuant.Q3_K_S.value: "Q3_K_S (~3.4 bpw)",
            GGUFQuant.Q3_K_M.value: "Q3_K_M (~3.9 bpw)",
            GGUFQuant.Q4_K_S.value: "Q4_K_S (~4.4 bpw)",
            GGUFQuant.Q4_K_M.value: "Q4_K_M (~4.8 bpw) ← popular",
            GGUFQuant.Q5_K_S.value: "Q5_K_S (~5.5 bpw)",
            GGUFQuant.Q5_K_M.value: "Q5_K_M (~5.7 bpw)",
            GGUFQuant.Q6_K.value:   "Q6_K   (~6.6 bpw) ← near-lossless",
            GGUFQuant.Q8_0.value:   "Q8_0   (~8.5 bpw) ← reference",
        }),
        # llama-quantize is a binary co-located with llama-server (same llama.cpp build).
        # No separate installation is needed — it runs from a registered llamacpp
        # environment, so if llama.cpp is set up this tool is available.
        output_dir_template="{stem}-GGUF-{option}",
        env_group="llamacpp",
        avail_executable="llama-quantize",
        install_note=(
            "llama-quantize not found in the llama.cpp environment. "
            "Build llama.cpp with CUDA support and register its bin/ directory as a "
            "llama.cpp environment — llama-quantize is built alongside llama-server."
        ),
    ),

    QuantizationTool(
        id="llama-hf-to-gguf",
        label="llama.cpp (HF → GGUF)",
        # convert_hf_to_gguf.py dequantizes FP8 (block, per-tensor, Mistral), ModelOpt,
        # single-group compressed-tensors and GPTQ checkpoints itself; DeepSeek-V4's
        # FP4 routed experts are repacked losslessly to MXFP4 whatever the outtype.
        input_formats=frozenset({
            ModelFormat.SAFETENSORS_BF16,
            ModelFormat.SAFETENSORS_FP16,
            ModelFormat.PYTORCH_BIN,
            ModelFormat.SAFETENSORS_FP8,
            ModelFormat.COMPRESSED_TENSORS,
            ModelFormat.GPTQ,
        }),
        output_format=ModelFormat.GGUF,
        quant_options=(
            QuantOption("BF16", "BF16 — brain floating point", args=("--outtype", "bf16")),
            QuantOption("F16", "F16 — 16-bit floating point", args=("--outtype", "f16")),
            QuantOption(
                "FP8-Q8_0", "FP8 tensors as Q8_0, the rest BF16 — ≈ source size, near-lossless",
                args=("--outtype", "bf16", "--fp8-as-q8"), applies_to=_is_fp8_source,
            ),
            QuantOption("Q8_0", "Q8_0 — 8-bit (~8.5 bpw), every tensor", args=("--outtype", "q8_0")),
            QuantOption("F32", "F32 — 32-bit floating point (2× BF16 size)", args=("--outtype", "f32")),
        ),
        output_dir_template="{stem}-GGUF-{option}",
        env_group="llamacpp",
        avail_script="convert_hf_to_gguf.py",
        python_executable=str(CONFIG.llama_convert_python),
        avail_modules=("torch", "transformers", "numpy"),
        uses_cuda=False,
        supports_vision=False,
        install_note=(
            "Requires convert_hf_to_gguf.py in the registered llama.cpp source tree and "
            "a Python interpreter with PyTorch, Transformers, and NumPy. Configure "
            "llama_convert_python in config.toml."
        ),
    ),

    QuantizationTool(
        id="exl2-convert",
        label="ExLlamaV2 (EXL2 convert)",
        input_formats=frozenset({
            ModelFormat.SAFETENSORS_BF16,
            ModelFormat.SAFETENSORS_FP16,
            ModelFormat.PYTORCH_BIN,
        }),
        output_format=ModelFormat.EXL2,
        quant_options=_options({
            "2.5": "2.5 bpw — very aggressive",
            "3.0": "3.0 bpw — aggressive",
            "3.5": "3.5 bpw",
            "4.0": "4.0 bpw — good quality/size balance",
            "4.5": "4.5 bpw",
            "5.0": "5.0 bpw — high quality",
            "6.0": "6.0 bpw — near-lossless",
            "8.0": "8.0 bpw — reference quality",
        }),
        output_dir_template="{stem}-exl2-{option}bpw",
        quant_label_template="{option} bpw",
        cuda_home=CONFIG.cuda_home_tabby,
        # exllamav2 is a pip-installable Python package (uses JIT CUDA compilation at
        # runtime, so no CUDA toolkit is required at install time).  It is already
        # installed in the default tabby environment as a dependency of tabbyAPI — if
        # tabbyAPI is set up, this tool is available with no further action.
        env_group="tabby",
        avail_module="exllamav2",
        pip_package="exllamav2",
    ),

    # Every data-free compressed-tensors preset some server can load is offered,
    # whatever the local GPUs; requires records what serving the output needs.
    # Labels name the servers that load each (vLLM 0.26, SGLang 0.5.9).  All were
    # converted and served on sm_86 except where requires says otherwise (their
    # requirement comes from vLLM's kernel checks, and they failed there as predicted).
    # Integer weights are RTN (no calibration): 4-bit is usable, 3-bit and below is not
    # without GPTQ/AWQ-style calibration.
    _llmc_tool(
        id="llmc-fp8",
        label="llm-compressor (FP8, data-free)",
        output_format=ModelFormat.SAFETENSORS_FP8,
        options=(
            QuantOption("FP8_BLOCK", "FP8 block — 128×128 weight blocks (~8 bpw)  ·  vLLM"),
            QuantOption(
                "FP8_DYNAMIC", "FP8 per-channel + per-token activations (~8 bpw)  ·  vLLM, SGLang",
                HardwareRequirement(0, "below sm_89 vLLM runs it weight-only (Marlin); FP16 models "
                                       "then need dtype bfloat16 — FP16 compute gives garbage"),
            ),
            QuantOption("MXFP8", "MXFP8 — FP8 weights + activations, 32-wide blocks  ·  vLLM", _SM100_MX),
            QuantOption("MXFP8A16", "MXFP8 weight-only — 32-wide blocks  ·  vLLM", _SM100_MX),
        ),
    ),

    _llmc_tool(
        id="llmc-int",
        label="llm-compressor (INT 2–8 bit / FP4, data-free)",
        output_format=ModelFormat.COMPRESSED_TENSORS,
        options=(
            QuantOption("W8A16", "INT8 weight-only — group 128 (~8.1 bpw)  ·  vLLM, SGLang"),
            QuantOption("W8A8", "INT8 weights + per-token INT8 activations (~8 bpw)  ·  vLLM, SGLang"),
            QuantOption("W7A16", "INT7 weight-only (~7.1 bpw)  ·  vLLM"),
            QuantOption("W7A8", "INT7 weights + INT8 activations (~7.1 bpw)  ·  vLLM"),
            QuantOption("W6A16", "INT6 weight-only (~6.1 bpw)  ·  vLLM"),
            QuantOption("W6A8", "INT6 weights + INT8 activations (~6.1 bpw)  ·  vLLM"),
            QuantOption("W5A16", "INT5 weight-only (~5.1 bpw)  ·  vLLM"),
            QuantOption("W5A8", "INT5 weights + INT8 activations (~5.1 bpw)  ·  vLLM"),
            QuantOption("W4A16", "INT4 weight-only, symmetric (~4.25 bpw)  ·  vLLM, SGLang"),
            QuantOption("W4A16_ASYM", "INT4 weight-only, asymmetric (~4.5 bpw)  ·  vLLM"),
            QuantOption("W4A8", "INT4 weights + INT8 activations (~4.25 bpw)  ·  vLLM, SGLang"),
            QuantOption("W4A4", "INT4 weights + INT4 activations (~4.25 bpw)  ·  vLLM, SGLang"),
            QuantOption("W4AFP8", "INT4 weights + FP8 activations (~4.25 bpw)  ·  vLLM", _SM90_W4AFP8),
            QuantOption("NVFP4A16", "NVFP4 weight-only — group 16 (~4.5 bpw)  ·  vLLM"),
            QuantOption("MXFP4A16", "MXFP4 weight-only — group 32 (~4.25 bpw)  ·  vLLM", _SM100_MX),
            QuantOption("MXFP4", "MXFP4 weights + activations — group 32 (~4.25 bpw)  ·  vLLM", _SM100_MX),
            QuantOption("W3A16", "INT3 weight-only (~3.1 bpw; heavily degraded without calibration)  ·  vLLM"),
            QuantOption("W3A8", "INT3 weights + INT8 activations (~3.1 bpw; heavily degraded)  ·  vLLM"),
            QuantOption("W3A4", "INT3 weights + INT4 activations (~3.1 bpw; heavily degraded)  ·  vLLM"),
            QuantOption("W2A16", "INT2 weight-only (~2.1 bpw; unusable without calibration)  ·  vLLM"),
            QuantOption("W2A8", "INT2 weights + INT8 activations (~2.1 bpw; unusable)  ·  vLLM"),
            QuantOption("W2A4", "INT2 weights + INT4 activations (~2.1 bpw; unusable)  ·  vLLM"),
        ),
    ),

    # Plain dequantization — makes FP8-only releases usable by every 16-bit tool
    # (EXL2, older llama.cpp builds, …) and servers; output is twice the FP8 size.
    _llmc_tool(
        id="llmc-dequant",
        label="llm-compressor (dequantize FP8 → BF16)",
        output_format=ModelFormat.SAFETENSORS_BF16,
        options=(QuantOption("BF16", "BF16 — exact dequantization (2× the FP8 size)", applies_to=_is_fp8_source),),
    ),
]


# ---------------------------------------------------------------------------
# AI model registry  (personal — lives in user_data/models_local.json, gitignored)
# ---------------------------------------------------------------------------

_CATALOG_PATH = Path(__file__).parent / "user_data" / "models_local.json"

try:
    with open(_CATALOG_PATH, encoding="utf-8") as _f:
        _catalog = json.load(_f)
    AI_MODELS: list[AIModel] = [AIModel.from_dict(m) for m in _catalog["models"]]
except FileNotFoundError:
    import sys
    print(
        f"\nERROR: {_CATALOG_PATH} not found.\n"
        "Copy  launcher/models_local.json.example\n"
        "  →   launcher/user_data/models_local.json\n"
        "and add your models.\n",
        file=sys.stderr,
    )
    sys.exit(1)
