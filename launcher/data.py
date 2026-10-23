"""
data.py — Generic server software registry, quantization tools and the model catalog.

Everything here is driven by config.toml and is safe to commit.
Your personal model inventory lives in models_local.py (gitignored).
Copy models_local.py.example → models_local.py to get started.

Adding a new server:
  1. Create a ServerSoftware() entry in SERVERS below, including any launch-time
     fields (env_kind, version_packages, strip_pythonpath, cuda_home, uses_staging_dir,
     staging_format_suffixes, launch_cwd) that the server requires.
  2. Create a matching builder function in launch.py and register it in _BUILDERS.
  3. Register at least one environment for it in user_data/environments.json.
"""
from __future__ import annotations

import json
from pathlib import Path

from launcher.models import (
    AIModel, AIType, GGUFQuant, ModelFormat,
    QuantizationTool, RuntimeQuant, ServerSoftware, VersionProbe,
)
from launcher.config import CONFIG


# ---------------------------------------------------------------------------
# Server software registry
# ---------------------------------------------------------------------------
# default_params defines the baseline for every launch on that server.
# A model's server_params[server.id] is merged on top at launch time;
# only values that differ from these defaults need to be specified there.

# Checkpoints described by an HF config.json, which vLLM / SGLang can override at launch.
_HF_CONFIG_FORMATS: frozenset[ModelFormat] = frozenset({
    ModelFormat.SAFETENSORS_BF16,
    ModelFormat.SAFETENSORS_FP16,
    ModelFormat.SAFETENSORS_FP8,
    ModelFormat.AWQ,
    ModelFormat.GPTQ,
    ModelFormat.INT8_BNB,
    ModelFormat.PYTORCH_BIN,
})

SERVERS: list[ServerSoftware] = [

    ServerSoftware(
        id="vllm",
        label="vLLM",
        supported_formats=_HF_CONFIG_FORMATS,
        supported_ai_types=frozenset({AIType.TEXT_GEN}),
        default_params={
            "tensor_parallel":   1,
            "pipeline_parallel": CONFIG.pipeline_parallel,
            # "" = let vLLM choose its own executor (mp for single-node, which is
            # simpler and more robust than ray).  Set to "ray"/"mp" per-model to force.
            "executor_backend":  "",
            # "auto" respects the model's own torch_dtype from config.json.  Override
            # per-model (e.g. "bfloat16") only when a checkpoint's native dtype is
            # undesirable (e.g. a float32 checkpoint that would otherwise OOM).
            "dtype":             "auto",
            # 0.90 is vLLM's own default and leaves headroom; 0.97 left too little and
            # was a frequent cause of load-time / CUDA-graph OOM.
            "gpu_mem_util":      0.90,
            # 0 = omit the flag (use vLLM's default of 256).  Set per-model to cap
            # concurrency for very large / memory-tight models.
            "max_num_seqs":      0,
            # False = allow CUDA graphs (faster).  Set True per-model only if graph
            # capture causes OOM.
            "enforce_eager":     False,
            # False = omit the flag (vLLM enables chunked prefill itself when helpful).
            "chunked_prefill":   False,
            # True = --enable-expert-parallel: MoE experts are distributed whole across
            # the TP ranks instead of each expert being TP-sharded.  Set per-model for
            # MoE checkpoints whose expert widths do not split evenly into their FP8
            # quantization blocks at the chosen TP (e.g. Qwen3.8-Flash-Next at TP8).
            "expert_parallel":   False,
            # True = --max-model-len -1: vLLM starts from the model's native context and
            # lowers it only as far as needed to fit the KV cache in memory (logged as
            # "Auto-fit max_model_len").  The model's ctx / ctx_override is not passed.
            # Set per-model for memory-tight models where the native context may not fit.
            "auto_fit_ctx":      False,
            # False = pass --no-enable-log-requests (quieter logs).  True enables
            # per-request logging.
            "log_requests":      False,
            "port":              CONFIG.server_ports.get("vllm", 5000),
            "extra_flags":       [],
        },
        quantizable_formats=frozenset({
            ModelFormat.SAFETENSORS_BF16,
            ModelFormat.SAFETENSORS_FP16,
            ModelFormat.PYTORCH_BIN,
        }),
        supported_runtime_quants=(RuntimeQuant.BNB_INT8,),
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
        default_params={
            "n_gpu_layers": "auto",
            "split_mode":   "layer",
            # Concurrent request slots.  Each slot gets the model's full effective_ctx:
            # with fixed slots llama.cpp splits --ctx-size evenly between them, so the
            # builder emits --ctx-size effective_ctx × parallel.
            "parallel":     1,
            # 0 = omit --threads; llama.cpp then uses the physical core count, which
            # is optimal for CPU math (hyperthreads only add contention).
            "threads":      0,
            # KV cache dtype for both K and V ("" = llama.cpp default f16).  q8_0 halves
            # KV VRAM at negligible quality cost — often what lets multiple long-context
            # slots fit at all.  Override per model with "f16" if quality suffers.
            "kv_cache_type": "q8_0",
            # Physical micro-batch size (--ubatch-size); None = llama.cpp default (512).
            # Compute buffers scale with ubatch × total context, so at very long contexts
            # (YaRN 1M × several slots) a smaller value (256/128) is what lets it fit,
            # at some cost in prompt-processing speed.
            "ubatch":       None,
            # Host-RAM prompt cache ceiling in MiB (--cache-ram); None = llama.cpp default.
            "cache_ram_mib": CONFIG.host_cache_ram_mib,
            # MTP speculative decoding: None = auto (sidecar mtp-*.gguf heads only),
            # True = also the model's embedded head (opt-in per model), False = off.
            "mtp":          None,
            "port":         CONFIG.server_ports.get("llamacpp", 5000),
            "extra_flags":  [],
        },
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
        default_params={
            "model_dir":    str(CONFIG.tabby_model_dir),
            "cache_mode":   "Q8",
            "disable_auth": True,
            "port":         CONFIG.server_ports.get("tabby", 5000),
            "extra_flags":  [],
        },
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
        default_params={
            "model_dir":    str(CONFIG.ooba_model_dir),
            # loader: "" = ooba auto-detects from file type; override per model if needed.
            # Valid: Transformers, llama.cpp, ExLlamav3_HF, ExLlamav3
            "loader":       "",
            # llama.cpp loader options
            "gpu_layers":   -1,     # offload all layers
            "tensor_split": "",     # "" = no flag; "1,1,...,1" for equal N-GPU split
            "cache_type":   "",     # "" = default; "q8_0" llama.cpp; "q8" ExLlama
            # MTP speculative decoding: None = auto (sidecar mtp-*.gguf heads only),
            # True = also the model's embedded head (opt-in per model), False = off.
            "mtp":          None,
            "api_port":     CONFIG.server_ports.get("ooba", 5000),
            "webui_port":   CONFIG.server_ports.get("ooba_webui", 7860),
            "nowebui":      False,  # True = skip Gradio WebUI, API only
            "extra_flags":  [],
        },
        quantizable_formats=frozenset({
            ModelFormat.SAFETENSORS_BF16,
            ModelFormat.SAFETENSORS_FP16,
            ModelFormat.PYTORCH_BIN,
        }),
        supported_runtime_quants=(RuntimeQuant.BNB_INT8, RuntimeQuant.BNB_NF4),
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
        default_params={
            # model_dir is the base under which per-model subdirs are created.
            # Each model is staged into model_dir/<model.comfyui_subdir>/<source_filename>
            # (staging_use_source_filename=True preserves the original filename so that
            # ComfyUI workflows can reference models by their natural names).
            "model_dir":   str(CONFIG.comfyui_models_dir),
            "port":        CONFIG.server_ports.get("comfyui", 8188),
            "extra_flags": [],
        },
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
        default_params={
            "tensor_parallel":   1,
            "pipeline_parallel": CONFIG.pipeline_parallel,
            "mem_fraction":      0.9,
            "port":              CONFIG.server_ports.get("sglang", 5000),
            "extra_flags":       [],
        },
        quantizable_formats=frozenset({
            ModelFormat.SAFETENSORS_BF16,
            ModelFormat.SAFETENSORS_FP16,
            ModelFormat.PYTORCH_BIN,
        }),
        supported_runtime_quants=(RuntimeQuant.BNB_INT8,),
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
# quant_options lists the option identifiers shown in the TUI picker.

QUANTIZATION_TOOLS: list[QuantizationTool] = [

    QuantizationTool(
        id="llama-quantize",
        label="llama.cpp (GGUF re-quantize)",
        input_formats=frozenset({ModelFormat.GGUF}),
        output_format=ModelFormat.GGUF,
        quant_options=(
            GGUFQuant.Q2_K.value,
            GGUFQuant.Q3_K_S.value, GGUFQuant.Q3_K_M.value,
            GGUFQuant.Q4_K_S.value, GGUFQuant.Q4_K_M.value,
            GGUFQuant.Q5_K_S.value, GGUFQuant.Q5_K_M.value,
            GGUFQuant.Q6_K.value,
            GGUFQuant.Q8_0.value,
        ),
        quant_option_labels={
            GGUFQuant.Q2_K.value:   "Q2_K   (~2.6 bpw)",
            GGUFQuant.Q3_K_S.value: "Q3_K_S (~3.4 bpw)",
            GGUFQuant.Q3_K_M.value: "Q3_K_M (~3.9 bpw)",
            GGUFQuant.Q4_K_S.value: "Q4_K_S (~4.4 bpw)",
            GGUFQuant.Q4_K_M.value: "Q4_K_M (~4.8 bpw) ← popular",
            GGUFQuant.Q5_K_S.value: "Q5_K_S (~5.5 bpw)",
            GGUFQuant.Q5_K_M.value: "Q5_K_M (~5.7 bpw)",
            GGUFQuant.Q6_K.value:   "Q6_K   (~6.6 bpw) ← near-lossless",
            GGUFQuant.Q8_0.value:   "Q8_0   (~8.5 bpw) ← reference",
        },
        # llama-quantize is a binary co-located with llama-server (same llama.cpp build).
        # No separate installation is needed — it runs from the default llamacpp
        # environment, so if llama.cpp is set up this tool is available.
        env_server_id="llamacpp",
        avail_executable="llama-quantize",
        install_note=(
            "llama-quantize not found in the default llama.cpp environment. "
            "Build llama.cpp with CUDA support and register its bin/ directory as a "
            "llama.cpp environment — llama-quantize is built alongside llama-server."
        ),
    ),

    QuantizationTool(
        id="llama-hf-to-gguf",
        label="llama.cpp (HF → GGUF)",
        input_formats=frozenset({
            ModelFormat.SAFETENSORS_BF16,
            ModelFormat.SAFETENSORS_FP16,
            ModelFormat.PYTORCH_BIN,
        }),
        output_format=ModelFormat.GGUF,
        quant_options=("F16", "BF16"),
        quant_option_labels={
            "F16": "F16 — 16-bit floating point",
            "BF16": "BF16 — brain floating point",
        },
        env_server_id="llamacpp",
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
        quant_options=("2.5", "3.0", "3.5", "4.0", "4.5", "5.0", "6.0", "8.0"),
        quant_option_labels={
            "2.5": "2.5 bpw — very aggressive",
            "3.0": "3.0 bpw — aggressive",
            "3.5": "3.5 bpw",
            "4.0": "4.0 bpw — good quality/size balance",
            "4.5": "4.5 bpw",
            "5.0": "5.0 bpw — high quality",
            "6.0": "6.0 bpw — near-lossless",
            "8.0": "8.0 bpw — reference quality",
        },
        cuda_home=CONFIG.cuda_home_tabby,
        # exllamav2 is a pip-installable Python package (uses JIT CUDA compilation at
        # runtime, so no CUDA toolkit is required at install time).  It is already
        # installed in the default tabby environment as a dependency of tabbyAPI — if
        # tabbyAPI is set up, this tool is available with no further action.
        env_server_id="tabby",
        avail_module="exllamav2",
        pip_package="exllamav2",
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
