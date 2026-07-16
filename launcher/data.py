"""
data.py — Generic server software registry and cache mount list.

Everything here is driven by config.toml and is safe to commit.
Your personal model inventory lives in models_local.py (gitignored).
Copy models_local.py.example → models_local.py to get started.

Adding a new server:
  1. Create a ServerSoftware() entry in SERVERS below, including any launch-time
     fields (strip_pythonpath, cuda_home, uses_staging_dir, staging_format_suffixes,
     launch_cwd) that the server requires.
  2. Create a matching builder function in launch.py and register it in _BUILDERS.
"""
from __future__ import annotations

import json
from pathlib import Path

from launcher.models import (
    AIModel, AIType, CacheMount, GGUFQuant, ModelFormat,
    QuantizationTool, RuntimeQuant, ServerSoftware,
)
from launcher.config import CONFIG


# ---------------------------------------------------------------------------
# Server software registry
# ---------------------------------------------------------------------------
# default_params defines the baseline for every launch on that server.
# A model's server_params[server.id] is merged on top at launch time;
# only values that differ from these defaults need to be specified there.

SERVERS: list[ServerSoftware] = [

    ServerSoftware(
        id="vllm",
        label="vLLM",
        supported_formats=frozenset({
            ModelFormat.SAFETENSORS_BF16,
            ModelFormat.SAFETENSORS_FP16,
            ModelFormat.SAFETENSORS_FP8,
            ModelFormat.AWQ,
            ModelFormat.GPTQ,
            ModelFormat.INT8_BNB,
            ModelFormat.PYTORCH_BIN,
        }),
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
        strip_pythonpath=True,
        launch_cwd="/tmp",
        # vLLM 0.25+ uses cu13 split wheels: shared libs live inside the Python package
        # tree (nvidia/cu13/lib/) and are not automatically on LD_LIBRARY_PATH.
        # CUDA_HOME points nvcc / JIT compilers at the bundled CUDA 13 toolkit.
        # LIBRARY_PATH + the libcudart.so symlink let flashinfer's runtime JIT link
        # against the bundled cudart (the wheel ships only libcudart.so.13).
        cuda_home=CONFIG.vllm_cuda_home,
        ld_library_path_prepend=CONFIG.vllm_cuda_libs,
        library_path_prepend=CONFIG.vllm_cuda_libs,
        lib_symlinks={"libcudart.so": "libcudart.so.13"},
        # VLLM_SLEEP_WHEN_IDLE: avoid 100% CPU spin in the engine loop while idle.
        extra_env={"VLLM_SLEEP_WHEN_IDLE": "1"},
        # pipeline_parallel must equal the number of GPUs selected; GpuSelectScreen
        # will auto-inject param_overrides["pipeline_parallel"] = selected_gpu_count.
        gpu_count_param="pipeline_parallel",
    ),

    ServerSoftware(
        id="llamacpp",
        label="llama.cpp",
        supported_formats=frozenset({ModelFormat.GGUF}),
        supported_ai_types=frozenset({AIType.TEXT_GEN}),
        default_params={
            "n_gpu_layers": 99,
            "split_mode":   "layer",
            "parallel":     1,
            "threads":      16,
            "port":         CONFIG.server_ports.get("llamacpp", 5000),
            "extra_flags":  [],
        },
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
        # model.id under model_dir pointing to the resolved model path.
        uses_staging_dir=True,
        # flash-attn was built against cuda-12.5; CUDA_HOME ensures nvcc resolution.
        cuda_home=CONFIG.cuda_home_tabby,
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
            "gpu_layers":   99,     # offload all layers
            "tensor_split": "",     # "" = no flag; "1,1,...,1" for equal N-GPU split
            "cache_type":   "",     # "" = default; "q8_0" llama.cpp; "q8" ExLlama
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
        supported_formats=frozenset({
            ModelFormat.SAFETENSORS_BF16,
            ModelFormat.SAFETENSORS_FP16,
            ModelFormat.SAFETENSORS_FP8,
            ModelFormat.AWQ,
            ModelFormat.GPTQ,
            ModelFormat.INT8_BNB,
            ModelFormat.PYTORCH_BIN,
            ModelFormat.GGUF,
        }),
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
        # flashinfer JIT-compiles CUDA kernels; CUDA_HOME points to the nvcc
        # compatible with gcc-13 (cuda-12.5 vs the bundled cuda-12.1 headers).
        cuda_home=CONFIG.cuda_home_sglang,
        # Same parallelism constraint as vLLM: pipeline_parallel must match GPU count.
        gpu_count_param="pipeline_parallel",
    ),

]


# ---------------------------------------------------------------------------
# Cache mount registry  (populated from config.toml [[cache_mounts]])
# ---------------------------------------------------------------------------

CACHE_MOUNTS: list[CacheMount] = [
    CacheMount(label=m.label, root=m.root, priority=m.priority)
    for m in CONFIG.cache_mounts
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
        # No separate installation is needed — if the llamacpp server is configured, this
        # binary is already present.
        avail_binary=str(CONFIG.llama_quantize),
        install_note=(
            f"llama-quantize binary not found at {CONFIG.llama_quantize}. "
            "Build llama.cpp with CUDA support and set llama_server in config.toml — "
            "llama-quantize is built alongside llama-server."
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
        # installed in the tabby conda env as a dependency of tabbyAPI — if tabbyAPI is
        # set up, this tool is available with no further action.
        avail_module=(str(CONFIG.tabby_python), "exllamav2"),
        auto_install_cmd=(str(CONFIG.tabby_python), "-m", "pip", "install", "exllamav2"),
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
