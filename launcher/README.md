# Model Server Launcher

Model Server Launcher is a keyboard-driven Textual control panel for a local AI workstation or server. It lets you browse a catalog of models, pick a serving backend, choose which GPUs to use, apply supported runtime quantization, and launch the server with your site-specific defaults from one place. It is designed for machines that host multiple model formats and backends, where repeatedly assembling the right command by hand is slow and error-prone.

![Launcher screenshot](../.github/assets/launcher.png)

## Requirements

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- One or more of: vLLM, llama.cpp, tabbyAPI installed (only the servers you actually use)

## Setup

### 1. Create the shared conda environment

A minimal shared conda environment is defined in `environment.yml` at the repo root. It covers both the launcher and `gpu_monitor`:

```bash
conda env create -f ~/ai-server-tools/environment.yml
```

This installs Python 3.11 and `textual` — the only runtime dependency of the launcher and gpu_monitor.  The model servers (vLLM, tabbyAPI, etc.) each have their own separate conda environments configured in `user_data/config.toml`.

To update the environment after pulling changes (e.g. if `textual` is upgraded):

```bash
conda env update -f ~/ai-server-tools/environment.yml --prune
```

### 2. Configure site paths

```bash
cp launcher/config.toml.example launcher/user_data/config.toml
$EDITOR launcher/user_data/config.toml
```

### 3. Add your models

```bash
cp launcher/models_local.json.example launcher/user_data/models_local.json
$EDITOR launcher/user_data/models_local.json
```

### 4. Run

```bash
cd ~/ai-server-tools
~/miniconda3/envs/ai-tools/bin/python -m launcher
```

Or create a convenience script (e.g. `~/ai_launch`):

```bash
cat > ~/ai_launch << 'EOF'
#!/bin/bash
cd ~/ai-server-tools
exec ~/miniconda3/envs/ai-tools/bin/python -m launcher "$@"
EOF
chmod +x ~/ai_launch
```

## Configuration

All system-specific paths live in **`user_data/config.toml`** (gitignored). The repo only ships `config.toml.example` — copy it to `user_data/` and fill in the fields for your machine.

| Section | Key | Description |
|---|---|---|
| `[paths]` | `conda_base` | Miniconda / Anaconda base directory |
| `[paths]` | `vllm_env` / `tabby_env` / `ooba_env` / `sglang_env` | Conda environment names |
| `[paths]` | `llama_server` | Path to compiled `llama-server` binary |
| `[paths]` | `tabby_main` / `tabby_model_dir` | tabbyAPI entry point and model staging directory |
| `[paths]` | `ooba_main` / `ooba_model_dir` | text-generation-webui entry point and model staging directory |
| `[paths]` | `cuda_home_tabby` | `CUDA_HOME` injected for tabbyAPI (flash-attn); leave `""` to skip |
| `[paths]` | `cuda_home_sglang` | `CUDA_HOME` injected for SGLang (flashinfer JIT); leave `""` to skip |
| `[paths]` | `cuda_visible_devices` | Explicit GPU list; `""` = auto-derive from `pipeline_parallel` |
| `[gpu]` | `pipeline_parallel` | GPU count for vLLM pipeline parallelism; `0` = auto-detect via NVML at startup |
| `[[cache_mounts]]` | `label` / `root` / `priority` | Local storage tiers for caching models locally from remote sources |

`~` and environment variables are expanded in all path values.

## Adding models

Edit **`user_data/models_local.json`**. Each entry in the `"models"` array is a JSON object. See `models_local.json.example` for the full field schema and documentation.

### Ingest tool

The ingest tool (`python -m launcher.ingest`) scans directories for models and manages the catalog from the command line. All subcommands support `--dry-run` to preview changes without writing.

#### Subcommands

| Subcommand | Purpose |
|---|---|
| `ingest` | Scan a directory for uncatalogued models and add them |
| `refresh` | Re-detect auto-derived fields for existing catalog entries |
| `copy-configs` | Copy HF config/tokenizer files from original model dirs into GGUF bundle dirs |
| `copy-mmproj` | Copy mmproj vision encoder files from original model dirs into GGUF bundle dirs |

#### ingest

```bash
# Scan a directory — add the first uncatalogued model found
python -m launcher.ingest ingest /path/to/models [--dry-run]

# Scan and add ALL uncatalogued models at once
python -m launcher.ingest ingest /path/to/models --all [--dry-run]

# Add a single specific model by path
python -m launcher.ingest ingest --path /path/to/specific/model [--dry-run]

# Ingest image generation models (diffusion checkpoints, FLUX, etc.)
python -m launcher.ingest ingest /path/to/models --ai-type image_gen --all [--dry-run]
```

Without `--all`, scanning stops after the first new model (useful for testing detection). With `--all`, every uncatalogued model is added in one pass.

Auto-detected fields: `id`, `name`, `format`, `ctx`, `size_gb`, `params_m`, `arch`, MoE expert counts, `vision`, `trust_remote_code`, `mmproj_path` (for vision GGUFs), and FP8 `min_sm` constraints.

HF directories with both safetensors and pytorch_bin weights get a separate catalog entry per format. GGUF bundle directories yield one entry per quant variant.

#### refresh

Re-detects auto-derived fields (`ctx`, `size_gb`, `params_m`, `arch`, MoE, `vision`, `mmproj_path`) for existing entries without touching user-curated fields (`id`, `name`, `format`, `server_status`, `favorite`, etc.):

```bash
# Refresh all entries (may be slow over NAS)
python -m launcher.ingest refresh [--dry-run]

# Refresh a single entry by path
python -m launcher.ingest refresh --path /path/to/model [--dry-run]
```

#### copy-configs / copy-mmproj

Preparation steps for GGUF bundles — run before `ingest` or `refresh` so that metadata detection has access to `config.json` and mmproj files:

```bash
# Copy HF config/tokenizer files into GGUF dirs that lack them
python -m launcher.ingest copy-configs /path/to/models [--dry-run]

# Copy mmproj-*.gguf vision encoder files into GGUF dirs that lack them
python -m launcher.ingest copy-mmproj /path/to/models [--dry-run]
```

Both commands search for the matching original (non-GGUF) model directory across the scan root and all configured cache mount subdirs.

### Manual entry

Quick example:

```json
{
  "id":       "my-model-q4",
  "name":     "My Model 7B",
  "path": "/mnt/models/text-generation/my-model.Q4_K_M.gguf",
  "format":   "gguf",
  "ai_types": ["text_gen"],
  "ctx":      32768,
  "size_gb":  4.5,
  "quant":    "Q4_K_M"
}
```

Override server defaults per model with a `server_params` key:

```json
"server_params": {
  "vllm": {"gpu_mem_util": 0.80, "tensor_parallel": 4}
}
```

## Adding server types

1. Create a `ServerSoftware(...)` entry in `data.py → SERVERS`.
2. Write `_build_<id>(server, model, params) → list[str]` in `launch.py`.
3. Register it in `_BUILDERS`.
4. If the server needs a staging model directory, add a symlink-creation block in `launch()`.

## Navigation

| Key | Action |
|---|---|
| `↑` / `↓` | Move cursor |
| `→` / `Enter` | Select / confirm (model list: advance to server or cache screen; server list: advance to GPU screen) |
| `←` / `Esc` / `b` | Go back |
| `q` | Quit from any screen |
| `Space` | Toggle GPU selection (GPU screen) |
| `a` | Select all GPUs (GPU screen, multi-GPU servers only) |
| `c` | Cache focused model to local storage (model list) |
| `u` | Uncache focused model (model list) |
| `l` | Set or clear context-length cap for focused model (model list) |
| `f` | Toggle favorite pin for focused model (model list) |
| `z` | Quantize focused model — opens tool/quant/destination picker (model list) |
| `g` / `x` / `?` | Mark focused server as good / bad / unknown (server list) |

## File map

```
launcher/
  __main__.py          Entry point: python -m launcher
  app.py               Textual App class
  screens.py           All TUI screens
  models.py            Schema: AIModel, ServerSoftware, AIType, ModelFormat
  data.py              SERVERS list + CACHE_MOUNTS (config-driven, safe to commit)
  launch.py            Command builders and subprocess launcher
  gpu.py               NVML polling (pynvml) + GpuRow helpers (shared with gpu_monitor)
  config.py            Config loader — reads user_data/config.toml at import time
  config.toml.example  Documented template (copy to user_data/config.toml)
  user_data/           ← gitignored; machine-specific and personal data
    config.toml          machine paths, GPU counts
    models_local.json    model catalog
  model_launcher.tcss    Textual CSS layout
```

## Status tracking

Test results are stored per model/server pair in `models_local.json` (`server_status` field) and updated in-place when you press `g` / `x` / `?` on the server selection screen. The model list shows a composite symbol:

| Symbol | Meaning |
|---|---|
| `✓` | All applicable servers confirmed good |
| `✗` | All applicable servers confirmed bad |
| `✓?` | At least one good, none bad |
| `✓✗` | Mixed — at least one good and one bad |
| `✗?` | At least one bad, none good |
| `?` | All untested |

## Quantizing models

Press `z` on any model in the model list to start a quantization job.  Only models in quantizable formats (GGUF, BF16/FP16 safetensors, PyTorch .bin) enable the keybinding.

**Flow:**
1. If more than one tool supports the model's format, pick a tool. Otherwise this step is skipped.
2. Pick a target quant level (e.g. Q4_K_M for GGUF, or 4.0 bpw for EXL2).
3. Pick an output destination — one of the configured `[[cache_mounts]]`, with current free space shown.
4. Review the full command and output path, then confirm.

The TUI exits, the conversion runs in your terminal with live output, and the log is written to `logs/`.

### Supported tools

| Tool | Converts | Prerequisite | Auto-install? |
|---|---|---|---|
| `llama-quantize` | GGUF → GGUF (re-quantize to a different quant level) | llama.cpp built binary — co-located with `llama-server`. No extra install if the `llamacpp` server is configured. | No (requires building llama.cpp with CUDA) |
| `exl2-convert` | BF16 / FP16 / BIN safetensors → EXL2 | `exllamav2` Python package in the `tabby` conda env. Already present if tabbyAPI is set up. If missing, the launcher offers to `pip install exllamav2` automatically (uses JIT CUDA compilation — no CUDA toolkit needed at install time). | **Yes** — press Enter in the install prompt |

### Adding a quantization tool

1. Add a `QuantizationTool(...)` entry to `QUANTIZATION_TOOLS` in `data.py`.
2. Write `_build_<id>(tool, model, quant_option, output_path) → list[str]` in `launch.py`.
3. Register it in `_QUANT_BUILDERS`.

