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

This installs Python 3.11 and `textual` — the only runtime dependency of the launcher and gpu_monitor.  The model servers (vLLM, tabbyAPI, etc.) each run from their own conda environments (or, for llama.cpp, a build directory), registered in `user_data/environments.json` — see [Server environments](#server-environments).

To update the environment after pulling changes (e.g. if `textual` is upgraded):

```bash
conda env update -f ~/ai-server-tools/environment.yml --prune
```

### 2. Configure site paths

```bash
cp launcher/config.toml.example launcher/user_data/config.toml
$EDITOR launcher/user_data/config.toml
```

### 3. Register server environments

```bash
cp launcher/environments.json.example launcher/user_data/environments.json
$EDITOR launcher/user_data/environments.json
```

Keep one entry per server you use (more can be added later from the TUI with `e`).

### 4. Add your models

```bash
cp launcher/models_local.json.example launcher/user_data/models_local.json
$EDITOR launcher/user_data/models_local.json
```

### 5. Run

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
| `[paths]` | `conda_base` | Miniconda / Anaconda base directory (conda environments resolve to `<conda_base>/envs/<name>`) |
| `[paths]` | `tabby_main` / `tabby_model_dir` | tabbyAPI entry point and model staging directory |
| `[paths]` | `ooba_main` / `ooba_model_dir` | text-generation-webui entry point and model staging directory |
| `[paths]` | `comfyui_main` / `comfyui_models_dir` | ComfyUI entry point and model staging directory |
| `[paths]` | `exl2_convert_script` | ExLlamaV2 `convert.py` (falls back to the module entry point when absent) |
| `[paths]` | `cuda_home_tabby` | `CUDA_HOME` injected for tabbyAPI (flash-attn); leave `""` to skip |
| `[paths]` | `cuda_home_sglang` | `CUDA_HOME` injected for SGLang (flashinfer JIT); leave `""` to skip |
| `[paths]` | `cuda_visible_devices` | Explicit GPU list; `""` = auto-derive from `pipeline_parallel` |
| `[gpu]` | `pipeline_parallel` | GPU count for vLLM pipeline parallelism; `0` = auto-detect via NVML at startup |
| `[memory]` | `host_cache_ram` | Ceiling for the host-RAM prompt cache (llama.cpp `--cache-ram`): a share of physical RAM (`"60%"`) or integer MiB; omit for the server default (8 GiB) |
| `[[cache_mounts]]` | `label` / `root` / `priority` | Local storage tiers for caching models locally from remote sources |

`~` and environment variables are expanded in all path values.

Which conda environment (or llama.cpp build) each server runs from is not set here — see [Server environments](#server-environments).

## Server environments

Each server launches from a registered **environment**: a conda env (vLLM, SGLang, tabbyAPI, text-generation-webui, ComfyUI) or a native build directory (llama.cpp). You can register several per server — for example keep a known-good `vllm` and try a newer release in `vllm-next` — and pick one at launch. The registry lives in `user_data/environments.json` (gitignored); the TUI maintains it.

When a server has more than one environment, choosing it opens an **environment picker** (the default ★ is pre-focused; installed versions such as `vllm 0.26.0` are shown). Servers with a single environment skip the picker.

Press `e` on the server list or the environment picker to **manage environments** for that server:

| Key | Action |
|---|---|
| `a` | Register an environment. If the conda env doesn't exist yet, you're offered to create it — empty (choose a Python version) or as a clone of an existing env — or register it anyway |
| `c` | Create the focused environment when it is missing on disk |
| `r` | Rename — renames the conda env itself (`conda rename`) and carries its test results over |
| `d` | Unregister — optionally also deletes the conda env (`conda env remove`); its test results are removed |
| `s` | Make the focused environment the default (pre-selected at launch; used by quantization tools) |
| `n` | Edit the note shown beside the environment |

Conda operations run in the TUI with live output. Upgrading packages inside an environment is done by hand (e.g. `conda activate vllm-next && pip install -U vllm`); the launcher picks up the new version automatically. For vLLM, a pip-bundled CUDA toolkit (`nvidia/cu13/`) is detected per environment and wired into `CUDA_HOME` / library paths at launch.

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
2. Write `_build_<id>(server, env, model, params) → list[str]` in `launch.py` (take the interpreter / binaries from `env`).
3. Register it in `_BUILDERS`.
4. If the server needs a staging model directory, add a symlink-creation block in `launch()`.
5. Register an environment for it (`e` on the server list).

## Navigation

| Key | Action |
|---|---|
| `↑` / `↓` | Move cursor |
| `→` / `Enter` | Select / confirm (model list: advance to server or cache screen; server list: advance to environment or GPU screen) |
| `←` / `Esc` / `b` | Go back |
| `q` | Quit from any screen |
| `Space` | Toggle GPU selection (GPU screen) |
| `a` | Select all GPUs (GPU screen, multi-GPU servers only) |
| `+` / `-` | More / fewer concurrent request slots (GPU screen, llama.cpp) |
| `c` | Cache focused model to local storage (model list) |
| `u` | Uncache focused model (model list) |
| `l` | Set or clear context-length cap for focused model (model list) |
| `f` | Toggle favorite pin for focused model (model list) |
| `z` | Quantize focused model — opens tool/quant/destination picker (model list) |
| `g` / `x` / `?` | Mark the launch as good / bad / unknown — on the deepest screen for that server (server list, environment picker, or quantization picker) |
| `e` | Manage environments for the focused server (server list, environment picker) |
| `Ctrl+Enter` | Execute the edited command (final Ready to launch screen) |
| `Ctrl+Q` | Quit from the final command editor (`q` inserts text there) |

The final **Ready to launch** screen is an editor containing the exact shell command that will run. Launcher-controlled values such as `CUDA_VISIBLE_DEVICES`, non-default `CUDA_HOME` and library paths, `PYTHONPATH` removal, and the launch working directory are explicit in that command. Edit any part of it to test alternate arguments or environment values; the edited text is written unchanged to the session log and executed through Bash.

## Concurrent slots and memory fit (llama.cpp)

The GPU screen shows how many requests llama.cpp will serve at once — `Slots N × 128K ctx` — and `+` / `-` change it. Every slot gets the model's full context (its `ctx`, or the `l` override): the launcher passes `--ctx-size` as context × slots, because llama.cpp divides it evenly between slots. The count you launch with is remembered for that model.

While you pick GPUs and slots, the launcher asks llama.cpp's own planner (`llama-fit-params`, from the selected build) whether the launch fits in the GPUs' currently free memory, and shows the answer a couple of seconds later:

- `✓ fits on GPU  needs 287 of 353 GiB free · weights 121 · KV 117 (23.4/slot) · compute 49 GiB`
- `⚠ only 49 layers fit on GPU — rest offloaded to CPU (slow)` — fewer slots, more GPUs, or a smaller quant will help.

Memory already used by other processes counts against the fit. Vision projectors (`--mmproj`) and `extra_flags` are not included in the estimate.

llama.cpp launches also default to a q8_0 KV cache (half the memory of f16, negligible quality loss; set `"kv_cache_type": "f16"` in a model's `server_params.llamacpp` to opt out) and to a host-RAM prompt cache sized by `[memory] host_cache_ram`, which lets returning conversations skip re-processing their prompt when there are more of them than slots.

## File map

```
launcher/
  __main__.py          Entry point: python -m launcher
  app.py               Textual App class
  screens.py           All TUI screens
  models.py            Schema: AIModel, ServerSoftware, AIType, ModelFormat
  data.py              SERVERS list + CACHE_MOUNTS (config-driven, safe to commit)
  environments.py      Server environments (conda envs / native builds) + registry
  environments.json.example  Template for user_data/environments.json
  launch.py            Command builders and subprocess launcher
  fit.py               Pre-launch memory-fit estimates (llama.cpp: llama-fit-params)
  gpu.py               NVML polling (pynvml) + GpuRow helpers (shared with gpu_monitor)
  config.py            Config loader — reads user_data/config.toml at import time
  config.toml.example  Documented template (copy to user_data/config.toml)
  user_data/           ← gitignored; machine-specific and personal data
    config.toml          machine paths, GPU counts, host RAM budget
    environments.json    registered environments per server
    models_local.json    model catalog
  model_launcher.tcss    Textual CSS layout
```

## Status tracking

Test results are stored per model, server **and environment** in `models_local.json` (`server_status` field) and updated in-place when you press `g` / `x` / `?`. Marking happens on the deepest screen of the launch: the server list when the server has a single environment and no runtime quantization options, otherwise the environment picker or the quantization picker. The server list shows the best result across a server's environments; the model list shows a composite symbol:

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

The TUI temporarily exits while the conversion runs in your terminal with live output, and the log is written to `logs/`. Outputs are placed under the AI-type subdirectory of the chosen cache mount (for example, `~/fast_models/text-generation/`). After a successful conversion, the launcher copies required EXL2 config/tokenizer metadata, removes resumable conversion work files, adds the generated model to `user_data/models_local.json`, and reopens with that model focused. Failed or interrupted outputs retain their work files and are not added to the catalog.

### Supported tools

| Tool | Converts | Prerequisite | Auto-install? |
|---|---|---|---|
| `llama-quantize` | GGUF → GGUF (re-quantize to a different quant level) | llama.cpp built binary — co-located with `llama-server` in the default llama.cpp environment. No extra install if llama.cpp is set up. | No (requires building llama.cpp with CUDA) |
| `exl2-convert` | BF16 / FP16 / BIN safetensors → EXL2 | `exllamav2` Python package in the default tabbyAPI environment. Already present if tabbyAPI is set up. If missing, the launcher offers to `pip install exllamav2` automatically (uses JIT CUDA compilation — no CUDA toolkit needed at install time). | **Yes** — press Enter in the install prompt |

### Adding a quantization tool

1. Add a `QuantizationTool(...)` entry to `QUANTIZATION_TOOLS` in `data.py`.
2. Write `_build_<id>(tool, env, model, quant_option, output_path) → list[str]` in `launch.py` (`env` is the default environment of the tool's `env_server_id`).
3. Register it in `_QUANT_BUILDERS`.

