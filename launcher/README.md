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
| `[paths]` | `llama_convert_python` | Python with PyTorch, Transformers, and NumPy for HF-to-GGUF conversion; defaults to the launcher interpreter |
| `[paths]` | `cuda_home_tabby` | `CUDA_HOME` injected for tabbyAPI (flash-attn); leave `""` to skip |
| `[paths]` | `cuda_home_sglang` | `CUDA_HOME` injected for SGLang (flashinfer JIT); leave `""` to skip |
| `[paths]` | `cuda_visible_devices` | Explicit GPU list; `""` = auto-derive from `pipeline_parallel` |
| `[gpu]` | `pipeline_parallel` | GPU count for vLLM pipeline parallelism; `0` = auto-detect via NVML at startup |
| `[memory]` | `host_cache_ram` | Ceiling for the host-RAM prompt cache (llama.cpp `--cache-ram`): a share of physical RAM (`"60%"`) or integer MiB; omit for the server default (8 GiB) |
| `[[locations]]` | `label` / `root` / `description` | Storage locations models live on (NAS and local drives alike), in display order. `label` is the short name shown in the model list's Location column; `description` (optional) is shown in destination pickers. Copy, convert and quantize can write to any location. |

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
| `sync-copies` | Copy files that another copy of a model has but this copy lacks |

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

Auto-detected fields: `id`, `name`, `format`, `ctx`, `size_gb`, `params_m`, `arch`, MoE expert counts, `vision`, `trust_remote_code`, `mmproj_path` (for vision GGUFs), `mtp_path` (MTP draft heads shipped as `mtp-*.gguf`), `mtp_embedded` (MTP layers inside the model itself — a GGUF's NextN layers, or an HF checkpoint whose `config.json` declares MTP layers and that ships their weights; used only after opting in — see [Model settings](#model-settings)), and FP8 `min_sm` constraints.

HF directories with both safetensors and pytorch_bin weights get a separate catalog entry per format. GGUF bundle directories yield one entry per quant variant.

#### refresh

Re-detects auto-derived fields (`ctx`, `size_gb`, `params_m`, `arch`, MoE, `vision`, `mmproj_path`, `mtp_path`, `mtp_embedded`) for existing entries without touching user-curated fields (`id`, `name`, `format`, `server_status`, `favorite`, etc.):

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

Both commands search for the matching original (non-GGUF) model directory across the scan root and the same directory under every other storage location.

#### sync-copies

Completes copies of a model that lack files another copy has — for example config/tokenizer files of copies made before those travelled with the model, without which `refresh` cannot read the model's architecture. It only adds missing files: nothing is overwritten or deleted, and files that differ between copies are left alone.

```bash
# Every model with copies (offline locations are skipped)
python -m launcher.ingest sync-copies [--dry-run]

# One copy, then re-detect the fields the new files describe
python -m launcher.ingest sync-copies --path /path/to/copy [--dry-run]
python -m launcher.ingest refresh --path /path/to/copy [--dry-run]
```

Ingesting a model found at the same relative path as a catalogued model on another location (e.g. a copy made outside the launcher) adds it as a copy of that model — same id, family and settings. Any other new model starts a new **family**, named after the model.

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

Override server defaults per model with a `server_params` key (or edit them on the [model settings](#model-settings) page):

```json
"server_params": {
  "vllm": {"gpu_mem_util": 0.80, "max_num_seqs": 16}
}
```

## Adding server types

1. Create a `ServerSoftware(...)` entry in `data.py → SERVERS`, declaring its launch parameters as typed `ParamSpec`s (`params.py`) — they appear on the model settings page automatically.
2. Write `_build_<id>(server, env, model, params) → list[str]` in `launch.py` (take the interpreter / binaries from `env`).
3. Register it in `_BUILDERS`.
4. If the server needs a staging model directory, add a symlink-creation block in `launch()`.
5. Register an environment for it (`e` on the server list).

## Model settings

Choosing a model opens its settings page before the server list (Enter continues straight on). The top shows what the model is — size, MoE, architecture, vision, MTP, and which storage locations hold copies. Below is every setting with its effective value and where it comes from:

- *default* — the server's default; *this model* — overridden for this model; *auto* — decided at launch (shown with the reason); *GPU screen* — set when you pick GPUs (tensor / pipeline parallel are previewed for all GPUs).
- **Model**: context / YaRN (e.g. YaRN factor `4` → 256K × 4 = 1M on llama.cpp, vLLM, SGLang and ooba GGUF, and/or a context-length override), [chat template patches](#chat-template-patches), tool-call parser, reasoning parser (vLLM / SGLang return the model's thinking separately as `reasoning_content`; e.g. `qwen3`), trust remote code, note.
- **One section per compatible server** with its launch parameters, each with a description in the help line. Model-specific ones appear only where they apply — expert parallel for MoE models, MTP for models with an MTP head.

`e` edits the focused setting (a picker or a validated text prompt), `Space` toggles on/off and cycles auto/on/off settings, `r` resets an override to the default. Changes are saved immediately to this catalog entry. Copies of a model at other locations are separate entries and keep their own settings (the page tells you which copies exist).

Notable settings:

- **Expert parallel (vLLM, MoE models)** — `auto` turns on `--enable-expert-parallel` exactly when the experts cannot be split across the tensor-parallel GPUs (e.g. Qwen3.8-Flash-Next FP8 at TP8: expert width 640 / 8 = 80 does not align to its 128-wide FP8 blocks). It has no effect when a model runs pipeline-parallel only (TP1), which is how most MoE models launch here. The GPU screen shows `expert parallel: on (auto)` when it applies.
- **MTP speculative decoding (vLLM)** — off by default; turn it on for checkpoints that ship multi-token-prediction layers (detected by ingest/refresh). Adds `--speculative-config '{"method":"mtp","num_speculative_tokens":N}'` with *MTP draft tokens* (default 2). It speeds up decoding — Qwen3.8-Flash-Next averaged ~2 tokens per step with 2 — but costs memory: its context auto-fit dropped from 262K to ~190K. Check `SpecDecoding metrics` in the log to tune the draft count.
- **MTP (llama.cpp / ooba GGUF)** — `auto` uses a sidecar `mtp-*.gguf` head; `on` also uses an embedded head (opt-in, as llama.cpp may not implement MTP for the architecture).
- Prefix caching needs no setting: vLLM enables it by default.

## Navigation

| Key | Action |
|---|---|
| `↑` / `↓` | Move cursor |
| `→` / `Enter` | Select / confirm (model list: open the model's settings, or the copy screen in image-gen model management; model settings: continue to the server list; server list: advance to environment or GPU screen) |
| `←` / `Esc` / `b` | Go back |
| `q` | Quit from any screen |
| `Space` | Toggle GPU selection (GPU screen) |
| `a` | Select all GPUs (GPU screen, multi-GPU servers only) |
| `+` / `-` | More / fewer concurrent request slots (GPU screen, llama.cpp) |
| `p` / `P` | Set the power limit for the highlighted GPU / all selected GPUs (GPU screen) |
| `c` | Copy focused model to another storage location (model list) — see [Storage locations and copies](#storage-locations-and-copies) |
| `Del` | Delete focused model's files and catalog entry (model list), after confirmation |
| `f` | Toggle favorite pin for focused model (model list) |
| `e` / `Space` / `r` | Edit / toggle / reset the focused setting (model settings) |
| `z` | Convert / quantize focused model — opens tool/output/destination picker (model list) |
| `g` / `x` / `?` | Mark the launch as good / bad / unknown — on the deepest screen for that server (server list, environment picker, or quantization picker) |
| `e` | Manage environments for the focused server (server list, environment picker) |
| `Ctrl+Enter` | Execute the edited command (final Ready to launch screen) |
| `Ctrl+Q` | Quit from the final command editor (`q` inserts text there) |

The final **Ready to launch** screen is an editor containing the exact shell command that will run. Launcher-controlled values such as `CUDA_VISIBLE_DEVICES`, non-default `CUDA_HOME` and library paths, `PYTHONPATH` removal, the launch working directory, and GPU power limits are explicit in that command. Edit any part of it to test alternate arguments or environment values; the edited text is written unchanged to the session log and executed through Bash.

## Concurrent slots and memory fit (llama.cpp)

The GPU screen shows how many requests llama.cpp will serve at once — `Slots N × 128K ctx` — and `+` / `-` change it. Every slot gets the model's full context (its `ctx`, or the override set on the model settings page): the launcher passes `--ctx-size` as context × slots, because llama.cpp divides it evenly between slots. The count you launch with is remembered for that model.

While you pick GPUs and slots, the launcher asks llama.cpp's own planner (`llama-fit-params`, from the selected build) whether the launch fits in the GPUs' currently free memory, and shows the answer a couple of seconds later:

- `✓ fits on GPU  needs 287 of 353 GiB free · weights 121 · KV 117 (23.4/slot) · compute 49 GiB`
- `⚠ only 49 layers fit on GPU — rest offloaded to CPU (slow)` — fewer slots, more GPUs, or a smaller quant will help.
- `✗ does not fit in free GPU memory — 21 GiB short` — shown within seconds; llama.cpp may then spend minutes searching for a CPU-offload split ("searching…"), which the estimate gives up on after 2 minutes, keeping the verdict.
- `⚠ fits in total GPU memory, but needs layers rebalanced across GPUs` — enough memory overall, but llama.cpp has to redistribute layers; llama-server does this itself at launch.

At very long contexts (e.g. YaRN 1M × several slots) compute buffers can outweigh the KV cache: they scale with the micro-batch size × total context. Setting the llama.cpp param `ubatch` to `256` (or `128`) for the model shrinks them substantially, at some cost in prompt-processing speed — Qwen3.8-Flash-Next Q8_0 fits 2 × 1M slots on 15 × 3090 with `ubatch: 256`.

Memory already used by other processes counts against the fit. Vision projectors (`--mmproj`), MTP draft heads, and `extra_flags` are not included in the estimate.

llama.cpp launches also default to a q8_0 KV cache (half the memory of f16, negligible quality loss; set `"kv_cache_type": "f16"` in a model's `server_params.llamacpp` to opt out) and to a host-RAM prompt cache sized by `[memory] host_cache_ram`, which lets returning conversations skip re-processing their prompt when there are more of them than slots.

## Chat template patches

Some chat templates reject requests the model itself handles fine. Qwen 3.5 / 3.6 / 3.8 templates (official and Unsloth GGUF) fail any request with a system message after the start of the conversation — `System message must be at the beginning.` — whereas Qwen2.5 / Qwen3 templates rendered such messages in place.

The launcher detects these templates and fixes them automatically at launch: it reads the model's template (GGUF header, `chat_template.jinja`, or `tokenizer_config.json`), applies the matching patches, writes the result to `user_data/chat_templates/<model id>.jinja`, and passes it to the server (`--chat-template-file` for llama.cpp and ooba, `--chat-template` for vLLM and SGLang, `--prompt-template` for tabbyAPI). Model files are never modified. The flag appears in the launch command, and the session log header names the applied patches (`# tmpl: …`).

Patches:

- `system_anywhere` — renders later system messages in place as ChatML system turns (`<|im_start|>system … <|im_end|>`) instead of failing. Requests with only a leading system message render exactly as before.

On the model settings page, `e` on *Chat template patches* shows which patches apply to the model and switches individual patches off (stored as `disabled_template_patches` in the catalog).

## GPU power limits

Each line on the GPU screen ends with the card's power limit in watts — `250↓ W` when it is below the card's factory default, `370→250↓ W` when the next launch will change it. Press `p` to set the limit for the highlighted GPU or `P` for every selected GPU (whole watts; leave blank for the factory default). The choice is remembered per physical card (by UUID, in `user_data/gpu_power.json`) and applies to every model.

Limits are applied as the first step of the launch command, so the session log reproduces them:

```
/home/you/ai-server-tools/launcher/scripts/gpu-power-limit 0,1,2=250 3=default && \
  env CUDA_VISIBLE_DEVICES=0,1,2,3 …
```

`gpu-power-limit` only calls `sudo nvidia-smi -pl` for GPUs whose current limit differs, so you are asked for your password only when something changes — typically the first launch after a reboot or GPU reset (NVIDIA power limits do not survive either). If sudo fails or is cancelled, the server does not start. The step is included whenever a launched GPU has a chosen limit or is currently not at its default, and launches with every card at its default omit it.

For prompt-free launches you can optionally allow exactly this `nvidia-smi` invocation without a password (`sudo visudo -f /etc/sudoers.d/gpu-power-limit`; requires sudo ≥ 1.9.10 for the regex; not yet verified on this machine):

```
youruser ALL=(root) NOPASSWD: /usr/bin/nvidia-smi ^-i [0-9\,]+ -pl [0-9]+$
```

Do not use a wildcard rule such as `nvidia-smi *` — `nvidia-smi -f <file>` would let it overwrite any file as root.

## File map

```
launcher/
  __main__.py          Entry point: python -m launcher
  app.py               Textual App class
  screens.py           All TUI screens
  models.py            Schema: AIModel, ServerSoftware, AIType, ModelFormat, editable model fields
  params.py            Typed server launch parameters (default, help, editor, auto resolution)
  hf_config.py         Model config.json facts for launch planning (parallelism, expert parallel)
  data.py              SERVERS list, quantization tools, catalog loading (safe to commit)
  storage.py           Storage locations; model copy and deletion file operations
  environments.py      Server environments (conda envs / native builds) + registry
  environments.json.example  Template for user_data/environments.json
  launch.py            Command builders and subprocess launcher
  chat_template.py     Chat template reading + launch-time patches (e.g. Qwen system messages)
  fit.py               Pre-launch memory-fit estimates (llama.cpp: llama-fit-params)
  gpu.py               NVML polling (pynvml) + GpuRow helpers (shared with gpu_monitor)
  gpu_power.py         Per-GPU power limits — remembered choices + launch-command step
  scripts/
    gpu-power-limit    Applies power limits (sudo nvidia-smi -pl) only where they differ
  config.py            Config loader — reads user_data/config.toml at import time
  config.toml.example  Documented template (copy to user_data/config.toml)
  user_data/           ← gitignored; machine-specific and personal data
    chat_templates/      patched chat templates (written at launch)
    config.toml          machine paths, GPU counts, host RAM budget
    environments.json    registered environments per server
    gpu_power.json       chosen GPU power limits (written by the GPU screen)
    models_local.json    model catalog
  model_launcher.tcss    Textual CSS layout
```

## Storage locations and copies

Every place models are stored — the NAS and each local drive — is a **location** (`[[locations]]` in `config.toml`). Each catalog entry is one model at one path; the model list's **Location** column (before Name) shows where it is, dimmed while that location is offline (e.g. the NAS is not mounted). There is no separate "cached copy": a model on two drives is two entries, which you pick between in the list.

- **Copy (`c`)** copies the model to another location at the same path relative to the location root, so every location mirrors the same layout. For a GGUF that is the model file (all shards of a split set), its companion files (mmproj, MTP head) and the `config.json` / tokenizer files describing it — not the rest of the bundle. A directory model is copied whole. The copy becomes a new catalog entry with the same id (served model name), family and a snapshot of every setting (test results, favorite, note, server params, context settings); afterwards the two entries are independent.
- **Delete (`Del`)** removes a model's files and its catalog entry. The confirmation lists every file or directory that will be deleted, with **Cancel** focused. Entries sharing the same directory (e.g. BF16 safetensors and `.bin` in one folder) are deleted together; companion files still used by another entry are kept; a GGUF bundle directory is removed once its last model is deleted. Deleting the **last copy** of a model additionally asks you to type its name. Any location can be written to and deleted from, including the NAS.

Models are grouped into **families**: the original model plus every copy, conversion and quantization made from it share a `family` key in the catalog, so they stay associated even after an intermediate (e.g. an F16 GGUF used to make a Q8_0) is deleted.

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

## Converting and quantizing models

Press `z` on any model in the model list to start a conversion or quantization job. Only GGUF, BF16/FP16 safetensors, and PyTorch `.bin` models have an applicable tool.

**Flow:**
1. If more than one tool supports the model's format, pick a tool. Otherwise this step is skipped.
2. Pick a target quant level (e.g. Q4_K_M for GGUF re-quantization, F16/BF16 for HF-to-GGUF conversion, or 4.0 bpw for EXL2).
3. Pick an output destination — one of the configured `[[locations]]`, with current free space shown (offline locations are disabled).
4. For CUDA quantizers, choose the GPUs to expose to the tool. The list shows current VRAM usage; all available GPUs start selected. Use Space to toggle a GPU, `a` to select all, and Enter / `→` to continue. At least one GPU must be selected; if GPU polling is unavailable, return and retry after restoring it. CPU-only converters skip this step.
5. Review the full command and output path, then confirm. For CUDA tools, the command includes `CUDA_VISIBLE_DEVICES`; Back returns to the GPU picker with your selection intact.

The TUI temporarily exits while the conversion runs in your terminal with live output, and the log is written to `logs/`. Outputs are placed under the AI-type subdirectory of the chosen location (for example, `~/fast_models/text-generation/`). After a successful conversion, the launcher copies the source's config/tokenizer files beside the output, removes EXL2 resumable work files, adds the generated model to `user_data/models_local.json`, and reopens with that model focused. The new entry joins the source's family and inherits its model facts and behaviour settings — context, parameter count, architecture, MoE layout, vision companion (`mmproj`) and MTP head, tool-call parser, YaRN scaling and disabled chat-template patches — but not its note, favorite pin, test results, server parameters or context override, which depend on the format and size. Failed or interrupted outputs retain their work files and are not added to the catalog.

### Supported tools

| Tool | Converts | Prerequisite | Auto-install? |
|---|---|---|---|
| `llama-quantize` | GGUF → GGUF (re-quantize to a different quant level) | llama.cpp built binary — co-located with `llama-server` in the default llama.cpp environment. No extra install if llama.cpp is set up. | No (requires building llama.cpp with CUDA) |
| `llama-hf-to-gguf` | BF16 / FP16 safetensors or PyTorch `.bin` → F16 / BF16 GGUF | `convert_hf_to_gguf.py` in the registered llama.cpp source tree and `llama_convert_python` with PyTorch, Transformers, and NumPy. Conversion runs on CPU; it does not need GPUs. Only architectures supported by that llama.cpp converter work; vision models are excluded because their separate projector is not converted by this workflow. | No |
| `exl2-convert` | BF16 / FP16 / BIN safetensors → EXL2 | `exllamav2` Python package in the default tabbyAPI environment. Already present if tabbyAPI is set up. If missing, the launcher offers to `pip install exllamav2` automatically (uses JIT CUDA compilation — no CUDA toolkit needed at install time). | **Yes** — press Enter in the install prompt |

### Adding a quantization tool

1. Add a `QuantizationTool(...)` entry to `QUANTIZATION_TOOLS` in `data.py`.
2. Write `_build_<id>(tool, env, model, quant_option, output_path) → list[str]` in `launch.py` (`env` is the default environment of the tool's `env_server_id`).
3. Register it in `_QUANT_BUILDERS`.
