# Copilot Instructions

Design docs for **GPU Monitor** and **Launcher**: `DESIGN.md` in ./gpu_monitor/ and ./launcher/. Read related document before making structural changes to a project. **Update `DESIGN.md` after making structural changes** — keep all sections current.

A README is also present in each project directory (`gpu_monitor/README.md`, `launcher/README.md`) — it contains user-facing instructions and should be kept up to date with any changes that affect usage.

## Development mode
This project is in development mode. There are not yet any other project deployments to be concerned about. When making changes to settings or to file formats, do not retain support for obsoleted settings or formats. Update local settings and data files to ensure the local test environment remains in sync with the current codebase.

## Guiding principles
**Domain-driven design** — Source code is organized into domains. Logic and content specific to a domain should be isolated to the source files related to that domain. For example, GPU polling and formatting logic lives in `gpu.py`, and the Textual UI lives in `app.py`. This separation of concerns makes it easier to find and modify code related to a specific aspect of the project.

**Single source of truth** — For any given piece of information, there should be a single authoritative source in the codebase. This reduces the risk of inconsistencies and makes it easier to update logic or content when requirements change.

**Design for extensibility** — The project evolves rapidly; write code that anticipates future use cases without over-engineering present ones. Abstract shared logic into helpers or modules when it appears in more than one place, or when a planned extension clearly warrants it — but not for one-off operations. Use flexible data structures, avoid tight coupling between modules, and keep specific data clearly separated from logic so that requirements can evolve without broad refactoring.

**Don't hack around** — Avoid quick-and-dirty solutions or hacks, even in development mode. Instead, take the time to implement features and fixes in a clean and robust way that adheres to the project's design principles. This will save time and effort in the long run by reducing technical debt and making the codebase easier to maintain and extend.

**Productization mindset** — Even though the project is in development mode, code should be written with an eye towards future production deployment. This means avoiding hardcoded values, ensuring that settings are configurable via environment variables or settings files, and writing code that is robust and maintainable. Specific data should be clearly separated from logic, and the codebase should be organized in a way that would support scaling and adding features in the future.

**Prefer objects over conditionals** — When a set of values or behaviours varies by type or variant (e.g. model formats, server software, GPU metrics), prefer encoding that variation in a class or dataclass rather than scattering `if variant == "x"` checks across the codebase. Use `@dataclass` for plain per-variant data, lightweight classes with `@property` when derived values are needed, and a full class hierarchy with `abc.ABC` / `@abstractmethod` when behaviour genuinely diverges. This keeps variant-specific logic co-located and makes adding new variants a matter of adding a new class rather than auditing existing branches. This leads to clearer, more maintainable code and makes it easier to add new variants in the future without modifying existing logic.

## Best Practices

**Report issues** - If you encounter potential issues while working on the codebase, consider reporting them to the user. Don't work around a real issue to satisfy the request at hand.

**Contribute creatively** - If you identify a missing feature or an area for improvement while working on the codebase, consider making a suggestion to the user. Don't get distracted, but don't limit your contributions to the specific request if you see an opportunity to add value.

**Consult the user** - Consider consulting the user for clarification if you encounter ambiguities or difficulty discerning or obtaining information that the user might be able to clarify or easily obtain. It's better to clarify uncertainties than to make assumptions that might lead to incorrect implementations.

**Narrow down the problem** - If you're expending a lot of effort investigating an issue, consider implementing temporary logging or composing a stand-alone test script to rule out entire code branches or categories of potential causes.

**Ingesting models** - When using the ingest tool (`python -m launcher.ingest ingest`) or (`python -m launcher.ingest refresh`), prefer to first run it in `--dry-run` mode to review the changes it will make before applying them. This helps catch any unexpected modifications to the model catalog and ensures that only intended changes are made.

## Changes file
A `changes` file lives in the repo root (gitignored). After every coding session make sure it accurately reflects all uncommitted changes — suitable for use directly as a commit message. Keep it terse and actively remove entries for changes that were reverted or superseded. The user clears the file after committing. Run `git diff HEAD` at any time to see exactly what has changed since the last commit.

**Format:** Use a blank line between logically distinct groups of changes. Within a group, one line per file or sub-topic. A group may be preceded by a short header line (no prefix) if it aids readability. Omit fine implementation details — describe *what* changed and *why* when not obvious, not *how*. For large refactors, a few summary lines beats a line-per-file inventory.

Do not include changes to local settings files (e.g. `config.toml`) or to data files (e.g. `models_local.json`), as those files do not get committed.

## Agent Tool Notes

### Logs — sensitivity caution
Session logs in `logs/` capture full server stdout/stderr and may contain system paths, GPU topology, port bindings, and (if request logging is enabled) prompt/response content. **Read logs only when specifically required for an active investigation, and only the log(s) directly relevant to that investigation** — typically the most recent one. Do not scan or summarise logs speculatively.

### Terminal (`run_in_terminal`)
- **Large output redirection**: When `run_in_terminal` output exceeds ~10 KB, VS Code silently routes it to a temp file and returns a file path instead of the content. To reliably read command output, redirect to a known file and read it with `read_file`:
  ```
  cmd > /tmp/out.txt 2>&1
  # then: read_file("/tmp/out.txt")  — do NOT cat it in another terminal call
  ```

---

## Launcher Entry Points

Textual TUI — `python -m launcher` from `~/ai-server-tools/`:
```bash
cd ~/ai-server-tools
~/miniconda3/envs/ai-tools/bin/python -m launcher
```

GPU Monitor — `python -m gpu_monitor` from `~/ai-server-tools/` (uses the same `ai-tools` env):
```bash
cd ~/ai-server-tools
~/miniconda3/envs/ai-tools/bin/python -m gpu_monitor
```

---

## File Map

> **Convention**: all machine-specific and runtime-generated data lives in `user_data/` directories (gitignored at each level). Source code, example templates (`*.example`), and docs are tracked.

```
~/ai-server-tools/
    .gitignore                      Gitignores user_data/ and logs/
    user_data/                      Gitignored — personal local data
        model_audit.md              Anomaly log from catalog audit (format fixes, open items)
        cache/                      Misc caches (Gradio, etc.)
    launcher/                       Textual TUI launcher (Python)
        __main__.py                 Entry point: python -m launcher
        app.py                      Textual App class + run() entry
        config.py                   SiteConfig loader — reads user_data/config.toml, exposes CONFIG singleton
        config.toml.example         Tracked template — copy to user_data/config.toml and edit
        data.py                     SERVERS list + CACHE_MOUNTS (config-driven, safe to commit)
        gpu.py                      NVML poller (pynvml) + GpuRow data + formatting helpers
        ingest.py                   CLI tool — scans a directory for new models and appends entries to models_local.json
        launch.py                   Command builders, process spawner, cache manager
        models.py                   Schema — ModelFormat enum, AIModel/ServerSoftware dataclasses
        models_local.json.example   Tracked template — schema reference + 3 sample entries
        screens.py                  All Textual screens (AITypeScreen → ModelSelectScreen → …)
        model_launcher.tcss         Textual CSS layout
        DESIGN.md                   Architecture and design decisions
        user_data/                  Gitignored — all machine-specific and personal data
            config.toml             Machine-specific paths and GPU counts
            models_local.json       AI_MODELS catalog (model paths, quant details, test results)
    gpu_monitor/                    Standalone GPU stats Textual app
        __main__.py                 Entry point: python -m gpu_monitor
        app.py                      Full-screen live DataTable + total VRAM summary bar
        gpu_monitor.tcss            CSS layout
        DESIGN.md                   Architecture and design decisions
```

---

Models are organised by AI type in subdirectories, e.g. `~/fast_models/text-generation/`.

**Principle**: Never pass a model *directory* to any server. Always pass paths explicitly.
Server tools scan config files at startup — on a NAS or slow drive this causes severe latency.

---

## tabbyAPI / ExLlamaV2

- **Conda env**: `tabby` (`~/miniconda3/envs/tabby/`)
- **Entry point**: `~/tabbyAPI/main.py`
- **Key advantage**: no attention-head divisibility constraint — uses all 11 GPUs via `--gpu-split-auto`
- **Formats**: EXL2, safetensors BF16/FP16, and legacy `.bin` — **GGUF is NOT supported** (use llama.cpp for GGUF; tabbyAPI's sister project YALS handles GGUF)
- **flash-attn**: 2.8.3 — built from source in `tabby` env with `CUDA_HOME=/usr/local/cuda-12.5`; cached at `~/.cache/pip/wheels/`
- **KV cache**: `--cache-mode Q8` (quantized KV, fast on Ampere)
- **Multi-GPU**: ExLlamaV2 row/column tensor split, not restricted by head count
- **Model directory approach**: tabbyAPI requires models to be subdirectories inside a `--model-dir`.
  At launch time `launch.py` automatically creates a symlink `~/tabbyAPI/models/<model.id>` → `model.resolve_path()`
  and passes `~/tabbyAPI/models/` as `--model-dir`. No pre-created symlinks or stub dirs are needed.
  The same pattern applies to any future tool that requires a models directory — add a block in `launch.py`.

---

## text-generation-webui (ooba)

- **Conda env**: `textgen` (`~/miniconda3/envs/textgen/`)
- **Entry point**: `~/text-generation-webui/server.py`
- **Staging symlink**: `launch.py` creates `~/text-generation-webui/user_data/models/<model.id>` → resolved path before spawning. GGUF files get a `.gguf` suffix in the link name so ooba's scanner identifies the format automatically.
- **Formats**: safetensors BF16/FP16, legacy `.bin`, GGUF, INT8_BNB, EXL2
- **Ports**: API on `--api-port` (default from `default_params`), WebUI on `--listen-port`

---

## SGLang

- **Conda env**: `sglang` (`~/miniconda3/envs/sglang/`, Python 3.11)
- **Version**: SGLang 0.5.9 with PyTorch 2.9.1+cu128
- **Entry point**: `python -m sglang.launch_server`
- **Formats**: safetensors BF16/FP16/FP8, GGUF (`--load-format gguf`), INT8_BNB (`--load-format bitsandbytes --quantization bitsandbytes`)
- **Parallelism**: `--tensor-parallel-size` / `--pipeline-parallel-size`; defaults from `server.default_params`
- **dtype**: always launched with `--dtype bfloat16` (compute dtype; weights loaded as-is)
- **CUDA_HOME**: set to `CONFIG.cuda_home_sglang` (from `cuda_home_sglang` in `config.toml`) to direct flashinfer's JIT compiler to the correct nvcc. The sglang env bundles cuda-12.1 headers which only support gcc ≤ 12; pointing `CUDA_HOME` at cuda-12.5 avoids the gcc version error.

---

## vLLM Configuration

### Parallelism
- `--distributed-executor-backend ray`
- PCIe topology: pipeline parallelism preferred over tensor parallelism (activations only cross between adjacent GPU pairs)
- **Attention-head constraint**: `--tensor-parallel-size` must evenly divide the model's attention head count (`num_attention_heads` in `config.json`). Use pipeline parallelism or tabbyAPI/ExLlamaV2 when TP=11 is not viable.

### Critical patch applied to installed vLLM
**File**: `~/miniconda3/envs/vllm/lib/python3.11/site-packages/vllm/v1/executor/ray_executor.py`

Ray's compiled DAG defaults to `_max_inflight_executions=10`, but with `pp=11` the engine tries to
submit 11 concurrent steps and crashes immediately on the first request. Patch (around line 632):

```python
_max_inflight = max(
    self.parallel_config.pipeline_parallel_size + 1, 16
)
return forward_dag.experimental_compile(
    enable_asyncio=enable_asyncio,
    _overlap_gpu_communication=envs.VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM,
    _max_inflight_executions=_max_inflight,
)
```

### Launch flags
- `--enforce-eager` — prevents CUDA graph pre-allocation OOM on large prompts with high gpu_mem_util
- `--enable-chunked-prefill` — safe with pp>1
- `--trust-remote-code` — needed for models that define custom architecture code (check `auto_map` in `config.json`)
- `cd /tmp` before launching — prevents `~/vllm/` source tree from shadowing the installed package
- **Port**: all scripts default to **5000**

---

## Textual Launcher API (`launcher/`)

**Entry**: `cd ~/ai-server-tools && ~/miniconda3/envs/ai-tools/bin/python -m launcher`

**Model config**: edit `launcher/user_data/models_local.json` — one JSON object per model in the `"models"` array.
See `launcher/models_local.json.example` for the full field schema.
Schema types live in `launcher/models.py` (`ModelFormat` enum, `AIModel.from_dict()` deserialiser).
Server definitions (SERVERS list, CACHE_MOUNTS) live in `launcher/data.py` (safe to commit).
`path` field: absolute path to the model — any location (NAS, local SSD, etc.). Replaces the old `nas_path` field.
`cached_path` field: optional faster local copy; written by `save_cached_path()` in `launch.py` after a successful rsync, cleared by passing `""`. `resolve_path()` returns `cached_path` when set and present on disk, else `path`.
Status field: `"good"` (green), `"bad"` (red), `"unknown"` (white) — persisted per `(model, server, quant)` triple via `save_server_status()` in `launch.py` (JSON roundtrip, no regex). `server_status` in JSON is a nested dict: `{server_id: {quant_key: status}}` where `quant_key` is `"none"` (full/stored precision) or a `RuntimeQuant.value` (e.g. `"bnb_int8"`). Only one quant key per server may be `"good"` (exclusive — `save_server_status` clears others). `best_status_for(server_id)` returns the best status across all quant entries for a server. `_quant_key(quant)` maps `RuntimeQuant | None` → string key.
`ctx_override` field: optional launch override; `0` = use native `ctx`. Can be higher or lower than `ctx` (e.g. for RoPE-extended inference). Set interactively with `l` on the model list (`SetCtxOverrideScreen`). Persisted via `save_ctx_override()` — same JSON roundtrip pattern as `save_server_status()`.
`favorite` field: boolean; `True` pins the model to the top of the sorted list. Toggled with `f` in `ModelSelectScreen`. Persisted via `save_favorite()` — key omitted from JSON when `False`. The `★` column shows `★` for favorites.
`min_sm` field: integer minimum CUDA compute capability (e.g. `89` for sm_89+); `0` = no constraint. Checked against `system_min_sm()` from `gpu.py` (cached). When `model.hw_compat(system_sm)` is `False`, the warn column shows `⛔` and the note bar below the model table shows `model.hw_note`.
`hw_note` field: string explaining the hardware requirement — shown in the launcher note bar when incompatible. Omitted from JSON when empty.
`comfyui_subdir` field: `image_gen` models only — ComfyUI `models/` subdirectory to stage into (e.g. `"checkpoints"` for SDXL, `"diffusion_models"` for FLUX UNets, `"vae"`, `"loras"`). Default `"checkpoints"` — key omitted from JSON for checkpoints. Staging uses the original source filename (`staging_use_source_filename=True` on ComfyUI's `ServerSoftware`) so ComfyUI workflows can reference models by their natural names (e.g. `flux1-dev.safetensors`, `ae.safetensors`).
`mmproj_path` field: absolute path to the companion `mmproj-*.gguf` vision encoder file; omitted from JSON when empty. Auto-detected by `ingest`/`refresh` when found in the GGUF bundle root (flat layout) or bundle root one level above the quant subdir. Populated via `copy-mmproj` + `refresh` workflow. Passed as `--mmproj` to llama.cpp and ooba at launch time.
Model list sort is multi-column: `_sort_keys` list of `(col, reverse)` tuples. Default order: `fav → status → name`. Clicking a header promotes it to primary (ascending); clicking the primary toggles direction. `_STATUS_RANK` maps composite status symbols (✓, ✓?, ✓✗, ✗?, ✗, ?) to integers 0–5 for sort.

**Log files**: `logs/{timestamp}_{server}_{model}.log`. Header contains metadata + a paste-ready command block (no `#` prefix on the command lines). Exit footer: `# exit: code N` or `# exit: killed by SIGSEGV (11)`. Captured via PTY so ANSI colours are preserved. SIGWINCH is forwarded to the PTY so progress bars resize live.

**GPU widget**: `gpu.py` contains `GpuRow`, `poll_gpus()`, and formatting helpers shared by both apps.
The launcher does **not** embed GPU stats — run `gpu_monitor` separately while a model is loaded.

**Flow** (text-gen): AITypeScreen → ModelSelectScreen → ServerSelectScreen → [QuantSelectScreen] → GpuSelectScreen → ConfirmLaunch → exits TUI → `Popen` → `proc.wait()`.
`GpuSelectScreen`: shows all GPUs with VRAM utilization; multi-select (Space) for multi-GPU servers, radio-button for `max_gpus=1` servers. `→`/`Enter` to confirm. `a` selects all. Status bar shows live `CUDA_VISIBLE_DEVICES` string and auto-adjusted `gpu_count_param` (e.g. `pipeline_parallel → N` for vLLM/SGLang). If NVML unavailable, skips to ConfirmLaunch with default devices. Selected devices → `cuda_devices` arg to `launch()`; GPU count → `param_overrides[gpu_count_param]` → `build_cmd()` last-layer merge.
`QuantSelectScreen` (`g`/`x`/`?`): marks the focused quant option good/bad/unknown; shows status dots per option. `ServerSelectScreen` disables `g`/`x`/`?` for servers where `available_runtime_quants(model.format)` is non-empty — marking must be done in `QuantSelectScreen`.
Path validation before GpuSelectScreen; missing files show a toast notification.

**Flow** (image-gen / hub): AITypeScreen → ImageGenHubScreen → "Launch Server" → ServerSelectScreen(model=None) → GpuSelectScreen → ConfirmLaunch → exits TUI.
"Manage Models" → ModelManageScreen → `→`/`Enter` → CacheActionScreen → CacheProgressScreen (rsync with live ProgressBar) or uncache.
`uses_hub_flow(servers, ai_type)` returns `True` when every server for that AI type has `stage_all_models=True`.
`ModelManageScreen` overrides `_open_model()` (template method in `ModelSelectScreen`) to push `CacheActionScreen`. The template method pattern is necessary because Textual calls all MRO classes' message handlers regardless of override — only plain method calls respect the most-derived class.
`CacheProgressScreen`: rsync launched with `--info=progress2`; stdout read via `select.select()` in a `@work(thread=True)` worker; `(\d+)%` regex drives a `ProgressBar` widget + speed/ETA detail label; `proc.wait()` called after stdout EOF before reading `returncode`. Single-file models detected with `src_p.is_file()` (not extension-based).

**Catalog ingest**: `python -m launcher.ingest <subcommand>` — four subcommands:
- `ingest <scan_root> [--all] [--ai-type text_gen|image_gen] [--dry-run]` — scans a directory for models not in the catalog and appends entries. `--path <path>` targets a specific model. `--ai-type image_gen` for diffusion checkpoints/FLUX.
- `refresh [--path <path>] [--dry-run]` — re-detects auto-derived fields (ctx, size_gb, params_m, arch, moe, vision, mmproj_path) for existing entries without touching user-curated fields.
- `copy-configs <scan_root> [--dry-run]` — copies HF config/tokenizer files into GGUF bundle directories before ingesting.
- `copy-mmproj <scan_root> [--dry-run]` — copies mmproj vision encoder files into GGUF bundle directories before ingesting/refreshing.

**Catalog anomaly log**: `user_data/model_audit.md` — records format misclassifications, size corrections,
and open items found during catalog audits. Gitignored (personal/local).

**Dependencies**: `textual 8.1.1` installed in the dedicated `launcher` conda env (defined in `environment.yml` at the repo root).

