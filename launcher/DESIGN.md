# Launcher — Design Document

## Purpose

A keyboard-driven Textual TUI that selects and launches large-model inference servers (vLLM, llama.cpp, tabbyAPI). The key objectives:

- Navigate a large model catalogue without touching config files each time
- Prefer the fastest cached copy of a model over the original source automatically
- Persist test-result annotations (good / bad / unknown) per model per server
- Keep all system-specific paths out of version control
- Quantize existing models to new formats / quant levels from within the TUI

---

## File Map and Separation of Concerns

```
config.py                   Site-specific loader — reads user_data/config.toml, exposes CONFIG singleton
config.toml.example         Tracked template — copy to user_data/config.toml and edit
models.py                   Schema — enums, dataclasses, pure logic (no instances, no I/O)
data.py                     Generic instances — SERVERS list + CACHE_MOUNTS (config-driven, committable)
models_local.json.example   Tracked template — schema reference + 3 sample entries
launch.py                   Command builders + process spawner + cache manager
screens.py                  All Textual screens
app.py                      Top-level App class, entry to screen stack
gpu.py                      NVML polling via pynvml (shared with gpu_monitor)
ingest.py                   CLI tool — scans a directory for new models and appends entries to models_local.json
user_data/                  Gitignored — all machine-specific and personal data
    config.toml               Machine-specific paths and GPU counts
    models_local.json         AI_MODELS catalog (model paths, quant details, test results)
```

**Rule**: nothing in `models.py` or higher imports from `config.py`. `config.py` may not import any sibling module. This prevents circular imports regardless of load order.

---

## Configuration Layering

```
user_data/config.toml (gitignored)
    └── config.py → SiteConfig dataclass (singleton CONFIG)
            ├── models.py  (cache mounts)
            ├── data.py    (SERVERS.default_params, CACHE_MOUNTS)
            └── launch.py  (binary paths, CUDA_VISIBLE_DEVICES, CUDA_HOME)
```

`pipeline_parallel = 0` in `config.toml` triggers NVML auto-detection at import time. The result is stored as `CONFIG.pipeline_parallel` (never zero afterwards). `resolved_cuda_devices` builds `"0,1,...,N-1"` if no explicit device string was given.

**Why TOML, not envvars or a Python file?** TOML is structured and typed without being executable. Envvars are awkward for multi-key sections (cache mounts) and invisible at a glance. A Python file would need `exec()` or an import that could shadow package modules.

---

## The Privacy Split: data.py vs models_local.py

`data.py` contains the SERVERS list and CACHE_MOUNTS table — both are fully config-driven (paths come from CONFIG) and are safe to commit.

`user_data/models_local.json` (gitignored) contains the AI_MODELS catalog. It is a plain JSON file: one object with a `"models"` array, each element corresponding to one `AIModel`. Being JSON rather than Python means:
- A future catalog scanner can generate or update it without touching Python source
- Status updates (`save_server_status`) are a trivial `json.load` / mutate / `json.dump` — no regex surgery
- Users can edit it without risk of breaking Python syntax

`ingest.py` can scan a directory and generate new entries automatically.

`models_local.json.example` is tracked and ships three template entries. The `_schema` object at the top documents every field.

`data.py` loads the JSON at import time via `AIModel.from_dict()` and exposes the result as `AI_MODELS`. All callers `from launcher.data import AI_MODELS`. If the file is missing, a clear error exits immediately.

---

## Data Model

### AIModel

Represents one deployable artefact (a specific file or directory). Key fields:

| Field | Purpose |
|---|---|
| `id` | Unique slug; becomes `--served-model-name` / llama.cpp `--alias` |
| `path` | Absolute path to the model file or directory — any accessible location (local SSD, network share, etc.) |
| `format` | Gates server compatibility (`ModelFormat` enum) |
| `ai_types` | frozenset of AIType — determines which type-selection menus it appears in |
| `ctx` | Native/trained maximum context length (tokens); used verbatim if no `ctx_override` is set |
| `ctx_override` | Optional launch override (tokens); `0` = use `ctx`. Can be higher or lower than `ctx`. Persisted in JSON. |
| `effective_ctx` | Property — returns `ctx_override` if non-zero, else `ctx`. This is what builders receive. |
| `favorite` | Boolean; `True` = pinned model floats to the top of the sorted list. Persisted in JSON (key omitted when `False`). |
| `min_sm` | Minimum CUDA compute capability required to load this model (integer, e.g. `89` for sm_89+). `0` = no constraint. Checked against `system_min_sm()` at display time. |
| `hw_note` | Optional human-readable explanation shown in the note bar when `hw_compat()` returns `False`. When omitted, `hw_compat_note` generates a default: `"Requires sm_{min_sm}+ GPU"`. |
| `comfyui_subdir` | `image_gen` models only — ComfyUI `models/` subdirectory to stage into. Default: `"checkpoints"` (omit from JSON). Use `"diffusion_models"` for FLUX UNets. Extensible: any valid subdirectory name accepted by ComfyUI. |
| `mmproj_path` | Absolute path to the companion `mmproj-*.gguf` vision encoder file. Set automatically by `ingest`/`refresh` when the file is present in the bundle root. Passed as `--mmproj` to llama.cpp and ooba at launch time. Omitted from JSON when empty. |
| `tool_call_parser` | vLLM tool-call parser name (e.g. `"openai"`, `"hermes"`, `"llama3_json"`). When set, `_build_vllm` emits `--enable-auto-tool-choice --tool-call-parser <name>` and `_build_sglang` emits `--tool-call-parser <translated_name>` (via `_SGLANG_TOOL_PARSER_MAP`). Omitted from JSON when empty. |
| `cached_path` | Optional absolute path to a faster local copy. Set by the TUI cache action; cleared on uncache. `resolve_path()` returns this when non-empty and present on disk, otherwise returns `path`. Multiple entries may share the same `cached_path` when they point to the same source directory — `save_cached_path()` fans out to all of them. || `server_status` | `{server_id: {quant_key: status}}` — outer key is server id; inner key is `"none"` (full precision) or a `RuntimeQuant.value` (e.g. `"bnb_int8"`); value is `"good"` / `"bad"` / `"unknown"`. Only one quant key may be `"good"` per server (exclusive). |
| `server_params` | Per-server overrides merged on top of server defaults at launch time |

Multiple `AIModel` entries for the same model family (BF16 / INT8 / Q8_0) are intentional — each format has different servers, VRAM footprint, and launch flags.

### QuantizationTool

Describes an offline conversion/requantization tool.  Instance data lives in `data.py` alongside `SERVERS`.

| Field | Purpose |
|---|---|
| `id` | Unique slug, e.g. `"llama-quantize"`, `"exl2-convert"` |
| `label` | Display label in the TUI picker |
| `input_formats` | `frozenset[ModelFormat]` — formats this tool accepts as input |
| `output_format` | `ModelFormat` — the format it produces |
| `quant_options` | `tuple[str, ...]` — ordered option identifiers shown in the quant picker |
| `quant_option_labels` | `dict[str, str]` — human-readable label per option |
| `cuda_home` | `CUDA_HOME` override injected at run time (non-empty to set) |

`tools_for_model(tools, model)` mirrors `servers_for_model` — returns tools whose `input_formats` contains `model.format`.

### ServerSoftware

Launch-time process setup is encoded as fields on `ServerSoftware` (set in `data.py`, consumed by `launch.py`):

| Field | Type | Purpose |
|---|---|---|
| `strip_pythonpath` | `bool` | Pop `PYTHONPATH` from env before spawning |
| `cuda_home` | `str` | Set `CUDA_HOME` in env when non-empty |
| `uses_staging_dir` | `bool` | Create symlinks under `model_dir` before launch |
| `stage_all_models` | `bool` | When `True`, stage every compatible catalog model (ComfyUI — model is chosen per API request, not at launch) |
| `staging_format_suffixes` | `dict[ModelFormat, str]` | Maps format → file extension suffix for the staging symlink (e.g. `{GGUF: ".gguf"}`; empty = no suffix) |
| `staging_subdir_attr` | `str \| None` | Name of an `AIModel` field whose value is appended as a subdirectory within `model_dir` (e.g. `"comfyui_subdir"` → `model.comfyui_subdir`) |
| `staging_use_source_filename` | `bool` | When `True`, the staged symlink is named after the source file (`Path(resolve_path()).name`) rather than `model.id + suffix`. Use for servers (e.g. ComfyUI) where the client selects models by original filename, not catalog ID. |
| `launch_cwd` | `str \| None` | Working directory for the spawned process (`None` = inherit) |
| `max_gpus` | `int` | Maximum GPUs selectable in `GpuSelectScreen`: `0` = unlimited (all GPUs, multi-select); `1` = single GPU only (radio-button) |
| `gpu_count_param` | `str \| None` | When set, `GpuSelectScreen` injects `{gpu_count_param: selected_count}` as a `param_overrides` entry so parallelism flags (e.g. `pipeline_parallel` for vLLM/SGLang) stay consistent with the selected GPU count. `None` for servers that handle variable GPU counts automatically (tabbyAPI `--gpu-split-auto`, llama.cpp layer distribution) |

Defined once per server type in `data.py`. The `default_params` dict establishes baseline launch parameters; per-model `server_params` only specify deltas.

### ModelFormat vs Server Compatibility
|---|---|---|
| `GGUF` | `"gguf"` | llamacpp, ooba, sglang |
| `SAFETENSORS_BF16` | `"safetensors_bf16"` | vllm, tabby, ooba, sglang, comfyui |
| `SAFETENSORS_FP16` | `"safetensors_fp16"` | vllm, tabby, ooba, sglang, comfyui |
| `SAFETENSORS_FP8` | `"safetensors_fp8"` | vllm, ooba, sglang, comfyui |
| `AWQ` | `"awq"` | vllm, ooba, sglang |
| `GPTQ` | `"gptq"` | vllm, tabby, ooba, sglang |
| `INT8_BNB` | `"int8_bnb"` | vllm, ooba, sglang |
| `EXL2` | `"exl2"` | tabby, ooba |
| `PYTORCH_BIN` | `"pytorch_bin"` | vllm, tabby, ooba, sglang |

`GGUFQuant` enum constants (in `models.py`) provide named values for the `quant` field on GGUF models — e.g. `GGUFQuant.Q4_K_M = "Q4_K_M"`. Covers F32/F16/BF16, legacy (Q4_0, Q5_0, Q8_0), K-quants (Q2_K … Q6_K), and I-quants (IQ1_S … IQ4_NL).

`servers_for_model()` filters by `model.format ∈ server.supported_formats`. A second filter checks `model.ai_types ∩ server.supported_ai_types ≠ ∅`. This means ComfyUI (image_gen only) does not appear when selecting a text-only model, and vice versa.

---

## Navigation and Screen Stack

The flow diverges based on whether the AI type uses `stage_all_models` (all servers for that type have `stage_all_models=True`). For text-gen types the user picks a model before a server; for image-gen types the launcher shows a hub screen first.

**Text-gen flow** (model selected at launch):
```
App.on_mount
    └─ one type, !hub_flow → ModelSelectScreen(type)
       multiple types      → AITypeScreen
                                └─ !hub_flow → ModelSelectScreen(chosen_type)
                                                    └─ ServerSelectScreen(model)
                                                            └─ [QuantSelectScreen]
                                                                    └─ GpuSelectScreen(server, model, quant)
                                                                            └─ ConfirmLaunch(description, factory)
                                                                                    └─ App.exit(factory)
```

**Image-gen (hub) flow** (`stage_all_models=True` for all servers of that type):
```
App.on_mount
    └─ one type, hub_flow  → ImageGenHubScreen(type)
       multiple types      → AITypeScreen
                                └─ hub_flow  → ImageGenHubScreen(chosen_type)

    ImageGenHubScreen
        ├─ Launch Server → ServerSelectScreen(model=None, ai_type=type)
        │                       └─ GpuSelectScreen(server, model=None)
        │                               └─ ConfirmLaunch(description, factory)
        │                                       └─ App.exit(factory, model=None, server)
        └─ Manage Models → ModelManageScreen(type)
                               └─ → / Enter → CacheActionScreen(model)
                                                   ├─ [Cache]   → CacheProgressScreen(model, mount)
                                                   └─ [Uncache] → removes cached copy, pops screen
```

`uses_hub_flow(servers, ai_type)` returns `True` when every server for a given AI type has `stage_all_models=True`. The check lives in `models.py` (domain logic) so neither `screens.py` nor `app.py` need to know about `stage_all_models` directly.

`App.exit(result)` where `result` is a `(factory, model, server_or_none)` 3-tuple — the TUI tears down, then `app.py::run()` calls `factory()` and blocks on `proc.wait()`. This means the TUI process *is* the parent process, so `Ctrl-C` propagates naturally.

For server launches `server` is a `ServerSoftware` instance; for quantize jobs it is `None`. `run()` uses `server is not None and model is not None` to decide whether to re-open the menu with a preselect (useful after a Ctrl-C to mark the test result immediately). Image-gen launches (model=None) do not generate a preselect — the menu re-opens from the top.

**`ImageGenHubScreen`**: Two-button menu. "Launch Server" pushes `ServerSelectScreen(model=None, ai_type=...)` which filters servers by type (via `servers_for_type()`), skips path validation, and suppresses g/x/? status marking. "Manage Models" pushes `ModelManageScreen` — all catalog management bindings work identically to `ModelSelectScreen`; `→`/`Enter` open `CacheActionScreen` for the focused model (cache/uncache) rather than advancing toward a server launch. Both buttons use `variant="primary"`.

**`ModelManageScreen`**: Subclass of `ModelSelectScreen` — overrides `_open_model()` (the template method called by both `action_dt_select` and `on_data_table_row_selected`) to push `CacheActionScreen` instead of `ServerSelectScreen`, and overrides `_title_text()` to show "manage models" in the title. `set_ctx_override` and `quantize_selected` are hidden via `_MANAGE_HIDDEN` in `check_action` (text-gen-only concepts irrelevant for diffusion models). The template method pattern is necessary because Textual calls all MRO classes' message handlers — both `@on` decorator and auto-named handlers — so standard Python MRO shadowing does not work for Textual message dispatch; only plain method calls (like `_open_model()`) respect the most-derived override.

**Root-screen back action**: Textual always keeps a blank `_DefaultScreen` at stack index 0. The root application screen sits at index 1. `action_dt_back()` on root screens (`ModelSelectScreen`, `AITypeScreen`) checks `len(screen_stack) > 2` (not `> 1`) before popping; at the root it calls `app.exit(None)` instead.

**Base screen classes**: Two shared base classes in `screens.py` eliminate boilerplate:
- `DataTableScreen(Screen)` — provides three hidden+priority bindings for `left`/`right`/`enter` dispatch (calling `action_dt_back` and `action_dt_select`). The default `action_dt_back` calls `app.pop_screen()`; root screens override it to exit with `app.exit(None)`. Subclasses define `action_dt_select`. All six DataTable/SelectionList screens inherit from this.
- `ButtonMenuScreen(Screen)` — provides `up`/`down` hidden bindings for focus cycling and the shared `action_press_focused` method. All seven button-based screens inherit from this.

**`→` / `Enter` on DataTable and SelectionList**: `priority=True` is required for `→`, `←`, and `enter` on screens backed by a DataTable or SelectionList — these widgets consume those keys before the screen sees them. However, Textual 8.x hoists `priority=True` visible bindings *before* non-priority ones in the footer, which breaks the intended ordering.

The solution is the **hidden-dispatch + phantom-key** pattern, centralised in `DataTableScreen`:
- `DataTableScreen.BINDINGS` provides three hidden+priority bindings (`left` → `dt_back`, `right`/`enter` → `dt_select`) — invisible to the footer (`show=False`).
- Each screen subclass adds its own **non-priority** visible bindings, which appear in definition order in the footer.
- The visible "Select" / "Confirm" footer label uses `f5` as its key (with `key_display="↵/→/dbl-click"`), positioned last. DataTable does not consume `f5`, so the non-priority binding shows at the last position.
- `enter` on DataTable screens is additionally covered by `on_data_table_row_selected` / `on_row_selected` event handlers (fired by mouse double-click), so double-click works independently.

Pure button screens need none of this — buttons don't consume `enter`, `right`, or `left`, so all their bindings are non-priority and appear in definition order.

**Back binding and `←` on DataTable screens**: `←` is consumed by DataTable for horizontal scrolling. A hidden+priority `Binding("left", action_back, "", show=False, priority=True)` intercepts it. The visible Back binding uses `"escape,b"` (neither consumed by DataTable) without priority.

**Footer binding conventions**: All screens follow a consistent ordering and glyph convention for footer bindings:

1. **Quit** — always first: `Binding("q", "quit_app", "Quit")`. Omitted on `SetCtxOverrideScreen` (Input captures `q`) and on `CacheProgressScreen` (`q` is the cancel key, not quit).
2. **Back / Cancel** — always second: `key_display="esc/←"`. Escape-only modals (e.g. `SetCtxOverrideScreen`) use `key_display="esc"`.
3. **Contextual bindings** — alphabetical by key character (e.g. `?`, `c`, `f`, `g`, `l`, `x`, `z`).
4. **Select / primary action** — always last:
   - DataTable screens: `key_display="↵/→/dbl-click"` — all three methods invoke the same action.
   - Button screens: `key_display="↵/→/Space"` — space activates the focused button.
   - `GpuSelectScreen`: `key_display="↵/→"` — space toggles GPU selection items, not confirm.

`CacheProgressScreen` is a special case: `escape,q,b,left` are all folded into a single Cancel binding displayed as `esc/←`; `q` means "abort transfer" here, not "quit app".

The App-level `q → quit_app` binding uses `show=False` to prevent duplication in the footer (each screen's own Quit binding takes precedence and displays first).

**Cursor preservation**: `ModelSelectScreen._populate()` saves `table.cursor_row` before rebuild and calls `table.move_cursor(row=saved_row)` after. This ensures navigating Back from the server screen and returning does not reset the position.

**`SetCtxOverrideScreen`**: Pressing `l` on the model list opens a modal `Input` screen. Accepts a plain integer or K-suffixed value (`32K`); blank or `0` clears the override. Values above the native ctx are permitted (for extended-context inference via RoPE scaling). On confirm, calls `save_ctx_override()` and pops itself — `ModelSelectScreen.on_screen_resume()` fires, re-sorts, and calls `_populate()` to refresh the Ctx column.

**Favorites (`f` key)**: Pressing `f` on a focused model calls `save_favorite(model, not model.favorite)`, toggling the pinned state. The star column (`★`) shows `★` for favorites and blank for others. Changing favorite state triggers an immediate re-sort and repopulate so the row moves to/from the top without requiring any other interaction.

**Quantize (`z` key)**: Pressing `z` on a focused model opens the quantize flow. `check_action` gates this to models where `tools_for_model(QUANTIZATION_TOOLS, model)` is non-empty. The flow:

```
ModelSelectScreen → z
    → (>1 tool) QuantizeToolScreen   — choose tool; skipped when exactly one applies
    → QuantizeOptionScreen            — choose quant level / bpw from tool.quant_options
    → QuantizeDestScreen              — choose output destination (CACHE_MOUNTS with free space)
    → ConfirmQuantize                 — show command + output path; exits TUI on confirm
```

`ConfirmQuantize.action_confirm()` calls `app.exit((factory, model, None))`. The `None` third element signals `run()` that this is a quantize job, not a server launch.

**Multi-column sort**: `ModelSelectScreen` tracks an ordered list of `(column_key, reverse)` pairs in `_sort_keys`. Index 0 is the primary sort, index 1 secondary, and so on. Clicking a column header:
- If it is the current primary column → toggles its sort direction.
- Otherwise → promotes it to primary (ascending), shifting the existing order one position down.

Groups of rows equal in the primary column cascade to the secondary, then tertiary, etc. The default sort order is `[("fav", False), ("status", False), ("name", False)]` — favorites first, then best-to-worst composite status, then A–Z by name.

**`_SORT_KEY` — single source of truth**: The module-level dict `_SORT_KEY: dict[str, Callable[[AIModel], Any]]` maps each column key to a sort-value extractor lambda. Both `_sort_models()` and `on_header()` derive their behaviour from this dict — `_sort_models` iterates it for sort values; `on_header` uses `col not in _SORT_KEY` as the guard for unsortable columns. **When adding a new table column, add one entry to `_SORT_KEY` and nothing else.**

Sort-direction conventions in `_SORT_KEY`:
- **Scalars** (`size`, `params`, `ctx`, `moe`): raw value — ascending = smallest first.
- **Boolean flags** (`fav`, `vision`, `trc`): expressed as `0 if flag else 1` so ascending (default) puts flagged/interesting rows at the top.
- **Cache**: tuple `(0, label)` if cached, `(1, "")` if not — cached rows come first; ties broken A–Z by mount label.
- **Text** (`name`, `format`, `note`): lower-cased string, A–Z ascending.

The `warn` column is intentionally excluded from `_SORT_KEY`: it is a derived visual icon whose underlying signals (model size vs VRAM, `min_sm`) are already captured by the `size` and `format` sort keys.

---

## GPU Selection

`GpuSelectScreen` sits between quant/server selection and `ConfirmLaunch` for every server launch. It lets the user choose which physical GPUs to expose to the spawned process.

**Mechanism**: `CUDA_VISIBLE_DEVICES` is the universal cross-tool GPU selector — every CUDA-based tool (vLLM, llama.cpp, tabbyAPI, SGLang, ComfyUI) respects it. Setting it to `"0,3,5"` remaps those physical GPUs to logical indices 0,1,2 for the process. Physical indices are stable (sourced from NVML/nvidia-smi).

**Screen behaviour**:
- Calls `poll_gpus()` synchronously on mount (fast NVML call); shows each GPU with current VRAM used/total.
- `max_gpus == 0` (default): multi-select with Space; `a` selects all.
- `max_gpus == 1`: radio-button behaviour — selecting a GPU auto-deselects any other. `SelectionList.SelectionToggled` is handled to enforce this.
- `→`/`Enter` confirms. Status bar shows the live `CUDA_VISIBLE_DEVICES` string and, when applicable, the auto-adjusted parallelism param.
- If `poll_gpus()` returns no valid rows (NVML unavailable), the screen skips itself and pushes `ConfirmLaunch` directly with the default device list.

**`gpu_count_param` auto-adjustment**: vLLM and SGLang require their parallelism flags to match the GPU count. When `server.gpu_count_param == "pipeline_parallel"`, `GpuSelectScreen` calls `compute_parallelism(n, model)` to determine the optimal TP×PP split and builds `param_overrides = {"tensor_parallel": tp, "pipeline_parallel": pp}`. The status bar shows the computed split (e.g. `PP=12` or `TP=2 × PP=7`). No auto-adjustment for tabbyAPI (`--gpu-split-auto`) or llama.cpp (distributes layers to visible GPUs automatically).

**Parallelism computation** (`compute_parallelism` in `launch.py`): reads the model's HF `config.json` for `num_attention_heads`, `num_key_value_heads`, `intermediate_size`, and `num_hidden_layers`. For PCIe topology (no NVLink), prefers the smallest valid TP (largest PP) to minimise cross-GPU tensor traffic. TP must evenly divide all three dimension fields. Additionally, for architectures *not* in `_TP_SCOPED_ARCHS`, `intermediate_size` must also be divisible by the total GPU count — this guards against model implementations that assert `intermediate_size % world_size == 0` using the global process count rather than the TP group size. When no valid plan exists for the selected GPU count, the count is automatically reduced to the largest compatible value and a note is shown in the status bar. Architecture is read from `model.arch` (catalog) with a fallback to `architectures[0]` in `config.json`. Results of `_read_hf_parallelism_fields` are LRU-cached per model path.

**`build_cmd` / `launch` signatures**:
- `build_cmd(server, model, runtime_quant, param_overrides)` — `param_overrides` is merged last in `_merged()`.
- `launch(server, model, runtime_quant, cuda_devices, param_overrides)` — `cuda_devices` overrides `CONFIG.resolved_cuda_devices`; `param_overrides` is forwarded to `build_cmd`.

---

## Command Building Pipeline

```
ServerSoftware.default_params
    + AIModel.server_params[server.id]   (merged at build time; skipped when model=None)
    + param_overrides                    (from GpuSelectScreen — e.g. pipeline_parallel = N)
    + AIModel.resolve_path() → injected as _model_path  (skipped when model=None)
    → _build_<server_id>(server, model, merged_params) → list[str]
```

All builders receive `model.effective_ctx` (not `model.ctx` directly) so that `ctx_override` values are respected without each builder needing to know about the field. Builders for `stage_all_models` servers (e.g. `_build_comfyui`) accept `model=None` and must not access model attributes.

`resolve_path()` returns `model.cached_path` if it is set and the path exists on disk (`Path.exists()` — handles both files and directories uniformly); otherwise returns `model.path`. The `mounts` list in `CacheMount` is used only to determine *where* to place new cached copies and to display disk-space info in the UI.

**Why inject `_model_path` into params rather than pass it separately?** To keep all builder functions with the same signature `(server, model, params) → list[str]`. New builders registered in `_BUILDERS` follow the same pattern without touching `build_cmd()`.

**Per-server environment injection**: `launch()` reads per-server launch fields from the `ServerSoftware` instance (set in `data.py`) rather than branching on `server.id`:
- `CUDA_VISIBLE_DEVICES` — set for all servers from `CONFIG.resolved_cuda_devices`
- `strip_pythonpath=True` (vLLM): removes `PYTHONPATH` to prevent source-tree shadowing of the installed package
- `cuda_home` (tabbyAPI, SGLang): sets `CUDA_HOME` in the environment; enforces the correct nvcc for flash-attn / flashinfer JIT compilation
- `uses_staging_dir=True` (tabbyAPI, ooba, ComfyUI): creates symlinks under `model_dir`; `staging_format_suffixes` maps `ModelFormat` → file extension suffix; `stage_all_models=True` (ComfyUI) stages every compatible catalog model rather than only the selected one; `staging_subdir_attr="comfyui_subdir"` (ComfyUI) routes each model to `model_dir/<model.comfyui_subdir>/` — `launch.py` reads `getattr(model, staging_subdir_attr)` generically with no branching on `server.id`
- `launch_cwd` (vLLM → `/tmp`, ooba → install dir): sets the working directory for the spawned process; vLLM avoids any `vllm/` directory on `sys.path`; ooba needs CWD at its install root because Gradio resolves `allowed_paths` relative to CWD

---

## Quantization Pipeline

### QuantizationTool registry (`data.py`)

`QUANTIZATION_TOOLS` is a `list[QuantizationTool]` analogous to `SERVERS`.  Each tool is keyed by `id` and matched to models via `tools_for_model(tools, model)` which filters on `tool.input_formats ∋ model.format`.

Initial tools:

| id | Input | Output | Binary |
|---|---|---|---|
| `llama-quantize` | `GGUF` | `GGUF` | `CONFIG.llama_quantize` (derived from `llama_server` sibling) |
| `exl2-convert` | `SAFETENSORS_BF16`, `SAFETENSORS_FP16`, `PYTORCH_BIN` | `EXL2` | `CONFIG.exl2_convert_script` or `-m exllamav2.conversion.convert_exl2` |

Adding a new tool: add a `QuantizationTool(...)` entry in `data.py` and a matching `_build_<id>` builder in `launch.py` registered in `_QUANT_BUILDERS`.

### Output path convention

`compute_quant_output_path(tool, model, quant_option, dest)` in `launch.py` always returns a **directory**. Every job gets its own uniquely-named directory so that removal is always consistent: delete the directory and its contents, regardless of format.

- **GGUF re-quantize**: `<dest.root>/<stem_no_quant>-GGUF-<quant>/`
  — The `.gguf` file inside is `<stem_no_quant>.<quant>.gguf`, placed there by the builder.
  Example: `<dest>/MyModel-GGUF-Q4_K_M/MyModel.Q4_K_M.gguf`

- **EXL2 convert**: `<dest.root>/<source_dirname>-exl2-<bpw>bpw/`
  — `source_dirname` is `Path(model.resolve_path()).name`.
  Example: `<dest>/MyModel-exl2-4.0bpw/`

The output directory is shown to the user in `ConfirmQuantize` before they confirm.

### `quantize()` function

Follows the same PTY + log structure as `launch()`:
- Log named `logs/{timestamp}_quantize_{tool.id}_{model.id}_{quant_option}.log`
- Output directory created before spawn
- `CUDA_HOME` set from `tool.cuda_home` when non-empty
- Same PTY window-size stamping and bash wrapper

### `exl2_convert_script` config field

`CONFIG.exl2_convert_script` (path, from `config.toml` key `exl2_convert_script`; default `~/exllamav2/convert.py`).  The `_build_exl2_convert` builder checks whether the script file exists on disk:
- If yes → `<tabby_python> <script> -i ... -o ... -b ...`
- If no → `<tabby_python> -m exllamav2.conversion.convert_exl2 -i ... -o ... -b ...`

`CONFIG.llama_quantize` is a derived property: `llama_server.parent / "llama-quantize"`. No extra config key needed.

---

## Log Capture

After the TUI exits, `launch()` in `launch.py` creates a log file and spawns the server process connected to a PTY:

1. **Log header** — a comment block written as plain text before spawn:
   ```
   # launcher session log
   # date:   <ISO timestamp>
   # model:  <id>  (<format>, <size_gb> GB)
   # server: <server_id>
   # path:   <resolved_path>
   # CUDA_VISIBLE_DEVICES=...
   #
   # command:
   <paste-ready command, one arg per line with backslash continuation>
   # ────────────────────────────────────────────────────────────────────────
   ```
   The command block has no `#` prefix so it can be copied directly to a terminal.

2. **PTY** — `pty.openpty()` creates a master/slave pair. The slave fd is given to the child as stdout+stderr. `TIOCSWINSZ` stamps the slave with the real terminal dimensions so width-sensitive output (tqdm progress bars etc.) renders correctly.

3. **Bash wrapper** — the actual command is wrapped as `bash -c "<shlex.join(cmd)>"`. This ensures shell signal diagnostics ("Segmentation fault (core dumped)") appear in the output and that the exit code uses bash's `128+signum` convention.

4. **Tee thread** — `_tee_output(master_fd, log_path)` runs in a background thread, reading from the PTY master and writing to both `sys.stdout` (terminal output) and the log file (`"ab"` mode to preserve the header). `OSError`/EIO on the read signals child EOF.

5. **SIGWINCH forwarding** — a `signal.signal(SIGWINCH, ...)` handler is installed around `proc.wait()` in `app.py::run()`. Each resize event re-issues `TIOCSWINSZ` on the master fd so the child sees the updated dimensions live.

6. **Exit footer** — after the tee thread joins, a footer line is appended:
   - `# exit: code <N>` for normal exits
   - `# exit: killed by SIGSEGV (11)` etc. for signal deaths (bash 128+N exit codes)

Log files are named `logs/{timestamp}_{server}_{model}.log` and are gitignored.

---

## tabbyAPI Staging Symlink Pattern

tabbyAPI requires all models to be subdirectories inside a single `--model-dir`. Rather than maintaining a permanent symlink forest, `launch()` creates `<tabby_model_dir>/<model.id>` → `resolved_path` immediately before spawning. Stale symlinks are replaced (unlink + re-link).

This generalises to any server that requires a model to be reachable under a specific directory: set `uses_staging_dir=True` and populate `staging_format_suffixes` if the server's scanner needs a file extension to detect the format. `launch()` reads `server.staging_format_suffixes.get(model.format, "")` to determine whether to append a suffix. No branching on `server.id` is needed.

---

## ComfyUI

ComfyUI differs from text generation servers in one key way: **it does not accept a model path at launch time**. Models are loaded on demand through its UI or HTTP API after startup; each API request specifies a checkpoint or diffusion model by name.

**Model staging**: at launch, `launch()` creates a symlink in the correct subdirectory of `comfyui_models_dir` (default `~/ComfyUI/models/`) for every image_gen model in the catalog:

| `model.comfyui_subdir` | Staging path | Model type |
|---|---|---|
| `"checkpoints"` (default) | `~/ComfyUI/models/checkpoints/` | SDXL / SD1.5 full checkpoints |
| `"diffusion_models"` | `~/ComfyUI/models/diffusion_models/` | FLUX UNets / diffusion transformers |

Each link target is `model.resolve_path()` — so a locally cached copy is used whenever one exists, and the source path is used otherwise. The symlink name is the **original source filename** (`Path(model.resolve_path()).name`) because ComfyUI workflows reference models by their natural filenames (e.g. `flux1-dev.safetensors`, `ae.safetensors`). This is controlled by `staging_use_source_filename=True` on the ComfyUI `ServerSoftware` entry.

This is fully driven by catalog entries. Adding a model to the catalog (via `ingest --ai-type image_gen`) and launching ComfyUI is all that's needed for it to appear in ComfyUI's model picker.

**Text encoders** (T5-XXL, CLIP-L) are not catalog entries — they are workflow components referenced directly by ComfyUI nodes. They are registered in `~/ComfyUI/extra_model_paths.yaml` under `text_encoders` pointing at the `t5/` directory.

**Installation**: Run directly from the cloned repo (`~/ComfyUI/`). `git pull` to update — no reinstallation needed. Dependencies live in the `comfyui` conda env (`requirements.txt` installed there).

**Multi-GPU (ComfyUI-MultiGPU / DisTorch)**: `~/ComfyUI/custom_nodes/ComfyUI-MultiGPU` (symlink to `~/ComfyUI-MultiGPU`) distributes model layers across multiple GPUs, giving FLUX and other large models access to more combined VRAM than a single GPU provides. Compute (latent space sampling) runs on the primary GPU; other GPUs act as VRAM donors holding weight blocks. All selected GPUs are exposed via `CUDA_VISIBLE_DEVICES`; the DisTorch node in the workflow controls how layers are distributed (e.g. `cuda:0,20gb;cuda:1,*`). Because multi-GPU is now supported, `max_gpus=0` (unlimited multi-select) is set on the ComfyUI `ServerSoftware` entry. Add `--cuda-device 0` to a model's `server_params["comfyui"]["extra_flags"]` only if you want to pin the primary compute device within the visible set.

---

## Server Status Persistence

Statuses are stored as a nested dict: `server_status[server_id][quant_key] = status`.

`quant_key` is `"none"` (full/stored precision — no additional quantization) or a `RuntimeQuant.value` string (e.g. `"bnb_int8"`). The helper `_quant_key(quant)` in `models.py` handles the conversion.

**Exclusivity rule**: only one quant key per `(model, server)` pair may be `"good"` at a time. `save_server_status()` clears any other `"good"` entries for the same server before writing the new one.

**`best_status_for(server_id)`** scans all quant keys for a server and returns `"good"` if any is good, `"bad"` if any is bad (and none good), otherwise `"unknown"`. Used for composite display and `ServerSelectScreen` dots.

**Where marks are set**:
- `ServerSelectScreen` (`g` / `x` / `?`) — only for servers where `available_runtime_quants(model.format)` returns empty (i.e. the model format is not quantizable or the server has no quant support). Calls `save_server_status(..., quant=None)`.
- `QuantSelectScreen` (`g` / `x` / `?`) — for quant-capable servers; marks the currently focused quant/precision option. Calls `save_server_status(..., quant=<focused_option>)`.
  `ServerSelectScreen` disables `g`/`x`/`?` for quant-capable servers (the bindings return `False` from `check_action`).

`save_ctx_override(model, value)` follows the same pattern: mutates `model.ctx_override` in-memory, then updates or removes the `ctx_override` key in the matching JSON entry (`value=0` → key removed).

`save_favorite(model, favorite)` follows the same pattern: mutates `model.favorite` in-memory, then sets or removes the `favorite` key in the JSON entry (`False` → key removed to keep the catalog tidy).

---

## vLLM-Specific Constraints

| Constraint | Root cause | Mitigations in this codebase |
|---|---|---|
| TP size must divide model dimensions | `num_attention_heads`, `num_key_value_heads`, and `intermediate_size` must be divisible by `--tensor-parallel-size` | `compute_parallelism()` picks the smallest valid TP for the selected GPU count |
| Global world_size assertions | Some model implementations assert `intermediate_size % world_size == 0` using total GPU count, not TP size | `compute_parallelism()` checks this for architectures not in `_TP_SCOPED_ARCHS` and auto-reduces GPU count |
| PCIe topology, no NVLink | Tensor parallel cross-GPU communication is expensive on PCIe | `compute_parallelism()` prefers smallest TP (largest PP) — only activations cross GPU boundaries |
| Ray compiled DAG limit | Default `_max_inflight_executions=10`; pp=11 submits 11 steps at once | Only relevant when `executor_backend="ray"` is set explicitly (no longer the default). Patch `ray_executor.py`: set `max(pp+1, 16)` |
| PYTHONPATH shadowing | `~/vllm/` source tree on `PYTHONPATH` shadows installed package | `launch()` removes `PYTHONPATH` from env before spawning vLLM |

### vLLM launch flags — data-driven, minimal by default

`_build_vllm` emits only the flags that are actually needed. Historically it force-set
`--distributed-executor-backend ray`, `--dtype bfloat16`, `--enable-chunked-prefill`,
`--enforce-eager`, `--max-num-seqs 64`, `--gpu-memory-utilization 0.97`, and an
unconditional `--trust-remote-code`. Several of these caused hard-to-diagnose load
failures (ray init/compiled-DAG issues, CUDA-graph OOM from `enforce-eager` + `0.97`
utilization, dtype mismatches). They are now optional and driven by `default_params`:

| Param (`default_params`) | Default | Flag emitted |
|---|---|---|
| `executor_backend` | `""` | `--distributed-executor-backend <v>` only when non-empty (else vLLM auto-picks `mp` for single-node) |
| `dtype` | `"auto"` | `--dtype <v>` only when not `"auto"` (else vLLM honours the model's `torch_dtype`) |
| `gpu_mem_util` | `0.90` | `--gpu-memory-utilization` always (vLLM default, leaves headroom) |
| `max_num_seqs` | `0` | `--max-num-seqs <v>` only when non-zero (else vLLM default 256) |
| `chunked_prefill` | `False` | `--enable-chunked-prefill` only when True |
| `enforce_eager` | `False` | `--enforce-eager` only when True (default allows faster CUDA graphs) |
| `log_requests` | `False` | `--no-enable-log-requests` when False; nothing when True |

Always-on flags: `--model`, `--served-model-name`, `--tensor-parallel-size`,
`--pipeline-parallel-size`, `--gpu-memory-utilization`, `--max-model-len`
(`model.effective_ctx`), `--host`, `--port`. `--trust-remote-code` is emitted only when
the model's `trust_remote_code` field is True. `--enable-auto-tool-choice
--tool-call-parser <name>` is emitted when the model's `tool_call_parser` is set.

Per-model overrides go in `server_params["vllm"]` (e.g. `{"enforce_eager": true,
"gpu_mem_util": 0.8}`) and are merged over these defaults at launch time.


---

## Cache Management

`cache_model()` runs `rsync -ah --info=progress2` from the model's `path` to a local cache mount. Single-file models are detected with `src_p.is_file()` (not extension-based, so `.safetensors`, `.gguf`, and any future single-file formats are handled correctly); directories use trailing-slash semantics to sync contents. `stdout=PIPE` is set so the TUI can read rsync output live; `stderr=PIPE` captures error text on failure. The returned `Popen` lets the UI poll progress without blocking.

`uncache_model()` deletes the cached copy only — it never touches the original source. Raises `FileNotFoundError` if `model.cached_path` is empty or the path no longer exists.

**Cache destination**: `cache_model()` places directory models at `mount.root / <cache_subdir> / src_p.name` (the source directory's own basename, e.g. `BeMag-123B-v0.1/`) and single-file models at `mount.root / <cache_subdir> / model.id / <filename>`. Using the source directory name means that multiple catalog entries pointing to the same source path (e.g. different format variants extracted from one directory) share the same cached location.

**Fan-out on save/clear**: `save_cached_path(model, path)` updates *all* in-memory `AI_MODELS` entries and all JSON catalog entries that share `model.path`, in a single catalog write. This keeps every co-path entry in sync when a directory is cached or uncached.

**Space estimation**: `cache_source_size_gb(path)` returns the actual on-disk footprint of the source, recursively summing all files while excluding `.git` subtrees. This is separate from `size_gb` (which records logical weight-file size only and is shown in the model list). `CacheActionScreen` computes the actual source size asynchronously on mount — buttons initially show `~{size_gb}` until the worker finishes and updates them with the accurate value. On rsync completion, `CacheProgressScreen` calls `save_cached_path(model, dest_path)`. `uncache_model()` deletes the path stored in `model.cached_path` and calls `save_cached_path(model, "")` to clear all co-path entries.

**Progress display**: `CacheProgressScreen._run_cache()` runs in a `@work(thread=True)` worker. It reads rsync stdout in a `select.select()` loop with a 0.5 s timeout, splits on `\r` and `\n` (rsync uses `\r` for in-place progress updates), and passes the most recent non-empty fragment to `_update_progress_line()`. That method parses `(\d+)%\s+([\d.]+\S+/s)\s+([\d:]+)` to extract percentage, speed, and ETA, then updates a Textual `ProgressBar` widget and a detail `Label`. After stdout EOF, `proc.wait()` is called to ensure `returncode` is populated (EOF alone does not guarantee this) before checking success. Stderr content is surfaced in the detail label on failure.

---

## Extension Points

**Add a model**: create an `AIModel` entry in `models_local.json`. No code changes needed.

**Add a server type**:
1. `ServerSoftware(...)` entry in `data.py` → `SERVERS`.
2. `_build_<id>(server, model, params) → list[str]` in `launch.py`.
3. Register in `_BUILDERS`.
4. If the server needs a staging model directory, add a block in `launch()`.

**Add a cache tier**: add a `[[cache_mounts]]` entry in `config.toml`. Priority determines probe order. No code changes needed.

**Add an AIType**: extend the `AIType` enum in `models.py`; create corresponding `AI_TYPE_LABELS` entry in `screens.py`; tag models and servers with the new type.

---

## Catalog Ingest Tool (ingest.py)

```
python -m launcher.ingest ingest <scan_root> [--all] [--ai-type text_gen|image_gen] [--dry-run]
python -m launcher.ingest ingest --path <path> [--ai-type text_gen|image_gen] [--dry-run]
python -m launcher.ingest refresh [--path <path>] [--dry-run]
python -m launcher.ingest copy-configs <scan_root> [--dry-run]
python -m launcher.ingest copy-mmproj <scan_root> [--dry-run]
```

Scans a given `scan_root` directory for model paths not already in `models_local.json` and appends new entries. Without `--all`, stops after the first new model (useful for testing detection); with `--all`, ingests every uncatalogued model in one pass.

### Subcommands

| Subcommand | Purpose |
|---|---|
| `ingest` | Add uncatalogued models — scan a directory or target a specific path |
| `refresh` | Re-detect auto-derived fields for existing entries without touching user-curated fields |
| `copy-configs` | Copy HF config/tokenizer files from original model dirs into GGUF bundle directories |
| `copy-mmproj` | Copy mmproj-*.gguf vision encoder files from original model dirs into GGUF bundle directories |

### Scanning order

Entries are sorted alphabetically. For each item in `scan_root`:

- **Top-level `.gguf` file** → yielded as a standalone candidate.
- **GGUF bundle directory** → each `.gguf` file inside yielded individually (one entry per quant variant). Scans flat bundles (`entry/*.gguf`) and per-quant-subdir bundles (`entry/<quant>/*.gguf`). Files named `mmproj-*` are excluded.
- **HF model directory** (has `config.json` or weight files) → yielded as a whole directory. Directories with both safetensors and pytorch_bin weights yield one entry per format so each becomes a separate catalog entry.

Directories with no `config.json` and no weight files are skipped.

### Detection — GGUF files

- **`quant`**: matched against `GGUFQuant` values by regex on the filename (longest match first to avoid partial matches).
- **`ctx`**: from `config.json` if present in the bundle dir (placed by `copy-configs`); falls back to borrowing from a sibling entry already in the catalog; `0` if neither available.
- **`size_gb`**: from `stat().st_size` of the individual GGUF file.
- **`params_m`**: from `config.json` `num_parameters` if present; falls back to regex extraction from filename (e.g. `70B` → 70000).
- **`arch`**, **MoE**, **`vision`**, **`trust_remote_code`**: from `config.json` when present.
- **`mmproj_path`**: auto-detected from `mmproj-*.gguf` in bundle root when `vision=True`.
- **`id`**: `<name-slug>-<quant-slug>` where quant slug drops underscores and trailing `0` (e.g. `Q8_0` → `q8`, `Q4_K_M` → `q4km`). Collision guard appends `-gguf` if the ID already belongs to a different path.

### Detection — HF model directories

Reads `config.json`:

- **`format`**: `quantization_config.quant_type` first (gptq / awq / exl2 / int8_bnb), then `quant_method` (fp8, bitsandbytes, modelopt), then `torch_dtype` for bf16/fp16/fp8, then file-type presence for pytorch_bin. Legacy GPTQ files detected by `4bit`/`8bit`/`gptq` patterns in safetensors filenames.
- **`ctx`**: first of `max_position_embeddings`, `max_seq_len`, `seq_length`, `context_length`, `n_positions`; `model_max_length` preferred when strictly larger (covers RoPE-extended models). Falls back to `text_config` for multimodal models.
- **`params_m`**: from `num_parameters` in `config.json`, or regex extraction from directory name.
- **`arch`**: from `architectures[0]` in `config.json`.
- **MoE fields**: `moe_experts` / `moe_active` from `num_local_experts` / `n_routed_experts` / `num_experts` and `num_experts_per_tok`.
- **`vision`**: `True` when `vision_config` is present.
- **`trust_remote_code`**: `True` when `auto_map` is present (custom architecture code).
- **`min_sm`**: set to `89` for static FP8 quantisation (non-dequantise, non-block-wise); `0` for block-wise FP8 (dequantises to BF16 at runtime, works on sm_86).
- **`size_gb`**: recursive `stat()` sum of weight files matching the format's glob patterns (format-aware — e.g. pytorch_bin counts only HF shard files, not unrelated `.bin` artefacts).
- **`id`**: `<name-slug>-<fmt-short>` (e.g. `…-bf16`, `…-fp16`, `…-int8`). Collision guard falls back to the raw directory name.

### Detection — image_gen models

With `--ai-type image_gen`, each `.safetensors` file in the scan directory becomes its own catalog entry:

- **`format`**: inferred from filename — `fp8`/`e4m3fn` → `safetensors_fp8`; `flux` → `safetensors_bf16`; everything else → `safetensors_fp16`.
- **`comfyui_subdir`**: `"diffusion_models"` for FLUX/diffusion/unet; `"checkpoints"` for standard checkpoints (SDXL, SD1.5).
- **`ai_types`**: `["image_gen"]`.

### refresh subcommand

`refresh` re-detects auto-derived fields (`ctx`, `size_gb`, `params_m`, `arch`, `moe_experts`, `moe_active`, `vision`, `mmproj_path`) for existing catalog entries. User-curated fields (`id`, `name`, `format`, `path`, `cached_path`, `ctx_override`, `favorite`, `server_status`, `server_params`, `min_sm`, `hw_note`, `comfyui_subdir`, `quant`, `trust_remote_code`, `note`) are never touched.

- `--path <path>` limits refresh to a single entry; otherwise all entries are refreshed.
- Size recompute uses the stored (user-curated) format, not the freshly-detected one — if a format has been manually corrected, the size reflects the corrected format's weight files.
- Zero-valued detections for `ctx` and `params_m` are treated as "unknown" and preserve the existing value rather than overwriting manually-set data.

### copy-configs / copy-mmproj

Preparation steps for GGUF bundles — run before `ingest` or `refresh` so that metadata detection has access to `config.json` and mmproj files.

Both commands search `scan_root` plus cache mount subdirs mirroring the same leaf directory (e.g. if `scan_root` is `/mnt/nas/ai_models/text-generation`, also checks `~/fast_models/text-generation` from configured cache mounts). For each GGUF bundle directory that lacks the target files, the original (non-GGUF) model is located via normalised name matching and the files are copied across.

Fields left for manual review: `ctx` for new GGUF bundles without config.json, `server_params`.

---

## Catalog Audit Notes

When auditing the model catalog for misclassifications or stale metadata, append findings to `user_data/model_audit.md` (gitignored — personal/local). Use the FIXED / NOTED / OPEN ITEMS structure. Key things to check:

- `torch_dtype` in each model's `config.json` — `float32` (BF16 weights stored as BF16 despite name), `float16` → `SAFETENSORS_FP16`, `float8_e4m3fn` → `SAFETENSORS_FP8`
- Actual disk size (`du -sh`) vs. `size_gb` in `models_local.json`
- `max_position_embeddings` in `config.json` vs. `ctx` in catalog
- Number and type of weight files (`.safetensors`, `.bin`, `.gguf`) to confirm declared format
