# Launcher — Design Document

## Purpose

A keyboard-driven Textual TUI that selects and launches large-model inference servers (vLLM, llama.cpp, tabbyAPI). The key objectives:

- Navigate a large model catalogue without touching config files each time
- Track every copy of a model across storage locations (NAS, local drives) and copy / delete them from the TUI
- Persist test-result annotations (good / bad / unknown) per model per server environment
- Run each server from one of several registered environments (e.g. a stable and an experimental vLLM install)
- Keep all system-specific paths out of version control
- Quantize existing models to new formats / quant levels from within the TUI

---

## File Map and Separation of Concerns

```
config.py                   Site-specific loader — reads user_data/config.toml, exposes CONFIG singleton
config.toml.example         Tracked template — copy to user_data/config.toml and edit
models.py                   Schema — enums, dataclasses, pure logic (no instances, no I/O); MODEL_FIELDS
params.py                   Typed launch parameters (ParamSpec kinds) — the schema behind every server setting
hf_config.py                Launch-planning facts from a model's config.json (parallelism fields, expert-parallel need)
data.py                     Generic instances — SERVERS (with their ParamSpecs), QUANTIZATION_TOOLS, catalog loading (committable)
storage.py                  Storage locations (LOCATIONS), mirrored paths, model files, copy + deletion plans
environments.py             Server runtime environments — Environment kinds (conda / native) + ENVIRONMENTS registry
environments.json.example   Tracked template — environment registry (copy to user_data/environments.json)
models_local.json.example   Tracked template — schema reference + 3 sample entries
launch.py                   Command builders + process spawner + catalog persistence (incl. copy/delete bookkeeping)
chat_template.py            Chat template reader + launch-time patches (TEMPLATE_PATCHES) → user_data/chat_templates/
fit.py                      Pre-launch memory-fit estimates (FitProbe per server — llama.cpp: llama-fit-params)
screens.py                  All Textual screens
app.py                      Top-level App class, entry to screen stack
gpu.py                      NVML polling via pynvml (shared with gpu_monitor)
gpu_power.py                Per-GPU power limits — PowerLimitStore (per-card choices) + PowerLimitStep (launch-command step)
scripts/gpu-power-limit     Idempotent applier: sudo nvidia-smi -pl only for GPUs whose limit differs
ingest.py                   CLI tool — scans a directory for new models and appends entries to models_local.json
user_data/                  Gitignored — all machine-specific and personal data
    chat_templates/           Patched chat templates, <model.id>.jinja (written at launch; regenerable)
    config.toml               Machine-specific paths and GPU counts
    environments.json         Registered environments per server (written by the TUI)
    gpu_power.json            Chosen GPU power limits keyed by GPU UUID (written by GpuSelectScreen; absent = all defaults)
    models_local.json         AI_MODELS catalog (model paths, quant details, test results)
```

**Rule**: nothing in `models.py` or higher imports from `config.py`. `config.py` may not import any sibling module. This prevents circular imports regardless of load order.

---

## Configuration Layering

```
user_data/config.toml (gitignored)
    └── config.py → SiteConfig dataclass (singleton CONFIG)
            ├── storage.py (LOCATIONS — [[locations]])
            ├── data.py    (SERVERS params defaults)
            ├── environments.py (conda_base → conda env locations)
            └── launch.py  (app paths, CUDA_VISIBLE_DEVICES, CUDA_HOME)

user_data/environments.json (gitignored, written by the TUI)
    └── environments.py → EnvironmentRegistry (singleton ENVIRONMENTS)
            └── launch.py / screens.py (interpreter + binaries for each launch)
```

`config.toml` holds hand-edited, machine-specific settings; `environments.json` holds the environment registry, which the TUI rewrites (add / rename / remove / default / note) — keeping it out of the TOML avoids clobbering hand-written comments.

`pipeline_parallel = 0` in `config.toml` triggers NVML auto-detection at import time. The result is stored as `CONFIG.pipeline_parallel` (never zero afterwards). `resolved_cuda_devices` builds `"0,1,...,N-1"` if no explicit device string was given.

`[memory] host_cache_ram` (a share of physical RAM such as `"60%"`, or integer MiB) is resolved once at load into `CONFIG.host_cache_ram_mib` (`None` when unset) and becomes llama.cpp's `cache_ram_mib` default (`--cache-ram`). It is a host-level budget rather than a per-server one because it is sized against the machine's RAM; other servers' host-side KV caches (SGLang hierarchical cache, vLLM KV offload) can draw on the same setting.

**Why TOML, not envvars or a Python file?** TOML is structured and typed without being executable. Envvars are awkward for multi-key sections (storage locations) and invisible at a glance. A Python file would need `exec()` or an import that could shadow package modules.

---

## The Privacy Split: data.py vs models_local.py

`data.py` contains the SERVERS list and quantization tools — fully config-driven (paths come from CONFIG) and safe to commit.

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
| `id` | Slug used as `--served-model-name` / llama.cpp `--alias`. Shared by copies of the model at other locations, so it is **not** the catalog key — an entry is identified by `AIModel.key` = `(path, format)` (one directory may hold several formats, each its own entry). |
| `path` | Absolute path to the model file or directory, inside a storage location. Each entry has exactly one path; a model on two drives is two entries. |
| `format` | Gates server compatibility (`ModelFormat` enum) |
| `ai_types` | frozenset of AIType — determines which type-selection menus it appears in |
| `ctx` | Native/trained maximum context length (tokens); used verbatim if no `ctx_override` is set |
| `ctx_override` | Optional launch override (tokens); `0` = use `ctx` (scaled by `rope_scaling` when set). Can be higher or lower than `ctx`. Persisted in JSON. |
| `rope_scaling` | Optional `RopeScaling` — JSON `{"factor": 4.0}` plus optional `"method"` (default `"yarn"`) and `"original_ctx"` (default `ctx`). Extends the default launch context to `ctx × factor` (`scaled_ctx`) on servers that can apply it; see "RoPE scaling" below. Omitted from JSON when unset. |
| `effective_ctx` | Property — returns `ctx_override` if non-zero, else `scaled_ctx` (`ctx`, or `ctx × rope_scaling.factor`). Builders receive it via `server.launch_ctx(model)`. |
| `favorite` | Boolean; `True` = pinned model floats to the top of the sorted list. Persisted in JSON (key omitted when `False`). |
| `min_sm` | Minimum CUDA compute capability required to load this model (integer, e.g. `89` for sm_89+). `0` = no constraint. Checked against `system_min_sm()` at display time. |
| `hw_note` | Optional human-readable explanation shown in the note bar when `hw_compat()` returns `False`. When omitted, `hw_compat_note` generates a default: `"Requires sm_{min_sm}+ GPU"`. |
| `comfyui_subdir` | `image_gen` models only — ComfyUI `models/` subdirectory to stage into. Default: `"checkpoints"` (omit from JSON). Use `"diffusion_models"` for FLUX UNets. Extensible: any valid subdirectory name accepted by ComfyUI. |
| `mmproj_path` | Absolute path to the companion `mmproj-*.gguf` vision encoder file. Set automatically by `ingest`/`refresh` when the file is present in the bundle root. Passed as `--mmproj` to llama.cpp and ooba at launch time — from the entry's own `mmproj_path`. Omitted from JSON when empty. |
| `mtp_path` | Absolute path to the companion `mtp-*.gguf` multi-token-prediction draft head. Set automatically by `ingest`/`refresh`. llama.cpp and ooba launch with `--spec-type draft-mtp --model-draft <mtp_path>` unless the `mtp` server param is `false`. Cached alongside the model. Omitted from JSON when empty. |
| `mtp_embedded` | `True` when the model carries its own MTP (NextN) layers. GGUF: `<arch>.nextn_predict_layers > 0` in its header (`gguf_header.py`). HF checkpoint: `config.json` declares `mtp_num_hidden_layers` / `num_nextn_predict_layers` > 0 **and** the weights include them (`mtp.*` tensors, or a layer index past `num_hidden_layers`) — quantized/abliterated copies often keep the keys but drop the weights. Set by `ingest`/`refresh`. llama.cpp / ooba use it (`--spec-type draft-mtp`, no draft file) only when their `mtp` param is `on` (many GGUFs keep NextN layers a llama.cpp build cannot run, e.g. GLM5-Next); vLLM uses it for `--speculative-config` when its `mtp` param is on (opt-in). Omitted from JSON when false. |
| `tool_call_parser` | vLLM tool-call parser name (e.g. `"openai"`, `"hermes"`, `"llama3_json"`). When set, `_build_vllm` emits `--enable-auto-tool-choice --tool-call-parser <name>` and `_build_sglang` emits `--tool-call-parser <translated_name>` (via `_SGLANG_TOOL_PARSER_MAP`). Omitted from JSON when empty. Editable on the model properties page (a `MODEL_FIELDS` entry, as are `trust_remote_code` and `note`). |
| `disabled_template_patches` | frozenset of `ChatTemplatePatch.id`s the model opts out of; every other patch matching its chat template is applied at launch (see *Chat Template Patches*). Toggled on the model properties page via `save_template_patch_enabled()`. Omitted from JSON when empty. |
| `family` | Shared by an original model and every copy, conversion and quantization of it (ingest assigns the slug of the model's name; `register_copy()` and `finalize_quantization()` carry it over). Groups related entries without a per-model link chain, so deleting an intermediate (e.g. an F16 GGUF used to make a Q8_0) loses nothing. |
| `server_status` | `{server_id: {env_name: {quant_key: status}}}` — server id → registered environment name → `"none"` (full precision) or a `RuntimeQuant.value` (e.g. `"bnb_int8"`) → `"good"` / `"bad"` / `"unknown"`. See *Server Status Persistence*. |
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
| `env_server_id` | Server whose **default environment** provides the tool (`"llamacpp"` → llama-quantize binary; `"tabby"` → exllamav2 Python) |
| `avail_executable` | Executable that must exist in that environment's bin dir |
| `avail_module` | Python module that must be importable in that environment |
| `pip_package` | Package `pip install`ed into that environment by the auto-install prompt; `""` = not auto-installable |
| `install_note` | Shown when the tool is unavailable and not auto-installable |

`tools_for_model(tools, model)` mirrors `servers_for_model` — returns tools whose `input_formats` contains `model.format`.

### ServerSoftware

Launch-time process setup is encoded as fields on `ServerSoftware` (set in `data.py`, consumed by `launch.py`):

| Field | Type | Purpose |
|---|---|---|
| `params` | `tuple[ParamSpec, ...]` | Every launch parameter — see *Launch Parameters* below. `default_params` (property) derives the baseline values; `param(key)`, `auto_params()` and `model_params(model)` (the specs shown for a model) are lookups |
| `env_kind` | `str` | `Environment.kind` registered for this server — `"conda"` (default) or `"native"` (llama.cpp) |
| `version_packages` | `tuple[str, ...]` | Python distributions whose installed versions label an environment in the UI and log header (e.g. `("vllm",)` → `vllm 0.26.0`) |
| `version_probe` | `VersionProbe \| None` | Command-based version detection for environments without package metadata — runs `executable args` in the env and formats the first matching regex (`patterns`, tried in order) with `template`. llama.cpp: `llama-server --version` → `llama.cpp b11461 (4d756bc72)`. Output is cached per executable mtime (`Environment.command_output`). `ServerSoftware.env_version_label(env)` combines both sources and is the single entry point for UI and log headers |
| `strip_pythonpath` | `bool` | Pop `PYTHONPATH` from env before spawning |
| `cuda_home` | `str` | Set `CUDA_HOME` in env when non-empty (overridden by a bundled toolkit, below) |
| `bundled_cuda_glob` | `str` | Glob, relative to the environment's site-packages, locating a pip-bundled CUDA toolkit (vLLM: `"nvidia/cu1[0-9]*"`). When it matches, that root becomes `CUDA_HOME` and its `lib/` is prepended to `LD_LIBRARY_PATH` / `LIBRARY_PATH` |
| `lib_symlinks` | `dict[str, str]` | `{link_name: target_glob}` linker-name symlinks self-healed inside the bundled toolkit's `lib/` before spawn (e.g. `libcudart.so` → `libcudart.so.13`) |
| `extra_env` | `dict[str, str]` | Extra `KEY=VALUE` pairs merged into the command's `env` prefix |
| `rope_scaling_formats` | `frozenset[ModelFormat]` | Formats for which the builder applies a model's `rope_scaling`. Empty = unsupported; `launch_ctx()` then caps a scaled model at its native `ctx` |
| `rope_scaling_env` | `dict[str, str]` | Env vars added only when `rope_scaling` is applied (vLLM `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`, SGLang `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`) |
| `chat_template_flag` | `str` | Flag taking a chat template file that replaces the model's own (llama.cpp / ooba `--chat-template-file`, vLLM / SGLang `--chat-template`, tabbyAPI `--prompt-template`). `""` = no override; chat template patches are then not applied |
| `uses_staging_dir` | `bool` | Create symlinks under `model_dir` before launch |
| `stage_all_models` | `bool` | When `True`, stage every compatible catalog model (ComfyUI — model is chosen per API request, not at launch) |
| `staging_format_suffixes` | `dict[ModelFormat, str]` | Maps format → file extension suffix for the staging symlink (e.g. `{GGUF: ".gguf"}`; empty = no suffix) |
| `staging_subdir_attr` | `str \| None` | Name of an `AIModel` field whose value is appended as a subdirectory within `model_dir` (e.g. `"comfyui_subdir"` → `model.comfyui_subdir`) |
| `staging_use_source_filename` | `bool` | When `True`, the staged symlink is named after the source file (`Path(model.path).name`) rather than `model.id + suffix`. Use for servers (e.g. ComfyUI) where the client selects models by original filename, not catalog ID. |
| `launch_cwd` | `str \| None` | Working directory for the spawned process (`None` = inherit) |
| `max_gpus` | `int` | Maximum GPUs selectable in `GpuSelectScreen`: `0` = unlimited (all GPUs, multi-select); `1` = single GPU only (radio-button) |
| `gpu_count_param` | `str \| None` | When set, `GpuSelectScreen` injects `{gpu_count_param: selected_count}` as a `param_overrides` entry so parallelism flags (e.g. `pipeline_parallel` for vLLM/SGLang) stay consistent with the selected GPU count. `None` for servers that handle variable GPU counts automatically (tabbyAPI `--gpu-split-auto`, llama.cpp layer distribution) |
| `concurrency_param` | `str \| None` | Param key holding the number of concurrently processed requests (llama.cpp: `"parallel"` slots). When set, `GpuSelectScreen` adjusts it with `+`/`-`, injects it into `param_overrides`, and remembers the launched value per model (`save_server_param`). The catalog `ctx` stays the per-request context; each builder translates (llama.cpp multiplies `--ctx-size` by the slot count) |

Defined once per server type in `data.py`. The `params` specs establish baseline launch parameters; per-model `server_params` only specify deltas.

### Launch Parameters (`params.py`)

Each server parameter is a `ParamSpec` — the single source of truth for its default, help text, how a value is shown and edited, which models it applies to and (for auto parameters) how it is resolved. Kinds are classes, so the properties page and launch resolution never branch on a parameter's type:

| Kind | Values | Edited by |
|---|---|---|
| `BoolParam` | `False` / `True` | picker (Space toggles) |
| `AutoBoolParam` | `None` (auto) / `True` / `False` — auto is resolved per launch by its `resolve(model, merged_params) → (value, reason)` | picker (Space cycles) |
| `ChoiceParam` | one of `choices`; `labels` names special values (e.g. `""` → "auto") | picker (Space cycles) |
| `IntParam` / `FloatParam` | numbers within `minimum`/`maximum`; `IntParam` optionally allows `None` (`none_label`) and keywords such as `"auto"` | text prompt with validation |
| `TextParam` | free text (`empty_label` names the blank value) | text prompt |
| `FlagsParam` | a list of extra command-line arguments | text prompt (shell syntax) |

Common fields: `hidden` (internal plumbing such as staging directories — never shown per model), `managed_by` (set elsewhere in the flow, e.g. TP/PP "set from the GPU selection" — shown read-only), `applies_to(model)` (e.g. MoE-only or MTP-only parameters). `default_value()` returns a fresh, comparable copy (lists for `FlagsParam`).

**Auto resolution**: `launch_params()` merges defaults, the model's `server_params` and `param_overrides`, then replaces every auto parameter still `None` with its resolved value — builders, the fit probe and the GPU screen only ever see booleans. `auto_param_states(server, model, overrides)` returns `(spec, value, reason)` for display (the GPU screen shows auto parameters that resolve on, e.g. `expert parallel: on (auto)`).

Current auto parameters:
- vLLM `expert_parallel` → `hf_config.expert_parallel_needed(model, tp)`: on when the expert width (`moe_intermediate_size`) does not divide by TP, or a TP shard does not align to the checkpoint's `weight_block_size` (Qwen3.8-Flash-Next FP8: 640 / 8 = 80 vs 128). Off at TP=1 (no effect there) and for dense models.
- llama.cpp / ooba `mtp` → on only for a sidecar `mtp_path` head; an embedded head needs an explicit on.

`MODEL_FIELDS` (`models.py`) reuses the same kinds for the simple user-editable `AIModel` fields (`tool_call_parser`, `trust_remote_code`, `note`), persisted with `save_model_field()`.

---

## Server Environments

Each server runs from one of several registered **environments** — the conda env or native build that provides its interpreter / binaries. Several can be registered per server (e.g. a stable `vllm` and an experimental `vllm-next` for testing a new architecture) so upgrading one never breaks a known-good setup.

**Kinds** (`environments.py`) — `Environment` is an `abc.ABC`; each kind encodes its own location, resolution and lifecycle so callers never branch on kind:

| Kind | Class | Location | Lifecycle supported by the launcher |
|---|---|---|---|
| `conda` | `CondaEnvironment` | `<conda_base>/envs/<name>` | create (empty with a Python version, or `--clone` of another env), rename (`conda rename`), delete (`conda env remove`) |
| `native` | `NativeEnvironment` | `bin_dir` (e.g. `~/llama.cpp/build/bin`) | registry-only (register / rename label / unregister) |

Common interface: `exists()`, `bin_dir`, `executable(name)`, `python`, `find_package_path(glob)` (site-packages-relative; conda only), `package_version(dist)` / `version_label(dists)` (read from `*.dist-info` names — no subprocess), `command_output(exe, args)` (cached per executable mtime; used by `VersionProbe`), `shares_underlying(other)`, and lifecycle hooks `create_cmd()` / `remove_cmd()` / `rename_cmd()` returning an argv (or `None` = registry-only). `extra_fields` declares kind-specific attributes the "add" dialog must prompt for (native: `bin_dir`), so the UI stays generic.

**Registry** — `EnvironmentRegistry` (singleton `ENVIRONMENTS`) loads `user_data/environments.json`:

```json
{"environments": {"vllm": [{"kind": "conda", "name": "vllm", "default": true, "note": "stable"},
                           {"kind": "conda", "name": "vllm-next"}],
                  "llamacpp": [{"kind": "native", "name": "llama.cpp", "bin_dir": "~/llama.cpp/build/bin", "default": true}]}}
```

Keyed by `ServerSoftware.id`. Names follow conda's rule (`[A-Za-z0-9_.-]+`) and are unique per server. `default_for(server_id)` returns the entry flagged `default` (else the first) — used to pre-focus the picker and to run quantization tools. Every mutation (`add`, `remove`, `rename`, `set_default`, `set_note`) persists immediately. One conda env may be registered for several servers; `sharing(env)` finds those entries so a rename updates them all and a delete is refused while others still use it.

**Launch resolution** — builders take the chosen environment: `str(env.python)` for Python servers, `env.executable("llama-server")` for llama.cpp. Environment-dependent setup (bundled CUDA toolkit, `lib_symlinks`) is resolved per environment at launch, so a second vLLM env with a different cu13 toolkit works without configuration.

**Lifecycle orchestration** (`EnvManageScreen` → `launch.py`) — the external command runs first in `CommandRunScreen` (live output; the screen cannot be left mid-operation); only on success is the registry updated. `apply_env_rename()` renames registry entries and migrates recorded results; `apply_env_removal()` unregisters and drops the env's results (the dialog states how many models are affected). A rename is validated up front (`rename_conflict()`) so the registry can never diverge from a conda rename that already happened.

---

## ModelFormat vs Server Compatibility
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
                                                    └─ ModelPropertiesScreen(model)   (Enter continues)
                                                            └─ ServerSelectScreen(model)
                                                            └─ [EnvSelectScreen]       (server has >1 environment)
                                                                    └─ [QuantSelectScreen]
                                                                            └─ GpuSelectScreen(server, env, model, quant)
                                                                                    └─ ConfirmLaunch(command, factory, model, server, env)
                                                                                            └─ App.exit(LaunchJob)

    ServerSelectScreen / EnvSelectScreen ── e ──→ EnvManageScreen(server)
```

**Image-gen (hub) flow** (`stage_all_models=True` for all servers of that type):
```
App.on_mount
    └─ one type, hub_flow  → ImageGenHubScreen(type)
       multiple types      → AITypeScreen
                                └─ hub_flow  → ImageGenHubScreen(chosen_type)

    ImageGenHubScreen
        ├─ Launch Server → ServerSelectScreen(model=None, ai_type=type)
        │                       └─ [EnvSelectScreen]
        │                               └─ GpuSelectScreen(server, env, model=None)
        │                                       └─ ConfirmLaunch(command, factory, None, server, env)
        │                                               └─ App.exit(LaunchJob(model=None))
        └─ Manage Models → ModelManageScreen(type)
                               └─ → / Enter → CopyModelScreen(model)
                                                   └─ [location] → CopyProgressScreen(model, dest)
```

`uses_hub_flow(servers, ai_type)` returns `True` when every server for a given AI type has `stage_all_models=True`. The check lives in `models.py` (domain logic) so neither `screens.py` nor `app.py` need to know about `stage_all_models` directly.

`App.exit(result)` returns either a `LaunchJob(factory, model, server, env)` or a `QuantizeJob` (both frozen dataclasses in `launch.py`). The TUI tears down, then `app.py::run()` calls the process factory and blocks on `proc.wait()`. This means the TUI process *is* the parent process, so `Ctrl-C` propagates naturally.

For server launches with a model, `run()` re-opens the menu after Ctrl-C with the `LaunchJob` as preselect (stack: model list → properties → server list): `ServerSelectScreen` focuses the server and, when it has more than one environment, `EnvSelectScreen` is pushed with the launched environment focused — so the result can be marked immediately. Image-gen launches (`model=None`) and interrupted quantization jobs reopen from the top. Successful quantization jobs finalize and focus the generated model.

**`ModelPropertiesScreen`** (after choosing a model; Enter / `→` continues to `ServerSelectScreen`): a summary (size, MoE, architecture, vision, MTP, location and other copies — settings are per copy, so it says which copies exist) and a table of every setting with its value and source (`default`, `this model`, `auto`, `GPU screen`). Sections: *Model* — context / YaRN (opens `ContextSettingsScreen`), chat template patches (`edit_template_patches()`; the row resolves the matching patches in a worker since the GGUF header may be on the NAS), then `MODEL_FIELDS`; then one section per compatible server listing `server.model_params(model)`. Auto values are previewed for a launch on all GPUs (`gpu_param_overrides(server, model, CONFIG.pipeline_parallel)`), so TP/PP show the planned split and `expert_parallel` shows `auto → on (reason)`. Rows are small objects (`_SectionRow`, `_ContextRow`, `_TemplatePatchesRow`, `_ModelFieldRow`, `_ServerParamRow` — the last two share `_SpecRow`) that know how to show, edit, cycle and reset themselves. Bindings: `e` edit (picker or validated prompt via `edit_spec_value()`), Space toggle/cycle picker kinds, `r` reset to default (only when overridden). Edits save immediately to this catalog entry.

**Environment step**: `ServerSelectScreen` shows an environment column (the single env with its versions, `N environments`, or `no environment`). A server with one environment skips `EnvSelectScreen`; with none, selection is refused with a hint to press `e`. `_continue_launch()` is the shared hand-off (used by both screens): it refuses a missing environment, then pushes `QuantSelectScreen` or `GpuSelectScreen`.

**`EnvManageScreen`** (`e`): table of the server's environments (★ default, name / missing flag, versions, note) with a detail bar (location, "also used by"). Bindings: `a` add, `c` create missing, `r` rename, `d` remove, `s` set default, `n` note. Multi-step flows are `@work` async methods using `push_screen_wait()` with three reusable dialogs: `TextPromptScreen` (validated single-line input), `ChoiceScreen` (message + button list) and `CommandRunScreen` (streams an external command's output; dismisses with success). Adding a conda env that does not exist offers: create empty (prompts for a Python version, default 3.11), clone any existing registered conda env (same-server first), register without creating, or cancel.

**`ImageGenHubScreen`**: Two-button menu. "Launch Server" pushes `ServerSelectScreen(model=None, ai_type=...)` which filters servers by type (via `servers_for_type()`), skips path validation, and suppresses g/x/? status marking. "Manage Models" pushes `ModelManageScreen` — all catalog management bindings work identically to `ModelSelectScreen`; `→`/`Enter` open `CopyModelScreen` for the focused model (copy to another location) rather than advancing toward a server launch. Both buttons use `variant="primary"`.

**`ModelManageScreen`**: Subclass of `ModelSelectScreen` — overrides `_open_model()` (the template method called by both `action_dt_select` and `on_data_table_row_selected`) to push `CopyModelScreen` instead of `ServerSelectScreen`, and overrides `_title_text()` to show "manage models" in the title. `quantize_selected` is hidden via `_MANAGE_HIDDEN` in `check_action` (a text-gen-only concept irrelevant for diffusion models). The template method pattern is necessary because Textual calls all MRO classes' message handlers — both `@on` decorator and auto-named handlers — so standard Python MRO shadowing does not work for Textual message dispatch; only plain method calls (like `_open_model()`) respect the most-derived override.

**Root-screen back action**: Textual always keeps a blank `_DefaultScreen` at stack index 0. The root application screen sits at index 1. `action_dt_back()` on root screens (`ModelSelectScreen`, `AITypeScreen`) checks `len(screen_stack) > 2` (not `> 1`) before popping; at the root it calls `app.exit(None)` instead.

**Base screen classes**: Shared base classes in `screens.py` eliminate boilerplate:
- `DataTableScreen(Screen)` — provides three hidden+priority bindings for `left`/`right`/`enter` dispatch (calling `action_dt_back` and `action_dt_select`). The default `action_dt_back` calls `app.pop_screen()`; root screens override it to exit with `app.exit(None)`. Subclasses define `action_dt_select`. All six DataTable/SelectionList screens inherit from this.
- `ButtonMenuScreen(Screen)` — provides `up`/`down` hidden bindings for focus cycling and the shared `action_press_focused` method. All seven button-based screens inherit from this.
- `GpuSelectionScreen(DataTableScreen)` — shared GPU polling, VRAM labels, multi-select / radio selection, and select-all controls. `GpuSelectScreen` adds server parallelism, slots, fit estimates, power limits, and launch confirmation; `QuantizeGpuScreen` adds offline quantization confirmation without server-only settings.

**`→` / `Enter` on DataTable and SelectionList**: `priority=True` is required for `→`, `←`, and `enter` on screens backed by a DataTable or SelectionList — these widgets consume those keys before the screen sees them. However, Textual 8.x hoists `priority=True` visible bindings *before* non-priority ones in the footer, which breaks the intended ordering.

The solution is the **hidden-dispatch + phantom-key** pattern, centralised in `DataTableScreen`:
- `DataTableScreen.BINDINGS` provides three hidden+priority bindings (`left` → `dt_back`, `right`/`enter` → `dt_select`) — invisible to the footer (`show=False`).
- Each screen subclass adds its own **non-priority** visible bindings, which appear in definition order in the footer.
- The visible "Select" / "Confirm" footer label uses `f5` as its key (with `key_display="↵/→/dbl-click"`), positioned last. DataTable does not consume `f5`, so the non-priority binding shows at the last position.
- `enter` on DataTable screens is additionally covered by `on_data_table_row_selected` / `on_row_selected` event handlers (fired by mouse double-click), so double-click works independently.

Pure button screens need none of this — buttons don't consume `enter`, `right`, or `left`, so all their bindings are non-priority and appear in definition order.

**Back binding and `←` on DataTable screens**: `←` is consumed by DataTable for horizontal scrolling. A hidden+priority `Binding("left", action_back, "", show=False, priority=True)` intercepts it. The visible Back binding uses `"escape,b"` (neither consumed by DataTable) without priority.

**Footer binding conventions**: All screens follow a consistent ordering and glyph convention for footer bindings:

1. **Quit** — always first: `Binding("q", "quit_app", "Quit")`. Omitted on `ContextSettingsScreen` (Input captures `q`) and on `CopyProgressScreen` (`q` is the cancel key, not quit).
2. **Back / Cancel** — always second: `key_display="esc/←"`. Escape-only modals (e.g. `ContextSettingsScreen`) use `key_display="esc"`.
3. **Contextual bindings** — alphabetical by key character (e.g. `?`, `c`, `f`, `g`, `x`, `z`).
4. **Select / primary action** — always last:
   - DataTable screens: `key_display="↵/→/dbl-click"` — all three methods invoke the same action.
   - Button screens: `key_display="↵/→/Space"` — space activates the focused button.
   - `GpuSelectScreen`: `key_display="↵/→"` — space toggles GPU selection items, not confirm.

`CopyProgressScreen` is a special case: `escape,q,b,left` are all folded into a single Cancel binding displayed as `esc/←`; `q` means "abort transfer" here, not "quit app".

The App-level `q → quit_app` binding uses `show=False` to prevent duplication in the footer (each screen's own Quit binding takes precedence and displays first).

**Cursor preservation**: `ModelSelectScreen._populate()` saves `table.cursor_row` before rebuild and calls `table.move_cursor(row=saved_row)` after. This ensures navigating Back from the server screen and returning does not reset the position.

**`ContextSettingsScreen`**: `e` on the *Context / YaRN* row of the model properties page opens a modal with two `Input`s: a YaRN factor (blank or `1` = off) and a ctx override (plain integer or K-suffixed value such as `32K`; blank or `0` = none). A live preview shows the resulting launch context, warns when it exceeds the (scaled) trained length, and lists which of the model's servers apply YaRN (derived from `rope_scaling_formats`). Editing the factor keeps an existing `method` / `original_ctx`. On confirm, calls `save_rope_scaling()` and `save_ctx_override()` and dismisses itself (`dismiss()`, not `pop_screen()`, so the properties page's `push_screen_wait()` resolves); the model list's Ctx column shows `effective/native` whenever they differ.

**Favorites (`f` key)**: Pressing `f` on a focused model calls `save_favorite(model, not model.favorite)`, toggling the pinned state. The star column (`★`) shows `★` for favorites and blank for others. Changing favorite state triggers an immediate re-sort and repopulate so the row moves to/from the top without requiring any other interaction.

**Quantize (`z` key)**: Pressing `z` on a focused model opens the quantize flow. `check_action` gates this to models where `tools_for_model(QUANTIZATION_TOOLS, model)` is non-empty. The flow:

```
ModelSelectScreen → z
    → (>1 tool) QuantizeToolScreen   — choose tool; skipped when exactly one applies
    → QuantizeOptionScreen            — choose quant level / bpw from tool.quant_options
    → QuantizeDestScreen              — choose output destination (LOCATIONS with free space; offline disabled)
    → QuantizeGpuScreen               — choose GPUs exposed to the quantization tool
    → ConfirmQuantize                 — show command + output path; runs job outside TUI
```

  `ConfirmQuantize.action_confirm()` returns a `QuantizeJob` containing the process factory and finalization metadata. After the child exits, `run()` reopens the launcher. Exit code 0 copies standard config/tokenizer files into EXL2 outputs, removes resumable converter work artifacts, adds the output to `models_local.json`, updates the in-memory catalog, and focuses the new model. Failed or interrupted jobs retain their work artifacts and are not catalogued.

`QuantizeGpuScreen` starts with all successfully polled GPUs selected and requires a non-empty selection. Space toggles a GPU, `a` selects all, and Enter / `→` advances. Physical NVML indices are sorted into `CUDA_VISIBLE_DEVICES`; confirmation Back preserves the selection. If polling yields no usable GPUs, quantization is blocked with an error notification rather than silently exposing all GPUs. The server picker's existing configured-device fallback is unchanged.

**Multi-column sort**: `ModelSelectScreen` tracks an ordered list of `(column_key, reverse)` pairs in `_sort_keys`. Index 0 is the primary sort, index 1 secondary, and so on. Clicking a column header:
- If it is the current primary column → toggles its sort direction.
- Otherwise → promotes it to primary (ascending), shifting the existing order one position down.

Groups of rows equal in the primary column cascade to the secondary, then tertiary, etc. The default sort order is `[("fav", False), ("status", False), ("name", False)]` — favorites first, then best-to-worst composite status, then A–Z by name.

**`_SORT_KEY` — single source of truth**: The module-level dict `_SORT_KEY: dict[str, Callable[[AIModel], Any]]` maps each column key to a sort-value extractor lambda. Both `_sort_models()` and `on_header()` derive their behaviour from this dict — `_sort_models` iterates it for sort values; `on_header` uses `col not in _SORT_KEY` as the guard for unsortable columns. **When adding a new table column, add one entry to `_SORT_KEY` and nothing else.**

Sort-direction conventions in `_SORT_KEY`:
- **Scalars** (`size`, `params`, `ctx`, `moe`): raw value — ascending = smallest first.
- **Boolean flags** (`fav`, `vision`, `trc`): expressed as `0 if flag else 1` so ascending (default) puts flagged/interesting rows at the top.
- **Location**: index of the model's location in `[[locations]]` order; paths outside every location sort last.
- **Text** (`name`, `format`, `note`): lower-cased string, A–Z ascending.

The `warn` column is intentionally excluded from `_SORT_KEY`: it is a derived visual icon whose underlying signals (model size vs VRAM, `min_sm`) are already captured by the `size` and `format` sort keys.

---

## GPU Selection

`GpuSelectScreen` sits between quant/server selection and `ConfirmLaunch` for every server launch. It lets the user choose which physical GPUs to expose to the spawned process.

**Mechanism**: `CUDA_VISIBLE_DEVICES` is the universal cross-tool GPU selector — every CUDA-based tool (vLLM, llama.cpp, tabbyAPI, SGLang, ComfyUI) respects it. Setting it to `"0,3,5"` remaps those physical GPUs to logical indices 0,1,2 for the process. Physical indices are stable (sourced from NVML/nvidia-smi).

**Screen behaviour**:
- Calls `poll_gpus()` synchronously on mount (fast NVML call); shows each GPU with current VRAM used/total and its power limit. The first GPU is highlighted.
- `max_gpus == 0` (default): multi-select with Space; `a` selects all.
- `max_gpus == 1`: radio-button behaviour — selecting a GPU auto-deselects any other. `SelectionList.SelectionToggled` is handled to enforce this.
- `p` / `P`: set the power limit for the highlighted / all selected GPUs (see Power limits below).
- `→`/`Enter` confirms. Status bar shows the live `CUDA_VISIBLE_DEVICES` string and, when applicable, the auto-adjusted parallelism param and the power-limit step.
- If `poll_gpus()` returns no valid rows (NVML unavailable), the screen skips itself and pushes `ConfirmLaunch` directly with the default device list.

**`gpu_count_param` auto-adjustment**: vLLM and SGLang require their parallelism flags to match the GPU count. When `server.gpu_count_param == "pipeline_parallel"`, `GpuSelectScreen` calls `compute_parallelism(n, model)` to determine the optimal TP×PP split and builds `param_overrides = {"tensor_parallel": tp, "pipeline_parallel": pp}`. The status bar shows the computed split (e.g. `PP=12` or `TP=2 × PP=7`). No auto-adjustment for tabbyAPI (`--gpu-split-auto`) or llama.cpp (distributes layers to visible GPUs automatically).

**Concurrency slots** (`server.concurrency_param`, llama.cpp `parallel`): a second status line shows `Slots N × <ctx> ctx`; `+`/`-` adjust N (1–64; the bindings are hidden for servers without the param). The starting value is the merged default (server default → model `server_params`), so the last-launched count comes back. `_launch_settings()` is the single source of the `(CUDA_VISIBLE_DEVICES, param_overrides)` pair used by both the fit estimate and the launch. The launch factory calls `save_server_param(model, server, concurrency_param, N)` when the launch actually happens (not on reaching `ConfirmLaunch`); a value equal to the server default removes the override.

**Memory-fit estimate** (`fit.py`): when a `FitProbe` is registered for the server (`_PROBES`, keyed by server id like `launch._BUILDERS`) and usable in the chosen environment, every selection or slot change re-estimates in a `@work(thread=True, exclusive=True, group="fit")` worker after a 0.4 s debounce. The result line shows a verdict — `✓ fits on GPU`, `⚠ only N layers fit on GPU` (llama.cpp would partially offload to CPU), `⚠ fits in total GPU memory, but needs layers rebalanced across GPUs`, or `✗ does not fit — X GiB short` — plus `needs X of Y GiB free · weights · KV (per slot) · compute`. A probe may report a provisional estimate first (`progress` callback → `searching=True`, shown with "searching…"), then the final one. A generation counter discards stale results; `exclusive=True` cancels the previous worker, and the probe polls `cancelled()` and kills its subprocess, so leaving the screen or launching never leaves an estimator holding GPU memory.

`LlamaCppFitProbe` runs `llama-fit-params` from the same build as `llama-server`, with the launch's model/context arguments (`llamacpp_model_args()`, shared with the builder) and `CUDA_VISIBLE_DEVICES` set to the selection. This is llama.cpp's own fitter (the one behind `llama-server --fit`): it reads only model metadata, so it is exact for every architecture — sliding-window, MLA, hybrid/recurrent — includes compute buffers, and measures against memory already in use by other processes.

The fitter logs its projection for the requested settings within ~2 s (a per-device `used / free vs. target` table and `projected to use … free`). If any device misses its target margin it then *searches* for a placement — rebalancing layers across GPUs, then partial CPU offload — testing allocations one by one on a single core, which for large MoE models at long context takes many minutes. So the probe streams the `-lv 4` log on reader threads and derives the verdict from that table itself, mirroring llama.cpp's rule (fits when every device keeps its target free; shortfall = Σ(target − free), which only offload can recover — verified against llama.cpp's own "need to use N MiB less"). That provisional estimate is reported via `progress` immediately; when the search finishes, stdout (`-c N -ngl M …`, `-ngl -1` = everything on GPU, for `n_gpu_layers=auto`) refines it, and on timeout (120 s) the provisional verdict is kept with `search_timed_out`. The table is logged on every path — including an explicit `n_gpu_layers`, where the fitter aborts non-zero ("already set by user") — so those get a verdict too. Not covered: `--mmproj` projectors (not accepted by `llama-fit-params`), MTP draft heads, and `extra_flags` — settings that change memory use belong in params emitted by `llamacpp_model_args()` (e.g. `ubatch`). Builds without `llama-fit-params` simply show no estimate.

**Power limits** (`gpu_power.py`): each GPU line shows the card's limit (`fmt_power_limit`: `250↓ W`, or `370→250↓ W` when the next launch will change it). `p` sets the limit for the highlighted GPU, `P` for all selected GPUs, via `TextPromptScreen` (whole watts within the cards' common min–max; blank = factory default). Choices persist immediately in `PowerLimitStore` (`POWER_LIMITS`, `user_data/gpu_power.json`), keyed by **full GPU UUID** so a limit follows the physical card across slot changes and applies to every model; a value equal to a card's default is stored as "no limit". `power_limit_step(rows, indices)` returns a `PowerLimitStep` (NVML index → watts, `None` = default) for the GPUs the launch uses (`_launch_indices()` — the selection trimmed to the parallelism plan), or `None` when every launched card is, and should stay, at its default. The step is included whenever a launched card has a choice *or* currently runs at a non-default limit, so the command also resets stray limits and stays reproducible. The status bar previews the step's arguments.

*Why apply at launch rather than at boot?* NVIDIA power limits are lost on reboot, driver reload and GPU reset; persistence mode does not help. Re-applying them in the launch command needs no privileged system setup (no root-owned service or helper), covers every reset cause, and puts the limits in the session log's reproducible command. The trade-off is that cards run at their defaults after a reboot until the next launcher (or pasted) launch, and servers started outside the launcher are unaffected.

*Why a script instead of `sudo nvidia-smi -pl` in the command?* `scripts/gpu-power-limit IDX[,IDX…]=WATTS|default …` queries the current limits first and only invokes `sudo nvidia-smi -i … -pl …` for GPUs that differ, so sudo (and its password prompt) is needed only after a reboot/reset rather than on every launch. The command runs in the server's PTY, but sudo prompts and reads the password on the controlling terminal (`/dev/tty`), which the child inherits from the launcher after the TUI has exited.

**Parallelism computation** (`compute_parallelism` in `launch.py`): reads the model's HF `config.json` for `num_attention_heads`, `num_key_value_heads`, `intermediate_size`, `num_hidden_layers`, and the linear-attention head counts `linear_num_key_heads` / `linear_num_value_heads` (hybrid Gated-DeltaNet models). Multimodal checkpoints nest these under `text_config`, which takes precedence over the top level. For PCIe topology (no NVLink), prefers the smallest valid TP (largest PP) to minimise cross-GPU tensor traffic. TP must evenly divide the attention, linear-attention and intermediate dimensions; `num_key_value_heads` must be divisible by TP *or* TP a multiple of it (vLLM/SGLang replicate KV heads when TP exceeds them). Architectures in `_NO_PP_ARCHS` (Qwen4Exp — its N-gram PLE embedding needs raw `input_ids`, which non-first PP ranks never receive) get the largest valid TP with PP=1, even if some selected GPUs go unused; the status bar notes why. Additionally, for architectures *not* in `_TP_SCOPED_ARCHS`, `intermediate_size` must also be divisible by the total GPU count — this guards against model implementations that assert `intermediate_size % world_size == 0` using the global process count rather than the TP group size. When no valid plan exists for the selected GPU count, the count is automatically reduced to the largest compatible value and a note is shown in the status bar. Architecture is read from `model.arch` (catalog) with a fallback to `architectures[0]` in `config.json`. The fields come from `hf_config.read_parallelism_fields()` (also the expert-parallel resolver's source), LRU-cached per model path. `gpu_param_overrides(server, model, n)` turns a GPU count into the parallelism params (planned TP×PP for `gpu_count_param == "pipeline_parallel"` servers) — shared by `GpuSelectScreen` and the properties page preview.

**`build_cmd` / `launch` signatures**:
- `build_cmd(server, env, model, runtime_quant, param_overrides)` — resolves params via `launch_params()` (`_merged()` with `param_overrides` last, auto parameters resolved, plus the injected `_model_path` / `_runtime_quant`); the fit probe uses the same resolver.
- `build_launch_command(server, env, model, runtime_quant, cuda_devices, param_overrides, power_step)` — wraps `build_cmd()` in the explicit environment and working-directory shell syntax shown in `ConfirmLaunch`; a `power_step` is prepended as `<gpu-power-limit …> && \` so the server never starts with its limits unapplied.
- `launch(server, env, model, runtime_quant, cuda_devices, param_overrides, power_step, command)` — `command` is the shell text edited in `ConfirmLaunch`; it is logged and executed verbatim. Callers that omit it get the default from `build_launch_command()`.

**Editable confirmation**: `ConfirmLaunch` displays the complete shell command in a focused `TextArea`. The user may change arguments, environment assignments, or shell syntax before launching with `Ctrl+Enter` or the Launch button. The edited text becomes the zero-argument process factory returned through `App.exit()`, so the preview, session log, and command passed to Bash cannot diverge. `Ctrl+Q` quits from this screen because plain `q` must remain available for editing.

---

## Command Building Pipeline

```
ServerSoftware.default_params            (from server.params specs)
    + AIModel.server_params[server.id]   (merged at build time; skipped when model=None)
    + param_overrides                    (from GpuSelectScreen — e.g. pipeline_parallel = N, parallel slots)
    → auto parameters still None resolved (AutoBoolParam.resolve)
    + AIModel.path → injected as _model_path  (skipped when model=None)
    + patched_template(model) → injected as _chat_template (path, or "" when no patch applies)
    → _build_<server_id>(server, env, model, merged_params) → list[str]
```

All builders take their context from `server.launch_ctx(model)` (not `model.ctx` directly), which is `model.effective_ctx` — so `ctx_override` and `rope_scaling` are respected without each builder knowing the fields — except that a `rope_scaling` model on a server that cannot apply it is capped at its native `ctx`. It is always the **per-request** context; a builder whose server sizes context for all concurrent requests together translates it (llama.cpp: `--ctx-size launch_ctx × parallel`).

**RoPE scaling** (`rope_scaling`, e.g. YaRN for Qwen3.8-Flash-Next 256K → 1M): when `server.applies_rope_scaling(model)`, each builder translates it, and `build_launch_command()` adds `server.rope_scaling_env`:

| Server | Flags |
|---|---|
| llama.cpp | `--rope-scaling <method> --rope-scale <factor> --yarn-orig-ctx <original>` — in `llamacpp_model_args()`, so `llama-fit-params` estimates the scaled KV cache too |
| ooba (GGUF) | the same flags inside `--extra-flags '…'`, which ooba forwards to its llama-server (a model's own ooba `--extra-flags` in `extra_flags` would override them) |
| vLLM | `--hf-overrides <json>` |
| SGLang | `--json-model-override-args <json>` |

The vLLM/SGLang JSON (`_hf_rope_override()`) starts from the model's own RoPE block in `config.json` (`text_config` for multimodal checkpoints; `rope_parameters` in transformers ≥ 5, else `rope_scaling`) and sets `rope_type`, `factor` and `original_max_position_embeddings`: the servers replace the block wholesale, so fields like `mrope_section`, `partial_rotary_factor` and `rope_theta` must be carried over. Frameworks implement static YaRN, which can cost quality on short prompts, so it is opt-in per model. Builders for `stage_all_models` servers (e.g. `_build_comfyui`) accept `model=None` and must not access model attributes.

Builders use the entry's own `path` and companion paths: the user picks which copy to launch by picking its row.

Builders take the interpreter / binaries from `env` (`env.python`, `env.executable(...)`) — never from config.

**Why inject `_model_path` into params rather than pass it separately?** To keep all builder functions with the same signature `(server, env, model, params) → list[str]`. New builders registered in `_BUILDERS` follow the same pattern without touching `build_cmd()`.

**Per-server environment injection**: `build_launch_command()` reads per-server launch fields from the `ServerSoftware` instance (set in `data.py`) and the chosen `Environment` rather than branching on `server.id`. Rather than mutating the child process's environment, launcher-controlled values are assembled into a leading `env KEY=VALUE …` **command prefix** (using the `env` utility). This keeps the confirmation editor and session log fully reproducible by copy/paste. `os.environ` is still inherited as the base; the explicit command assignments override it. Injected behavior:
- `CUDA_VISIBLE_DEVICES` — set for all servers from `CONFIG.resolved_cuda_devices`
- `strip_pythonpath=True` (vLLM): emits `env -u PYTHONPATH` to prevent source-tree shadowing of the installed package
- `cuda_home` (tabbyAPI, SGLang): sets `CUDA_HOME`; enforces the correct nvcc for flash-attn / flashinfer JIT compilation
- `bundled_cuda_glob` (vLLM): when the chosen environment contains a pip-bundled toolkit (`nvidia/cu13/`), it becomes `CUDA_HOME` and its `lib/` is prepended to `LD_LIBRARY_PATH` (runtime `.so` load) / `LIBRARY_PATH` (link-time `-l` resolution); the existing values are captured inline so the logged command reproduces exactly. `lib_symlinks` self-heals missing linker-name symlinks (e.g. `libcudart.so` → `libcudart.so.13`, the shortest glob match) in that `lib/` before spawn
- `extra_env`: arbitrary extra `KEY=VALUE` pairs merged into the prefix (no server currently sets any)
- `uses_staging_dir=True` (tabbyAPI, ooba, ComfyUI): creates symlinks under `model_dir`; `staging_format_suffixes` maps `ModelFormat` → file extension suffix; `stage_all_models=True` (ComfyUI) stages every compatible catalog model rather than only the selected one; `staging_subdir_attr="comfyui_subdir"` (ComfyUI) routes each model to `model_dir/<model.comfyui_subdir>/` — `launch.py` reads `getattr(model, staging_subdir_attr)` generically with no branching on `server.id`
- `launch_cwd` (vLLM → `/tmp`, ooba → install dir): emits a leading `cd <dir> &&` in the command; vLLM avoids any `vllm/` directory on `sys.path`; ooba needs CWD at its install root because Gradio resolves `allowed_paths` relative to CWD
- GPU power limits (any server, from `GpuSelectScreen`): a `PowerLimitStep` emits a leading `scripts/gpu-power-limit … &&` before everything else (see GPU Selection → Power limits)

---

## Chat Template Patches

Some chat templates reject conversations the model handles fine. Qwen 3.5 / 3.6 / 3.8 templates (official HF and Unsloth GGUF variants) raise `System message must be at the beginning.` for any system message after the leading one, whereas Qwen2.5 / Qwen3 templates rendered such messages in place as ChatML system turns. `chat_template.py` fixes these at launch without touching model files.

- **`ChatTemplatePatch`** (frozen dataclass): `id` (persisted key), `label` / `description` (UI), `pattern` (regex locating the faulty construct), `replacement` (inserted verbatim — no backreference processing, so Jinja `\n` escapes survive), `requires` (further regexes that must all match, guarding against templates where the replacement would be invalid). `TEMPLATE_PATCHES` is the registry.
- **`system_anywhere`** replaces the `raise_exception('System message must be at the beginning.')` statement with `{{- '<|im_start|>system\n' + content + '<|im_end|>\n' }}`. In both template variants the statement is reached only for system messages not already rendered as the leading system block, and `content` is the message loop's rendered content (a `requires` guard), so leading-system-only prompts render byte-identically. Verified with Python Jinja2 (vLLM / SGLang / ooba / tabby) and llama.cpp's own engine (`test-chat-template`).
- **Reading** (`read_chat_template(model)`): from `model.resolve_path()` — a GGUF's `tokenizer.chat_template` header key (via `gguf_header.py`), else `chat_template.jinja` → `chat_template.json` → `tokenizer_config.json` (transformers' precedence; for a list of named templates, the `"default"` one). Reads are LRU-cached per (file, mtime), so the per-build lookup is cheap even over the NAS.
- **Application**: `patched_template(model)` applies every matching patch not in `model.disabled_template_patches` and returns a `PatchedChatTemplate` (`None` when nothing applies). `launch_params()` injects its path as `_chat_template` for servers with a `chat_template_flag`; each builder emits `_chat_template_args()` just before `extra_flags` (so a hand-written override in `extra_flags` still wins). Like staging, the file is written by `launch()` — `user_data/chat_templates/<model.id>.jinja`, rewritten only when the content changed — and the log header gets a `# tmpl:` line naming the applied patches.
- **Why a sidecar file rather than patching the model?** Every text-gen server accepts a template override, so model files on every location stay pristine and the patch also works for models. Rewriting a GGUF's embedded template would change its header size, forcing a rewrite of the whole multi-GB file.
- **Opt-out**: automatic by default; the *Chat template patches* row of the model properties page runs `available_patches()` in a thread (the GGUF header may be on the NAS) and shows a `ChoiceScreen` toggling each matching patch, persisted via `save_template_patch_enabled()`.

---

## Quantization Pipeline

### QuantizationTool registry (`data.py`)

`QUANTIZATION_TOOLS` is a `list[QuantizationTool]` analogous to `SERVERS`.  Each tool is keyed by `id` and matched to models via `tools_for_model(tools, model)` which filters on `tool.input_formats ∋ model.format`.

Initial tools:

| id | Input | Output | Binary |
|---|---|---|---|
| `llama-quantize` | `GGUF` | `GGUF` | `llama-quantize` in the default `llamacpp` environment (built alongside `llama-server`) |
| `llama-hf-to-gguf` | `SAFETENSORS_BF16`, `SAFETENSORS_FP16`, `PYTORCH_BIN` | `GGUF` (`F16` or `BF16`) | `convert_hf_to_gguf.py` in the registered llama.cpp source tree; Python interpreter configured by `llama_convert_python` |
| `exl2-convert` | `SAFETENSORS_BF16`, `SAFETENSORS_FP16`, `PYTORCH_BIN` | `EXL2` | default `tabby` environment's Python: `CONFIG.exl2_convert_script` or `-m exllamav2.conversion.convert_exl2` |

Tools use their `env_server_id`'s default environment (`quant_tool_env(tool)`) for binaries and scripts; builders take `(tool, env, model, quant_option, output_path)`. A tool may use a separate configured Python interpreter and declare the modules needed to run its script. Availability is cached per tool, environment location, and interpreter, so switching the default environment or interpreter re-checks it.

Adding a new tool: add a `QuantizationTool(...)` entry in `data.py` and a matching `_build_<id>` builder in `launch.py` registered in `_QUANT_BUILDERS`.

### Output path convention

`compute_quant_output_path(tool, model, quant_option, dest)` in `launch.py` always returns a **directory**. Every job gets its own uniquely-named directory so that removal is always consistent: delete the directory and its contents, regardless of format.

- **GGUF output**: `<dest.root>/<model.type_subdir>/<source_stem>-GGUF-<quant>/`
  — GGUF re-quantization strips the source quant suffix; HF conversion keeps the source directory name. The builder writes a `.gguf` file inside the directory.
  Example: `<dest>/text-generation/MyModel-GGUF-Q4_K_M/MyModel.Q4_K_M.gguf` or `<dest>/text-generation/MyModel-GGUF-F16/MyModel.F16.gguf`

- **EXL2 convert**: `<dest.root>/<model.type_subdir>/<source_dirname>-exl2-<bpw>bpw/`
  — `source_dirname` is `Path(model.path).name`.
  Example: `<dest>/text-generation/MyModel-exl2-4.0bpw/`

The output directory is shown to the user in `ConfirmQuantize` before they confirm.

### `quantize()` function

Follows the same PTY + log structure as `launch()`:
- Log named `logs/{timestamp}_quantize_{tool.id}_{model.id}_{quant_option}.log`
- Output directory created before spawn
- `build_quantize_cmd(tool, model, quant_option, dest, cuda_devices=None)` includes an explicit `env CUDA_VISIBLE_DEVICES=…` prefix and `CUDA_HOME` from `tool.cuda_home` when non-empty. The picker passes its selected devices through confirmation to `quantize()`; non-TUI callers default to `CONFIG.resolved_cuda_devices`. Preview, log, and execution use the same builder.
- Same PTY window-size stamping and bash wrapper
- On exit code 0, `finalize_quantization()` copies the source's config/tokenizer files (from `ingest.model_metadata_dir()` — the HF directory itself, or the directory beside / one level above a GGUF) into the output, removes EXL2 work artifacts (`out_tensor`, calibration/hidden tensors, and job/measurement state), resolves the first shard of GGUF outputs, and registers the generated model through `ingest_model()`. The entry overrides start from `AIModel.derived_entry_fields()` — model-intrinsic facts and behaviour settings (`_DERIVED_FIELDS` in `models.py` plus `rope_scaling` and `disabled_template_patches`) — so values ingest cannot re-detect (e.g. a GGUF source without `config.json`) or that the user curated survive. Format-, size- and launch-specific fields (note, favorite, status, server params, ctx override, hardware requirements) are not inherited.
- The launcher then reopens with the generated model focused; failed or interrupted outputs remain on disk for inspection/resume but are not registered

### `exl2_convert_script` config field

`CONFIG.exl2_convert_script` (path, from `config.toml` key `exl2_convert_script`; default `~/exllamav2/convert.py`).  The `_build_exl2_convert` builder checks whether the script file exists on disk:
- If yes → `<tabby env python> <script> -i ... -o ... -b ...`
- If no → `<tabby env python> -m exllamav2.conversion.convert_exl2 -i ... -o ... -b ...`

`llama-hf-to-gguf` invokes llama.cpp's `convert_hf_to_gguf.py` with `--outtype f16|bf16`. It accepts full-precision, non-vision HF directories only; vision checkpoints are filtered out because this flow does not convert their separate projector. Conversion runs without CUDA/GPU selection and uses `CONFIG.llama_convert_python` (from `config.toml` key `llama_convert_python`; defaults to the Python running the launcher). That interpreter needs PyTorch, Transformers, and NumPy. The output GGUF and source config/tokenizer files are stored together; the normal GGUF ingest path detects the new model and records its F16/BF16 type.

---

## Log Capture

After the TUI exits, `launch()` in `launch.py` creates a log file and spawns the server process connected to a PTY:

1. **Log header** — a comment block written as plain text before spawn:
   ```
   # launcher session log
   # date:   <ISO timestamp>
   # model:  <id>  (<format>, <size_gb> GB)
   # server: <server_id>
   # env:    <env name>  (<kind>, <version_label>)
   # path:   <resolved_path>
   # tmpl:   <patched chat template path>  (patched: <patch ids>)   — only when patches apply
   #
   # command:
   <paste-ready command, one arg per line with backslash continuation>
   # ────────────────────────────────────────────────────────────────────────
   ```
  The command block has no `#` prefix so it can be copied directly to a terminal. It is the exact text accepted from `ConfirmLaunch`, with a default `env KEY=VALUE …` prefix carrying all launcher-injected environment variables (including `CUDA_VISIBLE_DEVICES`), a visible `cd` prefix when needed, and a leading `gpu-power-limit … &&` step when GPU power limits apply.

2. **PTY** — `pty.openpty()` creates a master/slave pair. The slave fd is given to the child as stdout+stderr. `TIOCSWINSZ` stamps the slave with the real terminal dimensions so width-sensitive output (tqdm progress bars etc.) renders correctly.

3. **Bash wrapper** — the exact edited command is passed to `bash -c`. This honors user-added shell syntax, ensures shell signal diagnostics ("Segmentation fault (core dumped)") appear in the output, and uses Bash's `128+signum` exit-code convention.

4. **Tee thread** — `_tee_output(master_fd, log_path)` runs in a background thread, reading from the PTY master and writing to both `sys.stdout` (terminal output) and the log file (`"ab"` mode to preserve the header). `OSError`/EIO on the read signals child EOF.

5. **SIGWINCH forwarding** — a `signal.signal(SIGWINCH, ...)` handler is installed around `proc.wait()` in `app.py::run()`. Each resize event re-issues `TIOCSWINSZ` on the master fd so the child sees the updated dimensions live.

6. **Exit footer** — after the tee thread joins, a footer line is appended:
   - `# exit: code <N>` for normal exits
   - `# exit: killed by SIGSEGV (11)` etc. for signal deaths (bash 128+N exit codes)

Log files are named `logs/{timestamp}_{server}_{model}.log` and are gitignored.

---

## tabbyAPI Staging Symlink Pattern

tabbyAPI requires all models to be subdirectories inside a single `--model-dir`. Rather than maintaining a permanent symlink forest, `launch()` creates `<tabby_model_dir>/<model.id>` → `resolved_path` immediately before spawning. Stale symlinks are replaced (unlink + re-link).

This generalises to any server that requires a model to be reachable under a specific directory: set `uses_staging_dir=True` and populate `staging_format_suffixes` if the server's scanner needs a file extension to detect the format. `_staged_name()` determines the entry name and `_stage_model()` creates it; builders that reference the staged model (ooba's `--model`) use `_staged_name()` too, so the two cannot drift. No branching on `server.id` is needed.

**Split GGUFs** are staged as a real directory `<model.id>/` holding one link per shard under its original filename, instead of a single renamed link: llama.cpp locates shards 2..N from the first shard's `-00001-of-000NN` name and directory. ooba is given `--model <model.id>/<shard-1 filename>` explicitly because its GGUF metadata reader takes the first hit of an unsorted glob. Re-staging removes a previous staging directory only when it contains nothing but links, so real model data in the staging dir is never deleted.

---

## ComfyUI

ComfyUI differs from text generation servers in one key way: **it does not accept a model path at launch time**. Models are loaded on demand through its UI or HTTP API after startup; each API request specifies a checkpoint or diffusion model by name.

**Model staging**: at launch, `launch()` creates a symlink in the correct subdirectory of `comfyui_models_dir` (default `~/ComfyUI/models/`) for every image_gen model in the catalog:

| `model.comfyui_subdir` | Staging path | Model type |
|---|---|---|
| `"checkpoints"` (default) | `~/ComfyUI/models/checkpoints/` | SDXL / SD1.5 full checkpoints |
| `"diffusion_models"` | `~/ComfyUI/models/diffusion_models/` | FLUX UNets / diffusion transformers |

Each link target is the model's `path`; a model with copies on several locations is staged once (`_one_copy_each()` — first copy present on disk in `[[locations]]` order). The symlink name is the **original source filename** (`Path(model.path).name`) because ComfyUI workflows reference models by their natural filenames (e.g. `flux1-dev.safetensors`, `ae.safetensors`). This is controlled by `staging_use_source_filename=True` on the ComfyUI `ServerSoftware` entry.

This is fully driven by catalog entries. Adding a model to the catalog (via `ingest --ai-type image_gen`) and launching ComfyUI is all that's needed for it to appear in ComfyUI's model picker.

**Text encoders** (T5-XXL, CLIP-L) are not catalog entries — they are workflow components referenced directly by ComfyUI nodes. They are registered in `~/ComfyUI/extra_model_paths.yaml` under `text_encoders` pointing at the `t5/` directory.

**Installation**: Run directly from the cloned repo (`~/ComfyUI/`). `git pull` to update — no reinstallation needed. Dependencies live in the `comfyui` conda env (`requirements.txt` installed there).

**Multi-GPU (ComfyUI-MultiGPU / DisTorch)**: `~/ComfyUI/custom_nodes/ComfyUI-MultiGPU` (symlink to `~/ComfyUI-MultiGPU`) distributes model layers across multiple GPUs, giving FLUX and other large models access to more combined VRAM than a single GPU provides. Compute (latent space sampling) runs on the primary GPU; other GPUs act as VRAM donors holding weight blocks. All selected GPUs are exposed via `CUDA_VISIBLE_DEVICES`; the DisTorch node in the workflow controls how layers are distributed (e.g. `cuda:0,20gb;cuda:1,*`). Because multi-GPU is now supported, `max_gpus=0` (unlimited multi-select) is set on the ComfyUI `ServerSoftware` entry. Add `--cuda-device 0` to a model's `server_params["comfyui"]["extra_flags"]` only if you want to pin the primary compute device within the visible set.

---

## Server Status Persistence

Statuses are stored as a nested dict: `server_status[server_id][env_name][quant_key] = status`.

`env_name` is a registered `Environment.name` for that server — results are per environment because a model may work on one install (e.g. `vllm-next`) and fail on another. `quant_key` is `"none"` (full/stored precision — no additional quantization) or a `RuntimeQuant.value` string (e.g. `"bnb_int8"`). The helper `_quant_key(quant)` in `models.py` handles the conversion. Several quant keys may be `"good"` at once.

**Lookups** (`AIModel`): `status_for(server_id, env_name, quant)` for one combo; `env_best_status(server_id, env_name)` across quants (EnvSelectScreen dots); `best_status_for(server_id)` across environments and quants (ServerSelectScreen dots, composite model-list status). All collapse with priority good > bad > unknown.

**Where marks are set** — always on the deepest selection screen for that launch, so a mark is never ambiguous:
- `ServerSelectScreen` (`g` / `x` / `?`) — only when the server has exactly one environment and `available_runtime_quants(model.format)` is empty. Calls `save_server_status(model, server_id, env_name, status, quant=None)`.
- `EnvSelectScreen` (`g` / `x` / `?`) — servers with several environments and no runtime quants for this model; marks the focused environment.
- `QuantSelectScreen` (`g` / `x` / `?`) — quant-capable servers; marks the focused quant/precision option for the chosen environment.
  Shallower screens disable `g`/`x`/`?` in those cases (the bindings return `False` from `check_action`).

**Environment renames / removals** migrate results catalog-wide in one write: `apply_env_rename()` moves `server_status[server][old]` → `[new]`; `apply_env_removal()` drops `server_status[server][name]` (`count_env_status()` feeds the confirmation dialog).

`save_ctx_override(model, value)` follows the same pattern: mutates `model.ctx_override` in-memory, then updates or removes the `ctx_override` key in the matching JSON entry (`value=0` → key removed). `save_rope_scaling(model, rope_scaling)` does the same for `rope_scaling` (`None` → key removed).

`save_favorite(model, favorite)` follows the same pattern: mutates `model.favorite` in-memory, then sets or removes the `favorite` key in the JSON entry (`False` → key removed to keep the catalog tidy).

`save_server_param(model, server, key, value)` persists one per-model launch parameter in `server_params[server.id]` (the remembered slot count; every edit on the properties page). A value equal to the spec's `default_value()` removes the key — that is how "reset" works — and emptied `server_params` dicts are dropped.

`save_model_field(model, spec, value)` does the same for a `MODEL_FIELDS` entry, omitting the key at its default.

---

## vLLM-Specific Constraints

| Constraint | Root cause | Mitigations in this codebase |
|---|---|---|
| TP size must divide model dimensions | `num_attention_heads`, linear-attention heads and `intermediate_size` must be divisible by `--tensor-parallel-size`; `num_key_value_heads` divisible by it or a divisor of it | `compute_parallelism()` picks the smallest valid TP for the selected GPU count |
| No pipeline parallelism for some architectures | Qwen4Exp's N-gram PLE embedding needs raw `input_ids` on every rank; vLLM rejects PP > 1 | `_NO_PP_ARCHS` → pure TP plan (PP=1) |
| FP8 block alignment of MoE experts under TP | Block-FP8 (128×128) experts cannot be TP-sharded when `moe_intermediate_size / TP` is not a multiple of 128 (Qwen3.8-Flash-Next: 640 / 8 = 80) | `expert_parallel` auto → `--enable-expert-parallel` exactly when needed (whole experts per rank; vLLM replicates the misaligned shared expert). Expert parallelism only acts at TP > 1, so pipeline-only launches never need it |
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
utilization, dtype mismatches). They are now optional and driven by the server's params:

| Param | Default | Flag emitted |
|---|---|---|
| `executor_backend` | `""` | `--distributed-executor-backend <v>` only when non-empty (else vLLM auto-picks `mp` for single-node) |
| `dtype` | `"auto"` | `--dtype <v>` only when not `"auto"` (else vLLM honours the model's `torch_dtype`) |
| `gpu_mem_util` | `0.90` | `--gpu-memory-utilization` always (vLLM default, leaves headroom) |
| `max_num_seqs` | `0` | `--max-num-seqs <v>` only when non-zero (else vLLM default 256) |
| `chunked_prefill` | `False` | `--enable-chunked-prefill` only when True |
| `expert_parallel` | auto | `--enable-expert-parallel` when resolved on — auto: needed at this TP (see *Launch Parameters*); MoE models only |
| `mtp` | `False` | `--speculative-config '{"method":"mtp","num_speculative_tokens":<mtp_tokens>}'` when on and the model is `mtp_embedded` — opt-in: the MTP weights and draft KV cost memory (Qwen3.8-Flash-Next FP8 at TP8: the auto-fitted context drops from 262K to ~190K) |
| `mtp_tokens` | `2` | `num_speculative_tokens`. vLLM defaults to the checkpoint's MTP layer count (1) and reuses the layer beyond it; each extra position is accepted less often (Qwen3.8-Flash-Next with 2: ~0.7 / ~0.4 per position, mean acceptance ~2.0 tokens/step) — read "SpecDecoding metrics" in the log to tune |
| `auto_fit_ctx` | `False` | `--max-model-len -1` instead of `model.effective_ctx` — vLLM starts at the native context and lowers it only as far as needed to fit the KV cache (logged as "Auto-fit max_model_len"); set per-model for memory-tight models |
| `enforce_eager` | `False` | `--enforce-eager` only when True (default allows faster CUDA graphs) |
| `log_requests` | `False` | `--no-enable-log-requests` when False; nothing when True |

Prefix caching is not a parameter: vLLM ≥ 0.31 enables it by default wherever the model supports it (hybrid Mamba/GDN models in "align" mode), and the launcher has no reason to override that.

Always-on flags: `--model`, `--served-model-name`, `--tensor-parallel-size`,
`--pipeline-parallel-size`, `--gpu-memory-utilization`, `--max-model-len`
(`model.effective_ctx`, or `-1` when `auto_fit_ctx`), `--host`, `--port`. `--trust-remote-code` is emitted only when
the model's `trust_remote_code` field is True. `--enable-auto-tool-choice
--tool-call-parser <name>` is emitted when the model's `tool_call_parser` is set.

Per-model overrides go in `server_params["vllm"]` (e.g. `{"enforce_eager": true,
"gpu_mem_util": 0.8}`) and are merged over these defaults at launch time.

---

## llama.cpp Launch Flags

`_build_llamacpp` = `llama-server` + `llamacpp_model_args()` (shared with `LlamaCppFitProbe`, so the estimate always describes the command that will run) + serving flags.

| Param | Default | Flag emitted |
|---|---|---|
| `n_gpu_layers` | `"auto"` | `--n-gpu-layers <v>` — `auto` lets llama.cpp's fitter place layers |
| `split_mode` | `"layer"` | `--split-mode <v>` |
| `parallel` | `1` | `--parallel <v>` and `--ctx-size effective_ctx × v` — with fixed slots llama.cpp splits `--ctx-size` evenly, so every slot gets the full per-request context. The `concurrency_param`, adjusted per launch on `GpuSelectScreen` |
| `kv_cache_type` | `"q8_0"` | `--cache-type-k <v> --cache-type-v <v>`; `""` = llama.cpp default (f16). q8_0 halves KV memory at negligible quality loss — for large dense models it is what lets several long-context slots fit (BeMag 123B: 5 × 128K is ≈117 GiB at q8_0 vs ≈220 GiB at f16) |
| `cache_ram_mib` | `CONFIG.host_cache_ram_mib` | `--cache-ram <v>` when not `None`. Host-RAM store of saved slot states, restored instead of re-processing a prompt when more conversations are active than slots; filled on demand. llama.cpp's 8 GiB default holds less than one full 128K slot of a large model |
| `ubatch` | `None` | `--ubatch-size <v>` when set (llama.cpp default 512). Compute buffers scale with ubatch × total context and outweigh the KV cache at YaRN 1M contexts (Qwen3.8-Flash-Next Q8_0, 3 × 1M: 187 GiB at 512 vs 131 GiB at 256); smaller values cost prompt-processing speed. In `llamacpp_model_args()`, so the fit estimate reflects it |
| `threads` | `0` | `--threads <v>` only when non-zero; otherwise llama.cpp uses the physical core count (hyperthreads only add contention for CPU math, and fully offloaded models barely use the CPU) |
| `mtp` | auto | MTP speculative decoding, via `_mtp_args()` (shared with ooba; receives the resolved value). auto → on only for a sidecar head: `--spec-type draft-mtp --model-draft <mtp_path>`. on: additionally `--spec-type draft-mtp` alone for an `mtp_embedded` model (opt-in per model, once its architecture's MTP graph is known to work on the build). off: none. Not passed to `llama-fit-params` (it rejects speculative flags), so the fit estimate excludes the head |

Always-on: `--model`, `--alias <model.id>`, `--host 0.0.0.0`, `--port`, `--log-prefix`, `--log-timestamps`; `--mmproj` when the model has a projector. `extra_flags` are appended last.

Possible extension: newer builds add `--kv-unified-per-slot N` (one shared KV pool sized `n_parallel × N`, each slot capped at N), which would allow more slots than the pool could hold at full length simultaneously. It needs build-capability gating before it can become a param.


---

## Storage Locations, Copies and Deletion

**Locations** (`storage.py`): every place models live — NAS and local drives — is a `Location(label, root, description)` from `config.toml` `[[locations]]`, in display order (`LOCATIONS`). There are no priorities and no cache tier: `location_of(path)` (innermost root containing the path) gives a model's location, shown in the model list's Location column (dimmed when `Location.online` is false, i.e. the root is missing — an unmounted NAS). `disk_info()` feeds the free-space hints of every destination picker.

**One entry per copy**: a catalog entry is one model at one path. A copy is another entry with the same `id` (served name), `format` and `family` at the **mirrored path** — the same path relative to its location root (`mirrored_path()`), so every location has the same layout. `AIModel.is_copy_of()` / `launch.copies_of()` find copies; `AIModel.key` = `(path, format)` identifies entries everywhere (catalog persistence via `_patch_catalog(model, …)`, DataTable row keys, preselection after quantize). Settings are a snapshot taken at copy time — afterwards copies are independent.

**Model files** (`model_files()`): a directory model is its directory (copied whole, minus `.git`). A file model is every shard of the GGUF, its companions (`mmproj_path`, `mtp_path`) and the `METADATA_FILES` (config/tokenizer) in its `model_metadata_dir()` — not the rest of its bundle. `model_size_gb()` sums them for the copy picker (computed in a worker; buttons show the catalog `size_gb` until then).

**Copy** (`storage.copy_model()` → `CopyProgressScreen` → `launch.register_copy()`): `rsync -ahR --info=progress2` with each source anchored as `<root>/./<relative path>`, so rsync recreates the relative layout under the destination root in one transfer (re-running copies only what is missing). On success `register_copy()` adds a snapshot of every entry sharing the model's path, relocated with `storage.relocated_entry()` (path and companion paths mirrored). Copying onto the model's own location is refused; locations already holding a copy are shown disabled.

**Delete** (`storage.plan_deletion()` → `_confirm_delete()` → `launch.delete_model()`): the `DeletionPlan` is computed first, shown verbatim in the confirmation (Cancel focused) and then executed as shown. It removes every entry sharing the model's path; for a directory model, the directory. For a file model: its shards plus companions no other entry references; if the model's directory then holds no other weights or subdirectories and no other entry points inside it, the whole directory goes (unless it is a top-level directory of the location). Directories left empty are pruned up to — never including — the location root. Deleting the last copy (no `copies_of()` for an entry in the plan) also requires typing the model name. Deletion is refused while the location is offline, so an unmounted NAS can never make files look absent and drop entries.

**Staging with copies**: `stage_all_models` servers (ComfyUI) stage one entry per `(id, format)` via `_one_copy_each()` — the first copy present on disk in `[[locations]]` order — so copies never collide on the same staged filename.

**Progress display**: `CopyProgressScreen._run_copy()` runs in a `@work(thread=True)` worker. It reads rsync stdout in a `select.select()` loop with a 0.5 s timeout, splits on `\r` and `\n` (rsync uses `\r` for in-place progress updates), and passes the most recent non-empty fragment to `_update_progress_line()`, which parses `(\d+)%\s+([\d.]+\S+/s)\s+([\d:]+)` into the `ProgressBar` and a speed/ETA label. After stdout EOF, `proc.wait()` populates `returncode` before checking success; stderr is surfaced on failure.

---

## Extension Points

**Add a model**: create an `AIModel` entry in `models_local.json`. No code changes needed.

**Add a server type**:
1. `ServerSoftware(...)` entry in `data.py` → `SERVERS` with its `params` (ParamSpecs) (set `env_kind` / `version_packages` or `version_probe`).
2. `_build_<id>(server, env, model, params) → list[str]` in `launch.py`.
3. Register in `_BUILDERS`.
4. If the server needs a staging model directory, add a block in `launch()`.
5. Register an environment for it — `e` on the server list, or `user_data/environments.json`.
6. Optional: set `concurrency_param` if it serves several requests concurrently, and register a `FitProbe` subclass in `fit._PROBES` if it has a cheap way to predict memory use.

**Add an environment**: `e` on the server or environment list → `a`. No code changes needed.

**Add a launch parameter**: add a `ParamSpec` to the server's `params` in `data.py` (default + help; `applies_to` for model-specific ones) and read `p["key"]` in its builder. It appears on the properties page with the right editor automatically. For an "auto" parameter use `AutoBoolParam(resolve=…)` — the resolver gets the model and the merged params (incl. the planned TP/PP) and returns `(value, reason)`.

**Add a user-editable model field**: add the `AIModel` field (and `from_dict()` handling) plus a `ParamSpec` in `models.MODEL_FIELDS`; the properties page and `save_model_field()` handle the rest.

**Add a chat template patch**: append a `ChatTemplatePatch` to `chat_template.TEMPLATE_PATCHES` (pattern, verbatim replacement, `requires` guards). Detection, launch flags, the properties-page toggle and opt-out persistence pick it up automatically. A new server type sets `chat_template_flag` to take part.

**Add an environment kind** (e.g. a venv or container image): subclass `Environment` in `environments.py`, implement `location` / `bin_dir` / `exists()` and whichever lifecycle hooks apply, declare `extra_fields`, and register it in `_ENV_KINDS`.

**Add a storage location**: add a `[[locations]]` entry in `config.toml`. No code changes needed; `ingest` can then recognise copies found there.

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

Every entry is built by `build_entry()`: a path at the same relative position (and format) as a catalogued model on another location is a **copy** — the entry is a snapshot of that model's entry (`storage.relocated_entry()`: same id, family and settings). Anything else is detected from disk and starts a new family (`family` = slug of the detected name).

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
- **GGUF bundle directory** → each `.gguf` file inside yielded individually (one entry per quant variant). Scans flat bundles (`entry/*.gguf`) and per-quant-subdir bundles (`entry/<quant>/*.gguf`). Files named `mmproj-*` (vision projectors) and `mtp-*` (multi-token-prediction draft heads) are companions, not models, and are excluded (`_COMPANION_GGUF_PREFIXES`).
- **llama.cpp split sets** (`<base>-00001-of-000NN.gguf`) → collapsed to the first shard only, which is the file passed to the server; it carries the metadata and pulls in the remaining shards on load. Without this a 7-shard model would become 7 bogus entries each sized as a single shard.
- **HF model directory** (has `config.json` or weight files) → yielded as a whole directory. Directories with both safetensors and pytorch_bin weights yield one entry per format so each becomes a separate catalog entry.

Directories with no `config.json` and no weight files are skipped.

### Detection — GGUF files

- **`quant`**: matched against `GGUFQuant` values by regex on the filename. A filename may contain more than one recognised label (e.g. a Q8_0 quant of a BF16 upcast named `<model>-bf16.Q8_0.gguf`), so the **rightmost** match wins — the trailing tag is the actual quantisation. Ties at the same position are broken by the longer label, so `Q4_K_M` still beats a shorter prefix.
- **`ctx`**: from `config.json` if present in the bundle dir (placed by `copy-configs`); else the GGUF header's `<arch>.context_length`; else borrowed from a sibling entry already in the catalog; `0` if none available.
- **`size_gb`**: from `stat().st_size` of the individual GGUF file; for a split set, the sum of every shard. Missing shards are reported as a warning.
- **`params_m`**: from `config.json` `num_parameters` if present; falls back to regex extraction from filename (e.g. `70B` → 70000).
- **`arch`**, **MoE**, **`vision`**, **`trust_remote_code`**: from `config.json` when present. Without it, MoE counts come from the GGUF header (`<arch>.expert_count` / `expert_used_count`), `vision` is inferred from an `mmproj-*.gguf` in the bundle dir, and `arch` stays empty (the catalog records HF class names, which GGUF lacks).
- **`mmproj_path`**: auto-detected from `mmproj-*.gguf` in bundle root when `vision=True`.
- **`mtp_path`**: auto-detected from `mtp-*.gguf` (multi-token-prediction draft head) in the bundle dir or the bundle root above it; when several are present, the one whose quant label matches the model's wins.
- **`mtp_embedded`**: from the GGUF header (`launcher/gguf_header.py`, a dependency-free reader of the key/value metadata; ~0.1 s per file even over the NAS).
- **`name`**: derived from the parent dir name when it contains "gguf", else the file stem. The split-shard suffix is stripped, and the quant label and `GGUF` tag are removed wherever they occur as whole tokens, so `Model-GGUF-Q5_K_S/…` yields `Model` and a trailing variant tag survives (`Model-GGUF-Q8_0-MTP` → `Model MTP`) rather than carrying tags that the quant slug would then duplicate. Unsloth's `UD` (Unsloth Dynamic) tag, which prefixes the quant label (`Model-GGUF-UD-Q6_K_XL`), is stripped along with the other provenance suffixes.
- **`id`**: `<name-slug>-<quant-slug>` where quant slug drops underscores and trailing `0` (e.g. `Q8_0` → `q8`, `Q4_K_M` → `q4km`). Collision guard appends `-gguf` if the ID already belongs to a different path.

### Detection — HF model directories

Reads `config.json`:

- **`format`**: `quantization_config.quant_type` first (gptq / awq / exl2 / int8_bnb), then `quant_method` (fp8, bitsandbytes, modelopt), then `torch_dtype` for bf16/fp16/fp8, then file-type presence for pytorch_bin. Legacy GPTQ files detected by `4bit`/`8bit`/`gptq` patterns in safetensors filenames.
- **`ctx`**: first of `max_position_embeddings`, `max_seq_len`, `seq_length`, `context_length`, `n_positions`; `model_max_length` preferred when strictly larger (covers RoPE-extended models). Falls back to `text_config` for multimodal models.
- **`params_m`**: from `num_parameters` in `config.json`, or regex extraction from directory name.
- **`arch`**: from `architectures[0]` in `config.json`.
- **MoE fields**: `moe_experts` / `moe_active` from `num_local_experts` / `n_routed_experts` / `num_experts` and `num_experts_per_tok`.
- **`vision`**: `True` when `vision_config` is present.
- **`mtp_embedded`**: `True` when `mtp_num_hidden_layers` / `num_nextn_predict_layers` (top level or `text_config`) is > 0 **and** the weights include the MTP layers — names from `model.safetensors.index.json` (else the shard headers): an `mtp.*` module or a `layers.<num_hidden_layers>.*` NextN layer.
- **`trust_remote_code`**: `True` when `auto_map` is present (custom architecture code).
- **`min_sm`**: set to `89` for static FP8 quantisation (non-dequantise, non-block-wise); `0` for block-wise FP8 (dequantises to BF16 at runtime, works on sm_86).
- **`size_gb`**: recursive `stat()` sum of weight files matching the format's glob patterns (format-aware — e.g. pytorch_bin counts only HF shard files, not unrelated `.bin` artefacts). Mistral repos that ship the same weights twice (HF `model-*.safetensors` alongside native `consolidated-*.safetensors`) count only the HF shards, so the size is not doubled. The two sets are recognised by their index files declaring the same total within 1% — the layouts pack tensors differently, so the totals agree to parts-per-million rather than exactly. Deduplication is skipped (with a warning) when any HF shard is missing from disk, so a partially-downloaded set is never used as the basis for the size.
- **`id`**: `<name-slug>-<fmt-short>` (e.g. `…-bf16`, `…-fp16`, `…-int8`). Collision guard falls back to the raw directory name.

### Detection — image_gen models

With `--ai-type image_gen`, each `.safetensors` file in the scan directory becomes its own catalog entry:

- **`format`**: inferred from filename — `fp8`/`e4m3fn` → `safetensors_fp8`; `flux` → `safetensors_bf16`; everything else → `safetensors_fp16`.
- **`comfyui_subdir`**: `"diffusion_models"` for FLUX/diffusion/unet; `"checkpoints"` for standard checkpoints (SDXL, SD1.5).
- **`ai_types`**: `["image_gen"]`.

### refresh subcommand

`refresh` re-detects auto-derived fields (`ctx`, `size_gb`, `params_m`, `arch`, `moe_experts`, `moe_active`, `vision`, `mmproj_path`, `mtp_path`, `mtp_embedded`) for existing catalog entries. User-curated fields (`id`, `name`, `format`, `path`, `family`, `ctx_override`, `favorite`, `server_status`, `server_params`, `min_sm`, `hw_note`, `comfyui_subdir`, `quant`, `trust_remote_code`, `note`) are never touched.

- `--path <path>` limits refresh to a single entry; otherwise all entries are refreshed.
- Size recompute uses the stored (user-curated) format, not the freshly-detected one — if a format has been manually corrected, the size reflects the corrected format's weight files.
- Zero-valued detections for `ctx` and `params_m` are treated as "unknown" and preserve the existing value rather than overwriting manually-set data.

### copy-configs / copy-mmproj

Preparation steps for GGUF bundles — run before `ingest` or `refresh` so that metadata detection has access to `config.json` and mmproj files.

Both commands search `scan_root` plus the same leaf directory under every other storage location (e.g. if `scan_root` is `/mnt/nas/ai_models/text-generation`, also checks `~/fast_models/text-generation` from the configured locations). For each GGUF bundle directory that lacks the target files, the original (non-GGUF) model is located via normalised name matching and the files are copied across.

The copied set (`_CONFIG_FILES`) covers `config.json`, the tokenizer files, `generation_config.json`, and `chat_template.jinja` / `chat_template.json` — transformers ≥ 4.55 ships the chat template as a standalone file instead of embedding it in `tokenizer_config.json`.

`copy-configs` skips any bundle that already has a `config.json`, so bundles processed before a new filename was added to `_CONFIG_FILES` are not backfilled automatically; call `copy_model_metadata(src, dst)` directly for those.

Fields left for manual review: `ctx` for new GGUF bundles without config.json, `server_params`.

---

## Catalog Audit Notes

When auditing the model catalog for misclassifications or stale metadata, append findings to `user_data/model_audit.md` (gitignored — personal/local). Use the FIXED / NOTED / OPEN ITEMS structure. Key things to check:

- `torch_dtype` in each model's `config.json` — `float32` (BF16 weights stored as BF16 despite name), `float16` → `SAFETENSORS_FP16`, `float8_e4m3fn` → `SAFETENSORS_FP8`
- Actual disk size (`du -sh`) vs. `size_gb` in `models_local.json`
- `max_position_embeddings` in `config.json` vs. `ctx` in catalog
- Number and type of weight files (`.safetensors`, `.bin`, `.gguf`) to confirm declared format
