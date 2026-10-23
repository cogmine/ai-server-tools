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
        data.py                     SERVERS list + QUANTIZATION_TOOLS + catalog loading (config-driven, safe to commit)
        storage.py                  Storage locations ([[locations]]), mirrored paths, model copy + deletion plans
        environments.py             Server environments (conda envs / native builds) + ENVIRONMENTS registry
        environments.json.example   Tracked template — copy to user_data/environments.json
        gpu.py                      NVML poller (pynvml) + GpuRow data + formatting helpers
        gpu_power.py                Per-GPU power limits (remembered per card UUID) + launch-command step
        scripts/gpu-power-limit     Applies power limits via sudo nvidia-smi -pl, only where they differ
        ingest.py                   CLI tool — scans a directory for new models and appends entries to models_local.json
        gguf_header.py              Dependency-free GGUF header (key/value metadata) reader used by ingest
        launch.py                   Command builders, process spawner, catalog persistence (copy/delete bookkeeping)
        fit.py                      Pre-launch memory-fit estimates (FitProbe per server; llama.cpp: llama-fit-params)
        models.py                   Schema — ModelFormat enum, AIModel/ServerSoftware dataclasses
        models_local.json.example   Tracked template — schema reference + 3 sample entries
        screens.py                  All Textual screens (AITypeScreen → ModelSelectScreen → …)
        model_launcher.tcss         Textual CSS layout
        DESIGN.md                   Architecture and design decisions
        user_data/                  Gitignored — all machine-specific and personal data
            config.toml             Machine-specific paths, GPU counts, host RAM budget ([memory])
            environments.json       Registered environments per server (maintained by the TUI)
            gpu_power.json          Chosen GPU power limits keyed by GPU UUID (written by GpuSelectScreen)
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

## llama.cpp

Three native environments (registered in the launcher, `kind: native`):
- **`llama.cpp`** (default) — `~/llama.cpp`, build b10109 (2026-07-24), `~/llama.cpp/build/bin`.
- **`llama.cpp-next`** — `~/llama.cpp-next`, a **git worktree** of the same repo at the latest
  `origin/master` (detached), built into `~/llama.cpp-next/build/bin` (b11461, 2026-10-07; includes
  `qwen4exp` / Qwen3.8-Flash-Next support). Binaries use RUNPATH to their own `build/bin`, so the
  builds never share libraries.
- **`llama.cpp-next-patched`** — `~/llama.cpp-next-patched`, a third worktree: `origin/master` b11706
  (2026-10-16) plus one **local commit** (never pushed) applying the fix from ggml-org/llama.cpp#29811
  (`src/models/qwen4exp.cpp`: the MTP draft graph builds k-pool inputs only when the MTP block's
  `attention.compress_ratios` entry is > 0). Without it, Qwen4Exp MTP heads converted before llama.cpp's
  merged MTP support (ratio 0 for the MTP block — e.g. orcarouter's embedded `-Q8_0-MTP` GGUF) abort at
  startup with `GGML_ASSERT(buffer)` in `set_input_k_idxs`. b11706 also fixes MTP + image input (#30257,
  #30397), which crashes b11461. Validated: `qwen3-8-flash-next-uncensored-mtp-q8`, 3 × 256K, text + image,
  ~60 t/s, draft acceptance 54–67%. **Update** (re-applies the patch; drop it once merged upstream):
  `cd ~/llama.cpp-next-patched && git fetch origin && git rebase origin/master && /usr/bin/cmake --build build -j 32`.
- **Build config** (identical for both): `/usr/bin/cmake` (the `~/.local/bin/cmake` pip shim is broken),
  `-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.5/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_NATIVE=ON
  -DGGML_CUDA_FA=ON -DGGML_CUDA_GRAPHS=ON -DGGML_CUDA_NCCL=ON -DGGML_CUDA_COMPRESSION_MODE=size`.
  nvcc 12.5 compiles (12.1's nvcc rejects GCC 13); CMake links the CUDA 12.1 runtime, matching the
  system `LD_LIBRARY_PATH`.
- **Update `llama.cpp-next`**: `cd ~/llama.cpp-next && git fetch origin && git checkout --detach origin/master
  && /usr/bin/cmake --build build -j 32` (~10 min). The launcher shows each build's version
  (`llama-server --version`, via `VersionProbe`); promote it with `s` (set default) in the
  environment manager once validated.
- **Launch flags** (`_build_llamacpp`, see launcher `DESIGN.md` → "llama.cpp Launch Flags"): `parallel`
  slots (set per launch with `+`/`-` on the GPU screen, remembered per model) → `--parallel N
  --ctx-size effective_ctx × N` (fixed slots split `--ctx-size` evenly, so catalog `ctx` stays per-request);
  `kv_cache_type` default `q8_0` (`--cache-type-k/-v`; lets BeMag 123B Q8 fit 5 × 128K slots — ≈117 GiB KV
  vs ≈220 GiB at f16); `cache_ram_mib` from `[memory] host_cache_ram` in `config.toml` (`--cache-ram`,
  `"60%"` ≈ 150 GiB here; llama.cpp's 8 GiB default holds < one full 128K slot state); `--threads` omitted
  (llama.cpp uses the 16 physical cores).
- **Memory-fit estimate**: `launcher/fit.py` runs `llama-fit-params` (same build, same model/context args via
  `llamacpp_model_args()`) in a background worker on the GPU screen — exact for any architecture, measured against
  currently free VRAM. The verdict comes from the streamed log within ~2 s; the fitter's placement search (minutes for
  big MoE models at long ctx) only refines it and may time out. Excludes `--mmproj`, MTP heads and `extra_flags`.
  `ubatch` param (`--ubatch-size`) shrinks compute buffers, which dominate at YaRN 1M contexts.

---

## tabbyAPI / ExLlamaV2

- **Conda env**: `tabby` (`~/miniconda3/envs/tabby/`)
- **Entry point**: `~/tabbyAPI/main.py`
- **Key advantage**: no attention-head divisibility constraint — uses all GPUs via `--gpu-split-auto`
- **Formats**: EXL2, safetensors BF16/FP16, and legacy `.bin` — **GGUF is NOT supported** (use llama.cpp for GGUF; tabbyAPI's sister project YALS handles GGUF)
- **flash-attn**: 2.8.3 — built from source in `tabby` env with `CUDA_HOME=/usr/local/cuda-12.5`; cached at `~/.cache/pip/wheels/`
- **KV cache**: `--cache-mode Q8` (quantized KV, fast on Ampere)
- **Multi-GPU**: ExLlamaV2 row/column tensor split, not restricted by head count
- **Model directory approach**: tabbyAPI requires models to be subdirectories inside a `--model-dir`.
  At launch time `launch.py` automatically creates a symlink `~/tabbyAPI/models/<model.id>` → `model.path`
  and passes `~/tabbyAPI/models/` as `--model-dir`. No pre-created symlinks or stub dirs are needed.
  The same pattern applies to any future tool that requires a models directory — add a block in `launch.py`.

---

## text-generation-webui (ooba)

- **Conda env**: `textgen` (`~/miniconda3/envs/textgen/`)
- **Entry point**: `~/text-generation-webui/server.py`
- **Staging symlink**: `launch.py` creates `~/text-generation-webui/user_data/models/<model.id>` → resolved path before spawning. GGUF files get a `.gguf` suffix in the link name so ooba's scanner identifies the format automatically. Split GGUFs are staged as a directory `<model.id>/` of per-shard links keeping their original names (llama.cpp finds shards 2..N by name), and launched with `--model <model.id>/<shard-1 filename>`.
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

### Version / CUDA toolchain (cu13 split wheels)
vLLM **0.26.0** with PyTorch **2.11.0+cu130** in the `vllm` conda env (0.26.0 keeps the same torch
as 0.25.1, so the cu13 toolchain is unchanged). The cu13 CUDA runtime
is bundled as pip split wheels under `…/site-packages/nvidia/cu13/` (shared libs in `lib/`, nvcc
in `bin/`, headers in `include/`). This is env-local — other envs stay on cu128.

flashinfer JIT-compiles CUDA kernels (e.g. the sampler) at runtime, which requires a *consistent*
bundled toolkit. Two constraints must hold, both wired into `launcher/data.py` / `launch.py`:
- **Toolchain version pinning** — `nvidia-cuda-nvcc`, `nvidia-nvvm`, and `nvidia-cuda-crt` must all
  match the bundled cudart header version (**13.0.88**, matching `nvidia-cuda-runtime` 13.0.x).
  A plain `pip install vllm` pulls nvcc/nvvm 13.2+, which mismatches the 13.0 cudart headers
  (libcudacxx "CUDA compiler and CUDA toolkit headers are incompatible") and generates PTX 9.2 that
  the 13.0 `ptxas` rejects ("Unsupported .version 9.2"). **After any vLLM reinstall/upgrade, re-pin:**
  `pip install "nvidia-cuda-nvcc==13.0.88" "nvidia-nvvm==13.0.88" "nvidia-cuda-crt==13.0.88"`.
- **Link-time cudart** — the cu13 wheel ships `libcudart.so.13` but omits the `libcudart.so` linker
  name and uses `lib/` (not `lib64/`). `launch.py` self-heals the `libcudart.so` symlink and sets
  `LIBRARY_PATH`; `-lcuda` resolves against the system driver stub. All auto-applied at launch, per
  environment, via the vLLM `ServerSoftware` fields `bundled_cuda_glob` (`nvidia/cu1[0-9]*`, found
  in the chosen env's site-packages → `CUDA_HOME` + `lib/` on `LD_LIBRARY_PATH` / `LIBRARY_PATH`)
  and `lib_symlinks` — so any additional vLLM environment (e.g. `vllm-next`) works the same way.
- **flashinfer-cubin version lock** — vLLM 0.26.0 pins `flashinfer-python==0.6.14` and hard-checks
  that any installed `flashinfer-cubin` matches exactly (else `RuntimeError: flashinfer-cubin
  version does not match flashinfer version`). When the cubin package lags behind (0.6.14 unpublished
  as of this writing, max 0.6.13), **uninstall `flashinfer-cubin`** so flashinfer JIT-compiles the
  0.6.14 kernels itself (needs `CUDA_HOME` + the link-time cudart fix above). Re-add the matching
  `flashinfer-cubin` once its version catches up to skip JIT.

### `vllm-next` environment
A second vLLM conda env registered in the launcher (cloned from `vllm`) for releases that add new
architectures — currently vLLM **0.31.0**, torch **2.13.0+cu130**, flashinfer-python **0.7.0.post1**
(no matching `flashinfer-cubin` → JIT), needed for `Qwen4Exp` (Qwen3.8-Flash-Next). The same
cu13 rules apply: the 13.0.88 nvcc/nvvm/crt pins survived the upgrade (re-check after any upgrade),
and the launcher wires the env's own bundled toolkit. vLLM 0.31 itself pins `torchaudio==2.11.0`
alongside torch 2.13 (no newer torchaudio exists). flashinfer warns that `nvidia-cuda-tileiras`
13.2.78 mismatches nvcc and falls back to the system tileiras — harmless (no 13.0 build exists;
cuTile is unused on sm_86; same in the `vllm` env). Smoke-tested end-to-end (serve + completion).
**Local patch (re-apply after any vLLM reinstall/upgrade until merged upstream)**: vllm-project/vllm#56362
— the Qwen4Exp pinned-host PLE lookup kernel passes FP8 pointers to Triton, which rejects `fp8e4nv` on
sm_8x ("type fp8e4nv not supported in this architecture"); the patch passes uint8 views (byte-identical
copy). Apply from `site-packages/`:
`curl -sfH "Accept: application/vnd.github.v3.diff" https://api.github.com/repos/vllm-project/vllm/pulls/56362 | patch -p1`.
Qwen3.8-Flash-Next-FP8 catalog params: TP8 + `expert_parallel`, `gpu_mem_util` 0.95, `max_num_seqs` 16,
`auto_fit_ctx` (at 0.90 / 256 seqs only ~52K ctx fit). Do not use `--kv-cache-dtype fp8` on Ampere — the
QSA kernels would hit the same Triton fp8e4nv limitation. Loading from the NAS takes ~21 min.

### Parallelism
- Executor backend is **not** forced by default — vLLM auto-selects `mp` (multiprocessing), which is simpler/more robust for this single node. Set `executor_backend` in `server_params["vllm"]` to `"ray"` only when a model needs it.
- PCIe topology: pipeline parallelism preferred over tensor parallelism (activations only cross between adjacent GPU pairs)
- **Attention-head constraint**: `--tensor-parallel-size` must evenly divide the model's attention head count (`num_attention_heads` in `config.json`, or under `text_config` for multimodal checkpoints) and any linear-attention head counts. Use pipeline parallelism or tabbyAPI/ExLlamaV2 when TP=11 is not viable.
- **No-PP architectures**: Qwen4Exp (Qwen3.8-Flash-Next) rejects PP > 1 (N-gram PLE embedding); `compute_parallelism` uses pure TP for archs in `_NO_PP_ARCHS` → TP=8 on 8 GPUs. Its block-FP8 experts (640 wide) don't split at TP8, so the catalog sets `expert_parallel: true` (`--enable-expert-parallel`). vLLM 0.31 offloads the 48 GiB PLE table to pinned host RAM by default (`--engram-config`, `cpu_offload=True`).

### Launch flags
`_build_vllm` is minimal by default and data-driven via `default_params` (see launcher `DESIGN.md` → "vLLM launch flags"). Only always-on flags are `--model`, `--served-model-name`, `--tensor-parallel-size`, `--pipeline-parallel-size`, `--gpu-memory-utilization` (default **0.90**), `--max-model-len`, `--host`, `--port`. The rest are opt-in:
- `--dtype` — omitted by default (`dtype="auto"` honours the model's `torch_dtype`); set per-model only when a checkpoint's native dtype must be overridden.
- `--enforce-eager` — off by default (CUDA graphs enabled = faster); enable per-model (`enforce_eager: true`) only if graph capture OOMs.
- `--enable-chunked-prefill` — off by default (vLLM enables it itself when helpful); enable per-model (`chunked_prefill: true`) if needed.
- `--enable-expert-parallel` — off by default; enable per-model (`expert_parallel: true`) for MoE checkpoints whose experts can't be TP-sharded (FP8 block alignment).
- `--max-model-len -1` — emitted instead of the model's ctx when `auto_fit_ctx: true`; vLLM auto-fits the context to the available KV-cache memory (check the log for "Auto-fit max_model_len").
- `--max-num-seqs` — omitted by default (vLLM default 256); set per-model to cap concurrency for tight-memory models.
- `--trust-remote-code` — emitted only when the model's `trust_remote_code` field is `true` (check `auto_map` in `config.json`).
- `--enable-auto-tool-choice --tool-call-parser <name>` — emitted when the model's `tool_call_parser` field is set.
- `cd /tmp` before launching — prevents `~/vllm/` source tree from shadowing the installed package
- **Port**: all scripts default to **5000**

---

## Textual Launcher API (`launcher/`)

**Entry**: `cd ~/ai-server-tools && ~/miniconda3/envs/ai-tools/bin/python -m launcher`

**Model config**: edit `launcher/user_data/models_local.json` — one JSON object per model in the `"models"` array.
See `launcher/models_local.json.example` for the full field schema.
Schema types live in `launcher/models.py` (`ModelFormat` enum, `AIModel.from_dict()` deserialiser).
Server definitions (SERVERS list) live in `launcher/data.py` (safe to commit). Storage locations (`LOCATIONS`, from `[[locations]]` in `config.toml` — NAS and local drives alike) live in `launcher/storage.py`.
**Environments**: each server launches from a registered `Environment` (`launcher/environments.py`): `CondaEnvironment` (`<conda_base>/envs/<name>`) or `NativeEnvironment` (`bin_dir`, llama.cpp). Registry `ENVIRONMENTS` ↔ `launcher/user_data/environments.json` (`{"environments": {server_id: [{kind, name, default?, note?, …}]}}`), maintained by `EnvManageScreen` (`e` on the server list / env picker: add — offering conda create/clone when missing — create, rename via `conda rename`, remove with optional `conda env remove`, set default, note). Builders take `env` and use `env.python` / `env.executable(...)`; quantization tools run from their `env_server_id`'s default environment. Several envs per server allow testing upgrades (e.g. `vllm-next`) without breaking the known-good one.
`path` field: absolute path to the model file or directory, inside one of the configured storage locations; one path per entry. The model list shows its location in the Location column (before Name; dim = location offline).
`family` field: shared by an original model and every copy, conversion and quantization of it (ingest assigns the slug of the name; copy/convert inherit it). Copies of a model at other locations are separate entries with the same `id` (served name) at the mirrored path (same path relative to the location root); the catalog key is `AIModel.key` = `(path, format)` — `_patch_catalog(model, …)` and all `save_*()` functions match on it, never on `id`. There is no cache concept: `c` copies a model to another location (`storage.copy_model()` + `launch.register_copy()`, a settings snapshot), `Del` deletes it (`storage.plan_deletion()` shown for confirmation, typed name for the last copy, then `launch.delete_model()`).
Status field: `"good"` (green), `"bad"` (red), `"unknown"` (white) — persisted per `(model, server, environment, quant)` via `save_server_status()` in `launch.py` (JSON roundtrip, no regex). `server_status` in JSON is a nested dict: `{server_id: {env_name: {quant_key: status}}}` where `quant_key` is `"none"` (full/stored precision) or a `RuntimeQuant.value` (e.g. `"bnb_int8"`). Multiple quant keys may be `"good"`. `status_for(server_id, env_name, quant)`; `env_best_status(server_id, env_name)` = best across quants; `best_status_for(server_id)` = best across envs and quants. Env rename/removal migrates/drops results (`apply_env_rename()` / `apply_env_removal()`). `_quant_key(quant)` maps `RuntimeQuant | None` → string key.
`ctx_override` field: optional launch override; `0` = use native `ctx` (scaled by `rope_scaling`). Can be higher or lower than `ctx`. Set interactively with `l` on the model list (`ContextSettingsScreen`). Persisted via `save_ctx_override()` — same JSON roundtrip pattern as `save_server_status()`.
`rope_scaling` field: optional `RopeScaling` (`{"factor": 4.0}`, optional `method` default `yarn`, `original_ctx` default `ctx`); default launch ctx becomes `ctx × factor`. Builders translate it per server when `server.applies_rope_scaling(model)` (llama.cpp `--rope-scaling/--rope-scale/--yarn-orig-ctx`; ooba GGUF via `--extra-flags`; vLLM `--hf-overrides` / SGLang `--json-model-override-args` built from the model's own config.json RoPE block, plus `rope_scaling_env`); other servers get the native ctx via `server.launch_ctx(model)`. Set with `l` (`ContextSettingsScreen`), persisted via `save_rope_scaling()`.
`favorite` field: boolean; `True` pins the model to the top of the sorted list. Toggled with `f` in `ModelSelectScreen`. Persisted via `save_favorite()` — key omitted from JSON when `False`. The `★` column shows `★` for favorites.
`min_sm` field: integer minimum CUDA compute capability (e.g. `89` for sm_89+); `0` = no constraint. Checked against `system_min_sm()` from `gpu.py` (cached). When `model.hw_compat(system_sm)` is `False`, the warn column shows `⛔` and the note bar below the model table shows `model.hw_note`.
`hw_note` field: string explaining the hardware requirement — shown in the launcher note bar when incompatible. Omitted from JSON when empty.
`comfyui_subdir` field: `image_gen` models only — ComfyUI `models/` subdirectory to stage into (e.g. `"checkpoints"` for SDXL, `"diffusion_models"` for FLUX UNets, `"vae"`, `"loras"`). Default `"checkpoints"` — key omitted from JSON for checkpoints. Staging uses the original source filename (`staging_use_source_filename=True` on ComfyUI's `ServerSoftware`) so ComfyUI workflows can reference models by their natural names (e.g. `flux1-dev.safetensors`, `ae.safetensors`).
`mmproj_path` field: absolute path to the companion `mmproj-*.gguf` vision encoder file; omitted from JSON when empty. Auto-detected by `ingest`/`refresh` when found in the GGUF bundle root (flat layout) or bundle root one level above the quant subdir. Populated via `copy-mmproj` + `refresh` workflow. Passed as `--mmproj` to llama.cpp and ooba at launch time; copying a model copies the mmproj along with the GGUF (mirrored path).
`mtp_path` field: absolute path to a companion `mtp-*.gguf` multi-token-prediction draft head (auto-detected by `ingest`/`refresh`; quant-matching file preferred). llama.cpp and ooba launch with `--spec-type draft-mtp --model-draft <path>` unless server param `mtp` is `false`; copied alongside the model like the mmproj.
`mtp_embedded` field: `true` when the GGUF itself has MTP/NextN layers (`<arch>.nextn_predict_layers`, read by `launcher/gguf_header.py`). Launched with `--spec-type draft-mtp` (no draft file) only when server param `mtp` is `true` — default `None` uses sidecar heads only, because llama.cpp may not implement MTP for the architecture (GLM5-Next does not yet).
`tool_call_parser` field: vLLM tool-call parser name (e.g. `"openai"`, `"hermes"`, `"llama3_json"`). When set, vLLM emits `--enable-auto-tool-choice --tool-call-parser <name>` and SGLang emits `--tool-call-parser <translated_name>` (via `_SGLANG_TOOL_PARSER_MAP` in `launch.py`). Omitted from JSON when empty.
Model list sort is multi-column: `_sort_keys` list of `(col, reverse)` tuples. Default order: `fav → status → name`. Clicking a header promotes it to primary (ascending); clicking the primary toggles direction. `_STATUS_RANK` maps composite status symbols (✓, ✓?, ✓✗, ✗?, ✗, ?) to integers 0–5 for sort.

**Log files**: `logs/{timestamp}_{server}_{model}.log`. Header contains metadata + a paste-ready command block (no `#` prefix on the command lines). Exit footer: `# exit: code N` or `# exit: killed by SIGSEGV (11)`. Captured via PTY so ANSI colours are preserved. SIGWINCH is forwarded to the PTY so progress bars resize live.

**GPU widget**: `gpu.py` contains `GpuRow`, `poll_gpus()`, and formatting helpers shared by both apps.
The launcher does **not** embed GPU stats — run `gpu_monitor` separately while a model is loaded.

**Flow** (text-gen): AITypeScreen → ModelSelectScreen → ServerSelectScreen → [EnvSelectScreen] → [QuantSelectScreen] → GpuSelectScreen → ConfirmLaunch → exits TUI with a `LaunchJob(factory, model, server, env)` → `Popen` → `proc.wait()`. `EnvSelectScreen` only appears when the server has >1 registered environment (default pre-focused); `_continue_launch()` refuses environments missing on disk.
`GpuSelectScreen`: shows all GPUs with VRAM utilization; multi-select (Space) for multi-GPU servers, radio-button for `max_gpus=1` servers. `→`/`Enter` to confirm. `a` selects all. Status bar shows live `CUDA_VISIBLE_DEVICES` string and auto-adjusted `gpu_count_param` (e.g. `pipeline_parallel → N` for vLLM/SGLang). If NVML unavailable, skips to ConfirmLaunch with default devices. Selected devices → `cuda_devices` arg to `launch()`; GPU count → `param_overrides[gpu_count_param]` → `build_cmd()` last-layer merge. For servers with `concurrency_param` (llama.cpp `parallel`), `+`/`-` set the slot count (→ `param_overrides`; persisted per model on launch via `save_server_param()`), and when `fit.py` has a probe for the server, a debounced background worker shows a memory-fit verdict for the current selection. `p`/`P` set the power limit (watts) for the highlighted/selected GPUs — remembered per card UUID in `user_data/gpu_power.json`, applied by a leading `launcher/scripts/gpu-power-limit IDX,…=W|default … &&` step in the launch command (sudo only when a limit differs; included when any launched card has a chosen or non-default limit).
`g`/`x`/`?` marking happens on the deepest screen of the launch: `ServerSelectScreen` only for single-env servers without runtime quants for the model; `EnvSelectScreen` for multi-env servers without quants; `QuantSelectScreen` (per env + quant option) otherwise. Shallower screens disable the bindings via `check_action`.
Path validation before GpuSelectScreen; missing files show a toast notification.

**Flow** (image-gen / hub): AITypeScreen → ImageGenHubScreen → "Launch Server" → ServerSelectScreen(model=None) → [EnvSelectScreen] → GpuSelectScreen → ConfirmLaunch → exits TUI.
"Manage Models" → ModelManageScreen → `→`/`Enter` → CopyModelScreen → CopyProgressScreen (rsync with live ProgressBar).
`uses_hub_flow(servers, ai_type)` returns `True` when every server for that AI type has `stage_all_models=True`.
`ModelManageScreen` overrides `_open_model()` (template method in `ModelSelectScreen`) to push `CopyModelScreen`. The template method pattern is necessary because Textual calls all MRO classes' message handlers regardless of override — only plain method calls respect the most-derived class.
`CopyProgressScreen`: `rsync -ahR --info=progress2` (sources anchored `<root>/./<rel>` so the layout is mirrored); stdout read via `select.select()` in a `@work(thread=True)` worker; `(\d+)%` regex drives a `ProgressBar` widget + speed/ETA detail label; `proc.wait()` called after stdout EOF before reading `returncode`; on success `launch.register_copy()` adds the copy's catalog entries.

**Catalog ingest**: `python -m launcher.ingest <subcommand>` — four subcommands:
- `ingest <scan_root> [--all] [--ai-type text_gen|image_gen] [--dry-run]` — scans a directory for models not in the catalog and appends entries. `--path <path>` targets a specific model. `--ai-type image_gen` for diffusion checkpoints/FLUX.
- `refresh [--path <path>] [--dry-run]` — re-detects auto-derived fields (ctx, size_gb, params_m, arch, moe, vision, mmproj_path) for existing entries without touching user-curated fields.
- `copy-configs <scan_root> [--dry-run]` — copies HF config/tokenizer files into GGUF bundle directories before ingesting.
- `copy-mmproj <scan_root> [--dry-run]` — copies mmproj vision encoder files into GGUF bundle directories before ingesting/refreshing.

**Catalog anomaly log**: `user_data/model_audit.md` — records format misclassifications, size corrections,
and open items found during catalog audits. Gitignored (personal/local).

**Dependencies**: `textual 8.1.1` installed in the dedicated `launcher` conda env (defined in `environment.yml` at the repo root).

