"""
launch.py — Unified command builder and process launcher.

build_cmd(server, model) → list[str]
    Merges server.default_params with model.server_params[server.id],
    then dispatches to the server-specific builder.

launch(server, model) → tuple[subprocess.Popen[bytes], Path, int]
    Builds the command, sets up the environment, spawns the process, and
    returns the Popen handle, the session log path, and the PTY master fd.
    stdout/stderr are connected to a PTY slave so the child process sees a
    real terminal (preserving ANSI colours). The caller owns master_fd and
    must close it after the tee thread finishes.

Adding a new server type:
  1. Write a _build_<name>(server, model, params) → list[str] function.
  2. Register it in _BUILDERS.
  3. Set any launch-time fields (strip_pythonpath, cuda_home, uses_staging_dir,
     staging_format_suffixes, launch_cwd) on the ServerSoftware entry in data.py.
"""
from __future__ import annotations

import fcntl
import json
import os
import pty
import shlex
import shutil
import struct
import subprocess
import termios
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from launcher.data import AI_MODELS, CACHE_MOUNTS
from launcher.models import (
    AIModel, CacheMount, GGUF_QUANTS_LONGEST_FIRST, ModelFormat,
    QuantizationTool, RuntimeQuant, ServerSoftware, _quant_key,
)
from launcher.config import CONFIG


_LOGS_DIR = Path(__file__).parent.parent / "logs"
_CATALOG_PATH = Path(__file__).parent / "user_data" / "models_local.json"

# Binary paths are resolved from CONFIG (launcher/user_data/config.toml) at import time.


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _merged(server: ServerSoftware, model: "AIModel | None", param_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge server defaults with the model's per-server overrides, then param_overrides."""
    p = dict(server.default_params)
    if model is not None:
        p.update(model.server_params.get(server.id, {}))
    if param_overrides:
        p.update(param_overrides)
    return p


def _patch_catalog(model_id: str, apply: "Callable[[dict], None]") -> None:
    """Load models_local.json, find the entry for model_id, call apply(entry), then save.

    Raises ValueError if model_id is not found in the catalog.
    """
    with open(_CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)
    for entry in catalog["models"]:
        if entry["id"] == model_id:
            apply(entry)
            break
    else:
        raise ValueError(f"Could not find model id={model_id!r} in models_local.json")
    with open(_CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
        f.write("\n")


# ---------------------------------------------------------------------------
# Per-server command builders
# ---------------------------------------------------------------------------

def _build_vllm(server: ServerSoftware, model: AIModel, p: dict) -> list[str]:
    cmd = [
        str(CONFIG.vllm_python), "-m", "vllm.entrypoints.openai.api_server",
        "--model",                        p["_model_path"],
        "--served-model-name",            model.id,
        "--tensor-parallel-size",         str(p["tensor_parallel"]),
        "--pipeline-parallel-size",       str(p["pipeline_parallel"]),
        "--distributed-executor-backend", p["executor_backend"],
        "--dtype",                        "bfloat16",
        "--gpu-memory-utilization",       str(p["gpu_mem_util"]),
        "--max-model-len",                str(model.effective_ctx),
        "--enable-chunked-prefill",
        "--max-num-seqs",                 str(p["max_num_seqs"]),
        "--enforce-eager",
        "--host",                         "0.0.0.0",
        "--port",                         str(p["port"]),
        "--trust-remote-code",
        "--no-enable-log-requests",
    ]
    rq = p.get("_runtime_quant")
    if rq == RuntimeQuant.BNB_INT8:
        cmd += ["--quantization", "bitsandbytes", "--load-format", "bitsandbytes"]
    cmd.extend(p.get("extra_flags", []))
    return cmd


def _build_llamacpp(server: ServerSoftware, model: AIModel, p: dict) -> list[str]:
    cmd = [
        str(CONFIG.llama_server),
        "--model",        p["_model_path"],
        "--alias",        model.id,
        "--n-gpu-layers", str(p["n_gpu_layers"]),
        "--split-mode",   p["split_mode"],
        "--ctx-size",     str(model.effective_ctx),
        "--parallel",     str(p["parallel"]),
        "--threads",      str(p["threads"]),
        "--host",         "0.0.0.0",
        "--port",         str(p["port"]),
        "--verbose-prompt",
        "--log-prefix",
        "--log-timestamps",
    ]
    cmd.extend(p.get("extra_flags", []))
    return cmd


def _build_tabby(server: ServerSoftware, model: AIModel, p: dict) -> list[str]:
    # model_dir is a staging directory; launch() creates a symlink named model.id
    # there pointing to the resolved (cached or NAS) path before this builder runs.
    cmd = [
        str(CONFIG.tabby_python), str(CONFIG.tabby_main),
        "--model-dir",    p["model_dir"],
        "--model-name",   model.id,
        "--cache-mode",   p["cache_mode"],
        "--max-seq-len",  str(model.effective_ctx),
        "--host",         "0.0.0.0",
        "--port",         str(p["port"]),
        "--disable-auth", str(p["disable_auth"]),
    ]
    cmd.extend(p.get("extra_flags", []))
    return cmd


def _build_ooba(server: ServerSoftware, model: AIModel, p: dict) -> list[str]:
    # For GGUF files, ooba's scanner identifies them by the .gguf suffix; the symlink
    # created in launch() is named accordingly (model.id + ".gguf").
    model_name = model.id + ".gguf" if model.format == ModelFormat.GGUF else model.id
    # ooba resolves user_data_dir relative to CWD when no flag is given; since the
    # launcher runs from ~/ai-server-tools, we must pass the absolute path explicitly.
    user_data_dir = str(CONFIG.ooba_main.parent / "user_data")
    cmd = [
        str(CONFIG.ooba_python), str(CONFIG.ooba_main),
        "--model",          model_name,
        "--model-dir",      p["model_dir"],
        "--user-data-dir",  user_data_dir,
        "--ctx-size",       str(model.effective_ctx),
        "--api",
        "--api-port",       str(p["api_port"]),
        "--listen",
        "--listen-host",    "0.0.0.0",
        "--listen-port",    str(p["webui_port"]),
    ]
    if p.get("loader"):
        cmd += ["--loader", p["loader"]]
    if p.get("nowebui"):
        cmd.append("--nowebui")
    if model.format == ModelFormat.GGUF:
        cmd += ["--gpu-layers", str(p["gpu_layers"])]
        if p.get("tensor_split"):
            cmd += ["--tensor-split", p["tensor_split"]]
    if p.get("cache_type"):
        cmd += ["--cache-type", p["cache_type"]]
    rq = p.get("_runtime_quant")
    if rq == RuntimeQuant.BNB_INT8:
        cmd.append("--load-in-8bit")
    elif rq == RuntimeQuant.BNB_NF4:
        cmd.append("--load-in-4bit")
    cmd.extend(p.get("extra_flags", []))
    return cmd


def _build_sglang(server: ServerSoftware, model: AIModel, p: dict) -> list[str]:
    cmd = [
        str(CONFIG.sglang_python), "-m", "sglang.launch_server",
        "--model-path",            p["_model_path"],
        "--served-model-name",     model.id,
        "--dtype",                 "bfloat16",
        "--tensor-parallel-size",  str(p["tensor_parallel"]),
        "--pipeline-parallel-size", str(p["pipeline_parallel"]),
        "--mem-fraction-static",   str(p["mem_fraction"]),
        "--context-length",        str(model.effective_ctx),
        "--trust-remote-code",
        "--host",                  "0.0.0.0",
        "--port",                  str(p["port"]),
        "--log-level",             "warning",
    ]
    if model.format == ModelFormat.GGUF:
        cmd += ["--load-format", "gguf"]
    elif (
        model.format == ModelFormat.INT8_BNB
        or p.get("_runtime_quant") == RuntimeQuant.BNB_INT8
    ):
        cmd += ["--load-format", "bitsandbytes", "--quantization", "bitsandbytes"]
    cmd.extend(p.get("extra_flags", []))
    return cmd


def _build_comfyui(server: ServerSoftware, model: AIModel, p: dict) -> list[str]:
    # ComfyUI discovers models via ~/ComfyUI/extra_model_paths.yaml at startup;
    # no model path is passed at launch time.
    cmd = [
        str(CONFIG.comfyui_python), str(CONFIG.comfyui_main),
        "--listen", "0.0.0.0",
        "--port",   str(p["port"]),
    ]
    cmd.extend(p.get("extra_flags", []))
    return cmd


_BUILDERS = {
    "vllm":     _build_vllm,
    "llamacpp": _build_llamacpp,
    "tabby":    _build_tabby,
    "ooba":     _build_ooba,
    "sglang":   _build_sglang,
    "comfyui":  _build_comfyui,
}


# ---------------------------------------------------------------------------
# Quantization tool availability
# ---------------------------------------------------------------------------

# Availability is checked once per tool and cached; exl2-convert requires a
# subprocess call (to check importability without triggering JIT compilation).
_tool_avail_cache: dict[str, bool] = {}


def quant_tool_available(tool: QuantizationTool) -> bool:
    """Return True if the tool's required binary or Python package is present.

    Result is cached after the first call — safe to call in hot paths.
    """
    if tool.id not in _tool_avail_cache:
        _tool_avail_cache[tool.id] = _check_tool_available(tool)
    return _tool_avail_cache[tool.id]


def _check_tool_available(tool: QuantizationTool) -> bool:
    if tool.avail_binary:
        return Path(tool.avail_binary).is_file()
    if tool.avail_module:
        python, module = tool.avail_module[0], tool.avail_module[1]
        try:
            r = subprocess.run(
                [python, "-c",
                 f"import importlib.util; raise SystemExit(0 if importlib.util.find_spec({module!r}) else 1)"],
                capture_output=True, timeout=5,
            )
            return r.returncode == 0
        except Exception:
            return False
    return True  # no check configured — assume available


def install_quant_tool(tool: QuantizationTool) -> tuple[bool, str]:
    """Run the tool's auto_install_cmd and return (success, log_path).

    Logs stdout/stderr to /tmp/launcher_install_<id>.log.
    Updates the availability cache on success.
    Returns (False, "") when the tool has no auto_install_cmd.
    """
    if not tool.auto_install_cmd:
        return False, ""
    log_path = f"/tmp/launcher_install_{tool.id}.log"
    with open(log_path, "w") as f:
        result = subprocess.run(
            list(tool.auto_install_cmd),
            stdout=f, stderr=subprocess.STDOUT,
            timeout=300,
        )
    success = result.returncode == 0
    _tool_avail_cache[tool.id] = success
    return success, log_path


# ---------------------------------------------------------------------------
# Quantization command builders
# ---------------------------------------------------------------------------

def _strip_gguf_quant(stem: str) -> str:
    """Strip a known GGUF quant suffix from a filename stem.

    "BeMag-123B-v0.1.Q8_0"  →  "BeMag-123B-v0.1"
    "MyModel.IQ3_XXS"        →  "MyModel"
    "NoQuant"                →  "NoQuant"   (returned unchanged)
    """
    upper = stem.upper()
    for q in GGUF_QUANTS_LONGEST_FIRST:
        suffix = f".{q.value}"
        if upper.endswith(suffix):
            return stem[: len(stem) - len(suffix)]
    return stem


def compute_quant_output_path(
    tool: QuantizationTool,
    model: AIModel,
    quant_option: str,
    dest: CacheMount,
) -> Path:
    """Derive the output *directory* for a quantization job.

    Every job produces its own uniquely-named directory so that removal
    is always consistent: delete the directory and its contents.

    GGUF re-quantize:
        <dest.root>/<stem_no_quant>-GGUF-<quant>/
        e.g. ~/fast_models/BeMag-123B-v0.1-GGUF-Q4_K_M/
        The .gguf file inside is <stem_no_quant>.<quant>.gguf,
        constructed by _build_llama_quantize.

    EXL2 convert:
        <dest.root>/<source_dirname>-exl2-<bpw>bpw/
        e.g. ~/fast_models/BeMag-123B-v0.1-exl2-4.0bpw/
    """
    src = Path(model.resolve_path())
    if tool.output_format == ModelFormat.GGUF:
        stem = _strip_gguf_quant(src.stem)
        return dest.root / f"{stem}-GGUF-{quant_option}"
    else:
        return dest.root / f"{src.name}-exl2-{quant_option}bpw"


def _build_llama_quantize(
    tool: QuantizationTool,
    model: AIModel,
    quant_option: str,
    output_dir: Path,
) -> list[str]:
    # output_dir is the per-job directory; the actual file goes inside it.
    src = Path(model.resolve_path())
    stem = _strip_gguf_quant(src.stem)
    output_file = output_dir / f"{stem}.{quant_option}.gguf"
    return [
        str(CONFIG.llama_quantize),
        model.resolve_path(),
        str(output_file),
        quant_option,
    ]


def _build_exl2_convert(
    tool: QuantizationTool,
    model: AIModel,
    quant_option: str,
    output_path: Path,
) -> list[str]:
    script = CONFIG.exl2_convert_script
    if script.exists():
        cmd_prefix = [str(CONFIG.tabby_python), str(script)]
    else:
        cmd_prefix = [str(CONFIG.tabby_python), "-m", "exllamav2.convert"]
    return [
        *cmd_prefix,
        "-i", model.resolve_path(),
        "-o", str(output_path),
        "-b", quant_option,
    ]


_QUANT_BUILDERS: dict[str, Any] = {
    "llama-quantize": _build_llama_quantize,
    "exl2-convert":   _build_exl2_convert,
}


def build_quantize_cmd(
    tool: QuantizationTool,
    model: AIModel,
    quant_option: str,
    dest: CacheMount,
) -> tuple[list[str], Path]:
    """Return (argv, output_path) for the given quantization job."""
    output_path = compute_quant_output_path(tool, model, quant_option, dest)
    builder = _QUANT_BUILDERS.get(tool.id)
    if builder is None:
        raise ValueError(f"No command builder registered for quantization tool: {tool.id!r}")
    return builder(tool, model, quant_option, output_path), output_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_cmd(server: ServerSoftware, model: "AIModel | None", runtime_quant: "RuntimeQuant | None" = None, param_overrides: dict[str, Any] | None = None) -> list[str]:
    """Return the full argv list for launching model on server.

    model may be None for servers that use stage_all_models (e.g. ComfyUI),
    where no single model is selected at launch time.

    param_overrides is merged on top of default_params and model.server_params;
    used by GpuSelectScreen to inject gpu_count_param (e.g. pipeline_parallel).
    """
    p = _merged(server, model, param_overrides)
    if model is not None:
        p["_model_path"] = model.resolve_path()  # fastest available copy
    p["_runtime_quant"] = runtime_quant
    builder = _BUILDERS.get(server.id)
    if builder is None:
        raise ValueError(f"No command builder registered for server: {server.id!r}")
    if server.stage_all_models:
        return builder(server, model, p)  # type: ignore[arg-type]  # builder ignores model
    assert model is not None, f"model is required for server {server.id!r}"
    return builder(server, model, p)


def format_cmd(cmd: list[str]) -> str:
    """Human-readable multi-line shell representation of a command."""
    return " \\\n  ".join(shlex.quote(c) for c in cmd)


def _spawn_with_pty(
    cmd: list[str],
    env: dict[str, str],
    cwd: "str | None" = None,
) -> "tuple[subprocess.Popen[bytes], int]":
    """Wrap cmd in bash, open a PTY sized to the current terminal, and spawn.

    Returns (proc, master_fd).  The caller is responsible for closing master_fd
    after the tee thread finishes reading from it.
    """
    master_fd, slave_fd = pty.openpty()
    # Match the PTY dimensions to the real terminal so width-sensitive output
    # (tqdm progress bars, torch-compile banners, etc.) renders correctly.
    try:
        term_cols, term_rows = shutil.get_terminal_size()
        fcntl.ioctl(
            slave_fd, termios.TIOCSWINSZ,
            struct.pack("HHHH", term_rows, term_cols, 0, 0),
        )
    except Exception:
        pass  # non-fatal — falls back to whatever tqdm/server defaults to
    # Wrap in bash so 'Segmentation fault (core dumped)' and other shell-level
    # signal diagnostics appear in the log.  Bash exits with 128+signum.
    bash_cmd = ["bash", "-c", shlex.join(cmd)]
    proc = subprocess.Popen(
        bash_cmd,
        env=env,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=slave_fd,
        stderr=slave_fd,
    )
    os.close(slave_fd)  # child has its own copy; close parent's reference
    return proc, master_fd


def launch(server: ServerSoftware, model: "AIModel | None", runtime_quant: "RuntimeQuant | None" = None, cuda_devices: str | None = None, param_overrides: dict[str, Any] | None = None) -> tuple[subprocess.Popen[bytes], Path, int]:
    """Build the command and spawn the server process.

    cuda_devices: overrides CONFIG.resolved_cuda_devices when set (from GpuSelectScreen).
    param_overrides: merged on top of server/model params (e.g. pipeline_parallel → GPU count).
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_devices if cuda_devices is not None else CONFIG.resolved_cuda_devices

    if server.strip_pythonpath:
        env.pop("PYTHONPATH", None)

    if server.cuda_home:
        env["CUDA_HOME"] = server.cuda_home

    if server.uses_staging_dir:
        p = _merged(server, model)
        staging_base = Path(p["model_dir"])
        staging_base.mkdir(parents=True, exist_ok=True)
        # stage_all_models: create symlinks for every compatible model in the catalog.
        # Used by servers (e.g. ComfyUI) where the client picks the model per-request.
        # stage single model: used by servers that load exactly one model at startup.
        if server.stage_all_models:
            models_to_stage: list[AIModel] = [m for m in AI_MODELS if m.compatible_with(server)]
        else:
            assert model is not None, f"model is required for server {server.id!r}"
            models_to_stage = [model]
        for m in models_to_stage:
            # staging_subdir_attr: route each model into the correct subdirectory.
            # e.g. ComfyUI uses model.comfyui_subdir ('checkpoints' or 'diffusion_models').
            if server.staging_subdir_attr:
                subdir = getattr(m, server.staging_subdir_attr, "")
                staging = staging_base / subdir if subdir else staging_base
                staging.mkdir(parents=True, exist_ok=True)
            else:
                staging = staging_base
            link_name = m.id + server.staging_format_suffixes.get(m.format, "")
            if server.staging_use_source_filename:
                link_name = Path(m.resolve_path()).name
            link = staging / link_name
            if link.is_symlink():
                link.unlink()
            link.symlink_to(m.resolve_path())

    cwd = server.launch_cwd
    _LOGS_DIR.mkdir(exist_ok=True)
    model_slug = model.id if model is not None else "no-model"
    log_path = _LOGS_DIR / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{server.id}_{model_slug}.log"
    cmd = build_cmd(server, model, runtime_quant=runtime_quant, param_overrides=param_overrides)
    quant_str = f" + {runtime_quant.value}" if runtime_quant else ""
    if model is not None:
        model_line = f"# model:  {model.id}  ({model.format.value}{quant_str}, {model.size_gb} GB)\n"
        path_line  = f"# path:   {model.resolve_path()}\n"
    else:
        model_line = "# model:  (all catalog models staged)\n"
        path_line  = ""
    header = (
        f"# launcher session log\n"
        f"# date:   {datetime.now().isoformat(timespec='seconds')}\n"
        f"{model_line}"
        f"# server: {server.id}\n"
        f"{path_line}"
        f"# CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}\n"
        f"#\n"
        f"# command:\n"
        f"{format_cmd(cmd)}\n"
        f"# {'─' * 72}\n"
    )
    log_path.write_text(header, encoding="utf-8")
    proc, master_fd = _spawn_with_pty(cmd, env, cwd)
    return proc, log_path, master_fd


def quantize(
    tool: QuantizationTool,
    model: AIModel,
    quant_option: str,
    dest: CacheMount,
) -> tuple[subprocess.Popen[bytes], Path, int]:
    """Build the quantization command and spawn the process.

    Follows the same PTY + log pattern as launch() so the caller can use
    the same tee / SIGWINCH / exit-footer machinery in app.py.
    Returns (proc, log_path, master_fd).
    """
    cmd, output_path = build_quantize_cmd(tool, model, quant_option, dest)

    # output_path is always a directory — create it before the process starts.
    output_path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if tool.cuda_home:
        env["CUDA_HOME"] = tool.cuda_home

    _LOGS_DIR.mkdir(exist_ok=True)
    safe_quant = quant_option.replace("/", "-")
    log_path = _LOGS_DIR / (
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        f"_quantize_{tool.id}_{model.id}_{safe_quant}.log"
    )
    header = (
        f"# launcher quantize log\n"
        f"# date:    {datetime.now().isoformat(timespec='seconds')}\n"
        f"# model:   {model.id}  ({model.format.value}, {model.size_gb} GB)\n"
        f"# tool:    {tool.id}\n"
        f"# option:  {quant_option}\n"
        f"# source:  {model.resolve_path()}\n"
        f"# output:  {output_path}\n"
        f"#\n"
        f"# command:\n"
        f"{format_cmd(cmd)}\n"
        f"# {'─' * 72}\n"
    )
    log_path.write_text(header, encoding="utf-8")
    proc, master_fd = _spawn_with_pty(cmd, env)
    return proc, log_path, master_fd


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def cache_disk_info(mount: CacheMount) -> tuple[float, float]:
    """Return (total_gb, free_gb) for the filesystem containing mount.root."""
    try:
        return (
            shutil.disk_usage(str(mount.root)).total / 1024**3,
            shutil.disk_usage(str(mount.root)).free  / 1024**3,
        )
    except OSError:
        return 0.0, 0.0


def cache_source_size_gb(path: Path) -> float:
    """Return the actual on-disk footprint of *path*, excluding any .git subtree.

    Used by the cache UI to estimate required space — independent of the size_gb
    catalog field, which records only the logical weight files.  For a single
    file, returns that file's size.  For a directory, recursively sums all files
    except those under a .git/ subdirectory.
    """
    if path.is_file():
        return path.stat().st_size / 1024**3
    total = 0
    for p in path.rglob("*"):
        if ".git" in p.parts:
            continue
        if p.is_file():
            total += p.stat().st_size
    return total / 1024**3


def cache_model(model: AIModel, mount: CacheMount) -> tuple["subprocess.Popen[bytes]", str]:
    """Rsync model from source to a local cache mount.

    Returns a running (Popen, dest_path) pair.  dest_path is where the cached
    copy will live once rsync completes.  On success the caller should persist
    it with save_cached_path(model, dest_path).
    """
    src_p = Path(model.path)
    dest_root = mount.root / model.cache_subdir
    if src_p.is_file():                    # single file → <mount>/<subdir>/<id>/<filename>
        dest_dir = dest_root / model.id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_p = dest_dir / src_p.name
        src, dst = str(src_p), str(dest_p)
    else:                                   # directory → <mount>/<subdir>/<src_dirname>/
        dest_p = dest_root / src_p.name
        dest_p.mkdir(parents=True, exist_ok=True)
        src = str(src_p) + "/"
        dst = str(dest_p) + "/"
    proc = subprocess.Popen(
        ["rsync", "-ah", "--info=progress2", "--exclude=.git", src, dst],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc, str(dest_p)


def uncache_model(model: AIModel, mount: CacheMount) -> None:
    """Delete the cached copy.  Never touches the original source.

    Raises FileNotFoundError if the model has no cached_path set.
    The mount parameter is accepted for API compatibility but the actual
    path is taken from model.cached_path.
    """
    if not model.cached_path:
        raise FileNotFoundError(f"No cached_path set for model {model.id!r}")
    cached_p = Path(model.cached_path)
    if not cached_p.exists():
        raise FileNotFoundError(f"Cached path not found on disk: {cached_p}")
    if cached_p.is_file():
        cached_p.unlink()
    else:
        shutil.rmtree(str(cached_p))


def save_server_status(
    model: AIModel,
    server_id: str,
    status: str,
    quant: "RuntimeQuant | None" = None,
) -> None:
    """Persist a model/server/quant status result back into models_local.json.

    Mutates the in-memory model immediately, then rewrites the server_status
    entry for that model in the JSON catalog so it survives the next launch.
    When status is 'good', any other quant entry that was 'good' for the same
    server is demoted to 'unknown' (exclusive: one good quant per server).
    """
    qk = _quant_key(quant)

    # 1. Enforce exclusivity: only one quant may be 'good' per (model, server).
    if status == "good":
        for k, v in list(model.server_status.get(server_id, {}).items()):
            if k != qk and v == "good":
                model.server_status[server_id][k] = "unknown"  # type: ignore[index]

    # 2. Update in-memory object immediately.
    model.server_status.setdefault(server_id, {})[qk] = status  # type: ignore[assignment]

    # 3. Persist to JSON catalog.
    def _apply(entry: dict) -> None:
        quant_dict = entry.setdefault("server_status", {}).setdefault(server_id, {})
        if status == "good":
            for k in list(quant_dict):
                if k != qk and quant_dict[k] == "good":
                    quant_dict[k] = "unknown"
        quant_dict[qk] = status

    _patch_catalog(model.id, _apply)


def save_ctx_override(model: AIModel, value: int) -> None:
    """Persist ctx_override for a model.  value=0 clears it (reverts to native ctx).

    Mutates the in-memory model immediately, then updates the JSON catalog.
    """
    model.ctx_override = value

    def _apply(entry: dict) -> None:
        if value:
            entry["ctx_override"] = value
        else:
            entry.pop("ctx_override", None)

    _patch_catalog(model.id, _apply)


def save_cached_path(model: AIModel, cached_path: str) -> None:
    """Persist cached_path for all catalog entries sharing the same source path.

    Multiple catalog entries can point to the same source path (e.g. different
    format variants of a directory-based model).  Caching that directory applies
    to all of them — every co-path entry receives the same cached_path, or has
    it cleared together when uncached.

    Mutates all matching in-memory models immediately, then rewrites all matching
    entries in the JSON catalog in a single file write.
    Called after a successful cache_model() rsync (to set) or after
    uncache_model() (to clear).
    """
    source_path = model.path

    # 1. Update all in-memory models with the same source path.
    for m in AI_MODELS:
        if m.path == source_path:
            m.cached_path = cached_path

    # 2. Update all matching entries in the JSON catalog in one write.
    with open(_CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)
    for entry in catalog["models"]:
        if entry["path"] == source_path:
            if cached_path:
                entry["cached_path"] = cached_path
            else:
                entry.pop("cached_path", None)
    with open(_CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
        f.write("\n")


def save_favorite(model: AIModel, favorite: bool) -> None:
    """Persist the favorite flag for a model.

    Mutates the in-memory model immediately, then updates the JSON catalog.
    favorite=False removes the key from JSON to keep the catalog tidy.
    """
    model.favorite = favorite

    def _apply(entry: dict) -> None:
        if favorite:
            entry["favorite"] = True
        else:
            entry.pop("favorite", None)

    _patch_catalog(model.id, _apply)
