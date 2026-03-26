"""
config.py — Site-specific configuration loader.

Reads launcher/user_data/config.toml (gitignored; copy from config.toml.example).
Exposes a single CONFIG singleton that the rest of the launcher imports.

No other launcher module is imported here — this file must stay
dependency-free within the package to avoid circular imports.
"""
from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

_CONFIG_PATH  = Path(__file__).parent / "user_data" / "config.toml"
_EXAMPLE_PATH = Path(__file__).parent / "config.toml.example"


def _expand(p: str) -> Path:
    """Expand ~ and $VAR in a path string, return an absolute Path."""
    return Path(os.path.expandvars(os.path.expanduser(p)))


def _count_gpus() -> int:
    """Return the number of available CUDA GPUs via NVML."""
    try:
        import pynvml
        pynvml.nvmlInit()
        return max(pynvml.nvmlDeviceGetCount(), 1)
    except Exception:
        return 1


@dataclass
class CacheMountConfig:
    """One local storage tier that caches a subset of the source model library."""
    label:    str
    root:     Path
    priority: int  # lower = checked first


@dataclass
class SiteConfig:
    """Fully-resolved site configuration derived from config.toml."""

    conda_base:           Path   # path to miniconda/anaconda installation
    vllm_env:             str    # conda env name for vLLM
    tabby_env:            str    # conda env name for tabbyAPI
    llama_server:         Path   # path to llama-server binary
    tabby_main:           Path   # path to tabbyAPI main.py
    tabby_model_dir:      Path   # staging dir where tabby symlinks are created
    ooba_env:             str    # conda env name for text-generation-webui
    ooba_main:            Path   # path to text-generation-webui/server.py
    ooba_model_dir:       Path   # staging dir where ooba model symlinks are created
    sglang_env:           str    # conda env name for SGLang
    cuda_home_tabby:      str    # CUDA_HOME set when launching tabbyAPI
    cuda_home_sglang:     str    # CUDA_HOME set when launching SGLang (for flashinfer JIT)
    cuda_visible_devices: str    # explicit GPU list, or "" to derive from pipeline_parallel
    pipeline_parallel:    int    # GPU count for vLLM PP (never 0 after load)
    exl2_convert_script:  Path   # path to ExLlamaV2 convert.py; falls back to -m exllamav2.convert if absent
    comfyui_env:          str    # conda env name for ComfyUI
    comfyui_main:         Path   # path to ComfyUI main.py
    comfyui_models_dir:   Path   # base staging dir for ComfyUI symlinks (~/ComfyUI/models/)
    server_ports:         dict[str, int] = field(default_factory=dict)  # per-server port overrides from [ports]
    cache_mounts:         list[CacheMountConfig] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Derived paths
    # ------------------------------------------------------------------

    @property
    def llama_quantize(self) -> Path:
        """Path to the llama-quantize binary, co-located with llama-server."""
        return self.llama_server.parent / "llama-quantize"

    @property
    def vllm_python(self) -> Path:
        return self.conda_base / f"envs/{self.vllm_env}/bin/python"

    @property
    def tabby_python(self) -> Path:
        return self.conda_base / f"envs/{self.tabby_env}/bin/python"

    @property
    def ooba_python(self) -> Path:
        return self.conda_base / f"envs/{self.ooba_env}/bin/python"

    @property
    def sglang_python(self) -> Path:
        return self.conda_base / f"envs/{self.sglang_env}/bin/python"

    @property
    def comfyui_python(self) -> Path:
        return self.conda_base / f"envs/{self.comfyui_env}/bin/python"

    @property
    def resolved_cuda_devices(self) -> str:
        """GPU device list for CUDA_VISIBLE_DEVICES.

        Returns the explicit string from config if set, otherwise builds
        "0,1,...,N-1" from pipeline_parallel (which equals the GPU count
        when auto-detected).
        """
        if self.cuda_visible_devices:
            return self.cuda_visible_devices
        return ",".join(str(i) for i in range(self.pipeline_parallel))


def _load() -> SiteConfig:
    if not _CONFIG_PATH.exists():
        print(
            f"\nERROR: {_CONFIG_PATH} not found.\n"
            f"Copy  {_EXAMPLE_PATH}\n"
            f"  →   {_CONFIG_PATH}\n"
            "and fill in your system paths.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(_CONFIG_PATH, "rb") as f:
        raw = tomllib.load(f)

    paths = raw.get("paths", {})
    gpu   = raw.get("gpu",   {})
    ports = {k: int(v) for k, v in raw.get("ports", {}).items()}

    pp = int(gpu.get("pipeline_parallel", 0))
    if pp == 0:
        pp = _count_gpus()

    mounts: list[CacheMountConfig] = []
    for m in raw.get("cache_mounts", []):
        mounts.append(CacheMountConfig(
            label=m["label"],
            root=_expand(m["root"]),
            priority=int(m["priority"]),
        ))
    mounts.sort(key=lambda m: m.priority)

    return SiteConfig(
        conda_base=_expand(
            paths.get("conda_base", "~/miniconda3")),
        vllm_env=paths.get("vllm_env", "vllm"),
        tabby_env=paths.get("tabby_env", "tabby"),
        llama_server=_expand(
            paths.get("llama_server", "~/llama.cpp/build/bin/llama-server")),
        tabby_main=_expand(
            paths.get("tabby_main", "~/tabbyAPI/main.py")),
        tabby_model_dir=_expand(
            paths.get("tabby_model_dir", "~/tabbyAPI/models")),
        ooba_env=paths.get("ooba_env", "textgen"),
        ooba_main=_expand(
            paths.get("ooba_main", "~/text-generation-webui/server.py")),
        ooba_model_dir=_expand(
            paths.get("ooba_model_dir", "~/text-generation-webui/user_data/models")),
        sglang_env=paths.get("sglang_env", "sglang"),
        cuda_home_tabby=paths.get("cuda_home_tabby", ""),
        cuda_home_sglang=paths.get("cuda_home_sglang", ""),
        cuda_visible_devices=paths.get("cuda_visible_devices", ""),
        pipeline_parallel=pp,
        exl2_convert_script=_expand(
            paths.get("exl2_convert_script", "~/exllamav2/convert.py")),
        comfyui_env=paths.get("comfyui_env", "comfyui"),
        comfyui_main=_expand(
            paths.get("comfyui_main", "~/ComfyUI/main.py")),
        comfyui_models_dir=_expand(
            paths.get("comfyui_models_dir", "~/ComfyUI/models")),
        server_ports=ports,
        cache_mounts=mounts,
    )


CONFIG = _load()
