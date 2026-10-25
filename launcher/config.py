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


def expand_path(p: str) -> Path:
    """Expand ~ and $VAR in a path string, return an absolute Path."""
    return Path(os.path.expandvars(os.path.expanduser(p)))


def _total_ram_mib() -> int:
    """Physical RAM of this host in MiB."""
    return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") // (1024 * 1024)


def _parse_ram_mib(value: int | str | None) -> int | None:
    """Resolve a RAM size setting: "60%" → share of physical RAM in MiB; an int → MiB as given."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text.endswith("%"):
            raise ValueError(f"RAM size must be an integer MiB or a percentage like '60%', got {value!r}")
        return int(_total_ram_mib() * float(text[:-1]) / 100)
    return int(value)


def _count_gpus() -> int:
    """Return the number of available CUDA GPUs via NVML."""
    try:
        import pynvml
        pynvml.nvmlInit()
        return max(pynvml.nvmlDeviceGetCount(), 1)
    except Exception:
        return 1


@dataclass
class LocationConfig:
    """One storage location models can live on (NAS, NVMe, …)."""
    label:       str    # short name shown in the model list, e.g. "NAS"
    root:        Path
    description: str = ""


@dataclass
class SiteConfig:
    """Fully-resolved site configuration derived from config.toml."""

    conda_base:           Path   # miniconda/anaconda installation (conda envs live in envs/)
    tabby_main:           Path   # path to tabbyAPI main.py
    tabby_model_dir:      Path   # staging dir where tabby symlinks are created
    ooba_main:            Path   # path to text-generation-webui/server.py
    ooba_model_dir:       Path   # staging dir where ooba model symlinks are created
    cuda_home_tabby:      str    # CUDA_HOME set when launching tabbyAPI
    cuda_home_sglang:     str    # CUDA_HOME set when launching SGLang (for flashinfer JIT)
    cuda_visible_devices: str    # explicit GPU list, or "" to derive from pipeline_parallel
    pipeline_parallel:    int    # GPU count for vLLM PP (never 0 after load)
    exl2_convert_script:  Path   # path to ExLlamaV2 convert.py; falls back to -m exllamav2.conversion.convert_exl2 if absent
    llama_convert_python: Path   # Python interpreter with llama.cpp HF-converter dependencies
    comfyui_main:         Path   # path to ComfyUI main.py
    comfyui_models_dir:   Path   # base staging dir for ComfyUI symlinks (~/ComfyUI/models/)
    # Ceiling (MiB) for a server's host-RAM prompt cache (llama.cpp --cache-ram);
    # None = not configured, servers keep their own default.
    host_cache_ram_mib:   int | None = None
    # Host RAM (MiB) a model conversion may hold; larger sources spool converted
    # tensors to disk instead (llama.cpp convert_hf_to_gguf --use-temp-file).
    convert_ram_mib:      int = 0
    server_ports:         dict[str, int] = field(default_factory=dict)  # per-server port overrides from [ports]
    locations:            list[LocationConfig] = field(default_factory=list)  # in config order

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

    paths  = raw.get("paths",  {})
    gpu    = raw.get("gpu",    {})
    memory = raw.get("memory", {})
    ports = {k: int(v) for k, v in raw.get("ports", {}).items()}

    pp = int(gpu.get("pipeline_parallel", 0))
    if pp == 0:
        pp = _count_gpus()

    locations = [
        LocationConfig(
            label=loc["label"],
            root=expand_path(loc["root"]),
            description=loc.get("description", ""),
        )
        for loc in raw.get("locations", [])
    ]

    return SiteConfig(
        conda_base=expand_path(
            paths.get("conda_base", "~/miniconda3")),
        tabby_main=expand_path(
            paths.get("tabby_main", "~/tabbyAPI/main.py")),
        tabby_model_dir=expand_path(
            paths.get("tabby_model_dir", "~/tabbyAPI/models")),
        ooba_main=expand_path(
            paths.get("ooba_main", "~/text-generation-webui/server.py")),
        ooba_model_dir=expand_path(
            paths.get("ooba_model_dir", "~/text-generation-webui/user_data/models")),
        cuda_home_tabby=paths.get("cuda_home_tabby", ""),
        cuda_home_sglang=paths.get("cuda_home_sglang", ""),
        cuda_visible_devices=paths.get("cuda_visible_devices", ""),
        pipeline_parallel=pp,
        exl2_convert_script=expand_path(
            paths.get("exl2_convert_script", "~/exllamav2/convert.py")),
        llama_convert_python=expand_path(
            paths.get("llama_convert_python") or sys.executable),
        comfyui_main=expand_path(
            paths.get("comfyui_main", "~/ComfyUI/main.py")),
        comfyui_models_dir=expand_path(
            paths.get("comfyui_models_dir", "~/ComfyUI/models")),
        host_cache_ram_mib=_parse_ram_mib(memory.get("host_cache_ram")),
        convert_ram_mib=_parse_ram_mib(memory.get("convert_ram", "50%")) or 0,
        server_ports=ports,
        locations=locations,
    )


CONFIG = _load()
