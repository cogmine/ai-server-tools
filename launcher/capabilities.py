"""capabilities.py — What a server environment implements, read from the environment itself.

Architecture support differs between registered environments (builds / versions),
so features that only some architectures implement are detected per environment
rather than listed by hand.

llama.cpp MTP: the compiled libllama.so exports the model classes that have an MTP
draft graph (``llama_model_<x>::graph_mtp``), and the build's source tree maps GGUF
architecture names to those classes.  Cached per file modification time, so a
rebuild is picked up automatically.

vLLM MTP: vLLM's own speculative-config mapping is run on the model's config.json
inside the environment (a short subprocess — importing vLLM takes seconds), so the
verdict is persisted per vLLM version and config.json content.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from launcher.gguf_header import read_gguf_metadata

if TYPE_CHECKING:
    from launcher.environments import Environment
    from launcher.models import AIModel

_LIBRARY = "libllama.so"
# Itanium-mangled nested name: <length>llama_model_<x>9graph_mtp
_MTP_GRAPH_SYMBOL = re.compile(rb"(\d+)(llama_model_\w+?)9graph_mtp")
_ARCH_NAME = re.compile(r'\{\s*(LLM_ARCH_\w+)\s*,\s*"([^"]+)"\s*\}')
_ARCH_CLASS = re.compile(r"((?:case\s+LLM_ARCH_\w+\s*:\s*)+)return\s+new\s+(llama_model_\w+)\s*\(")


def _mtime(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def _length_prefixed(digits: bytes, name: bytes) -> bool:
    """True when a suffix of *digits* is the mangled length of *name* (other digits may precede it)."""
    return any(int(digits[-k:]) == len(name) for k in range(1, len(digits) + 1))


@lru_cache(maxsize=None)
def _mtp_classes(library: str, _mtime_ns: int) -> frozenset[str]:
    data = Path(library).read_bytes()
    return frozenset(
        m.group(2).decode() for m in _MTP_GRAPH_SYMBOL.finditer(data) if _length_prefixed(m.group(1), m.group(2))
    )


@lru_cache(maxsize=None)
def _arch_classes(arch_src: str, model_src: str, _mtimes: tuple[int, int]) -> dict[str, str]:
    """GGUF architecture name → llama_model_* class, from the build's source tree."""
    names = dict(_ARCH_NAME.findall(Path(arch_src).read_text(encoding="utf-8", errors="replace")))
    classes: dict[str, str] = {}
    for cases, cls in _ARCH_CLASS.findall(Path(model_src).read_text(encoding="utf-8", errors="replace")):
        for enum in re.findall(r"LLM_ARCH_\w+", cases):
            if enum in names:
                classes[names[enum]] = cls
    return classes


@lru_cache(maxsize=None)
def _gguf_architecture(path: str, _mtime_ns: int) -> str:
    return str(read_gguf_metadata(Path(path)).get("general.architecture", ""))


def llamacpp_mtp_support(bin_dir: Path, gguf_path: str) -> tuple[bool | None, str]:
    """Whether the build in *bin_dir* implements MTP drafting for the GGUF's architecture.

    Returns (supported, reason); supported is None when it cannot be determined
    (no shared library or source tree, unreadable GGUF).  An architecture missing
    from the source counts as unsupported (that build cannot load the model at all).
    """
    library = bin_dir / _LIBRARY
    src = bin_dir.parent.parent / "src"
    arch_src, model_src, gguf = src / "llama-arch.cpp", src / "llama-model.cpp", Path(gguf_path)
    lib_mtime, arch_mtime, model_mtime, gguf_mtime = map(_mtime, (library, arch_src, model_src, gguf))
    if lib_mtime is None:
        return None, f"no {_LIBRARY} in the build"
    if arch_mtime is None or model_mtime is None:
        return None, "build source tree not found"
    if gguf_mtime is None:
        return None, "model file unavailable"
    try:
        arch = _gguf_architecture(str(gguf), gguf_mtime)
    except (OSError, ValueError) as e:
        return None, f"unreadable GGUF header: {e}"
    cls = _arch_classes(str(arch_src), str(model_src), (arch_mtime, model_mtime)).get(arch)
    if cls is None:
        return False, f"this build does not know the {arch!r} architecture"
    if cls in _mtp_classes(str(library), lib_mtime):
        return True, f"this build implements MTP for {arch}"
    return False, f"this build has no MTP graph for {arch}"


_VLLM_CACHE_PATH = Path(__file__).parent / "user_data" / "cache" / "vllm_mtp.json"
_VLLM_PROBE_TIMEOUT_S = 180
# argv: model dir, trust_remote_code ("1"/"0").  Prints one JSON line.
_VLLM_MTP_PROBE = """
import json, sys
from vllm import ModelRegistry
from vllm.config.speculative import SpeculativeConfig
from vllm.transformers_utils.config import get_config
cfg = get_config(sys.argv[1], trust_remote_code=sys.argv[2] == "1")
before = cfg.architectures[0]
after = SpeculativeConfig.hf_config_override(cfg).architectures[0]
print(json.dumps({"before": before, "after": after,
                  "supported": after != before and after in ModelRegistry.get_supported_archs()}))
"""


@lru_cache(maxsize=1)
def _vllm_cache() -> dict[str, list]:
    try:
        return json.loads(_VLLM_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _run_vllm_probe(env: Environment, model: AIModel) -> tuple[bool | None, str, bool]:
    """(supported, reason, cacheable) from vLLM's own MTP mapping, run inside *env*."""
    proc_env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    try:
        r = subprocess.run(
            [str(env.python), "-c", _VLLM_MTP_PROBE, model.path, "1" if model.trust_remote_code else "0"],
            capture_output=True, text=True, timeout=_VLLM_PROBE_TIMEOUT_S, cwd="/tmp", env=proc_env,
        )
    except subprocess.TimeoutExpired:
        return None, "vLLM probe timed out", False
    except OSError as e:
        return None, f"vLLM probe failed: {e}", False
    lines = r.stdout.strip().splitlines()
    try:
        result = json.loads(lines[-1])
    except (IndexError, ValueError):
        errors = [line for line in r.stderr.splitlines() if "Error" in line]
        if not errors:
            return None, f"vLLM probe exit code {r.returncode}", True
        return None, errors[-1].strip().split(". ")[0][:160], True
    if result["supported"]:
        return True, f"this vLLM drafts {result['before']} with {result['after']}", True
    return False, f"this vLLM has no MTP model for {result['before']}", True


def vllm_mtp_support(env: Environment, model: AIModel) -> tuple[bool | None, str]:
    """Whether the vLLM in *env* can use *model*'s MTP layers as a draft (--speculative-config method mtp).

    Returns (supported, reason); supported is None when it cannot be determined.
    The first check per vLLM version and config.json content takes a few seconds;
    later ones (also for copies at other locations) are read from
    user_data/cache/vllm_mtp.json.
    """
    version = env.package_version("vllm")
    if not version:
        return None, "vLLM not installed in the environment"
    try:
        config_digest = hashlib.sha256((Path(model.path) / "config.json").read_bytes()).hexdigest()[:16]
    except OSError:
        return None, "config.json unavailable"
    key = f"{env.location}|vllm {version}|{config_digest}|{int(model.trust_remote_code)}"
    cache = _vllm_cache()
    if key not in cache:
        supported, reason, cacheable = _run_vllm_probe(env, model)
        if not cacheable:
            return supported, reason
        cache[key] = [supported, reason]
        _VLLM_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _VLLM_CACHE_PATH.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    supported, reason = cache[key]
    return supported, reason
