"""
environments.py — Server runtime environments (conda envs, native builds).

Each ServerSoftware runs inside one of several registered *environments*: a conda
env providing the Python interpreter and packages (vLLM, SGLang, tabbyAPI, …) or a
native build directory providing binaries (llama.cpp).  Several environments may be
registered per server (e.g. a stable "vllm" and an experimental "vllm-next"); the
user picks one at launch time.

The registry is persisted in launcher/user_data/environments.json (gitignored;
copy from environments.json.example).  It is keyed by ServerSoftware.id, or by a
quantization tool's env_group when the tool needs an environment of its own
(e.g. "llm-compressor"):

    {"environments": {"vllm": [{"kind": "conda", "name": "vllm", "default": true}, …]}}

Adding a new environment kind:
  1. Subclass Environment, set `kind` / `kind_label`, implement the abstract members
     and override lifecycle hooks (create/remove/rename) where the kind supports them.
  2. Register the class in _ENV_KINDS.
"""
from __future__ import annotations

import functools
import json
import re
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from launcher.config import CONFIG, expand_path


_ENV_PATH     = Path(__file__).parent / "user_data" / "environments.json"
_EXAMPLE_PATH = Path(__file__).parent / "environments.json.example"

# Conda's own env-name rule (also safe as a JSON key and in log/file names).
_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


@functools.lru_cache(maxsize=64)
def _cached_command_output(exe: str, args: "tuple[str, ...]", mtime_ns: int) -> str:
    """Run a quick probe command; mtime_ns is part of the cache key only."""
    try:
        r = subprocess.run([exe, *args], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return ""
    return r.stdout + r.stderr


@dataclass(frozen=True)
class EnvField:
    """A kind-specific attribute the user must supply when registering an environment."""
    key:         str   # to_dict()/from_dict() key and constructor kwarg
    label:       str   # prompt label
    placeholder: str = ""


# ---------------------------------------------------------------------------
# Environment kinds
# ---------------------------------------------------------------------------

class Environment(ABC):
    """A runtime environment that a server process is launched from."""

    kind:       ClassVar[str]                    # JSON discriminator
    kind_label: ClassVar[str]                    # short display label
    # Extra attributes requested when registering a new environment of this kind.
    extra_fields: ClassVar[tuple[EnvField, ...]] = ()
    # True when create_cmd() can build the underlying environment from scratch.
    supports_create: ClassVar[bool] = False

    def __init__(self, server_id: str, name: str, note: str = "", default: bool = False) -> None:
        self.server_id = server_id
        self.name      = name
        self.note      = note
        self.default   = default

    # -- identity / location ------------------------------------------------

    @property
    @abstractmethod
    def location(self) -> Path:
        """Filesystem root of the environment (shown in the UI)."""

    @property
    @abstractmethod
    def bin_dir(self) -> Path:
        """Directory holding the environment's executables."""

    @abstractmethod
    def exists(self) -> bool:
        """True when the underlying environment is present on disk."""

    def shares_underlying(self, other: "Environment") -> bool:
        """True when *other* refers to the same on-disk environment as self."""
        return type(self) is type(other) and self.location == other.location

    # -- resolution -----------------------------------------------------------

    def executable(self, name: str) -> Path:
        """Absolute path of an executable inside this environment."""
        return self.bin_dir / name

    @property
    def python(self) -> Path:
        return self.executable("python")

    def find_package_path(self, pattern: str) -> Path | None:
        """Resolve a glob relative to the env's Python package dir; None if unsupported/absent."""
        return None

    def package_version(self, distribution: str) -> str:
        """Installed version of a Python distribution, or "" when unknown."""
        return ""

    def version_label(self, distributions: "tuple[str, ...]") -> str:
        """e.g. 'vllm 0.26.0' — installed versions of the given distributions ('' if none found)."""
        found = ((d, self.package_version(d)) for d in distributions)
        return "  ·  ".join(f"{d} {v}" for d, v in found if v)

    def command_output(self, executable: str, args: "tuple[str, ...]") -> str:
        """Combined stdout+stderr of a short-lived command in this env ('' on failure).

        Cached per executable path and modification time, so a rebuild is picked
        up while repeated UI refreshes cost nothing.
        """
        exe = self.executable(executable)
        try:
            mtime = exe.stat().st_mtime_ns
        except OSError:
            return ""
        return _cached_command_output(str(exe), args, mtime)

    # -- lifecycle hooks (argv for the external tool; None = registry-only) ----

    def create_cmd(self, python_version: str = "", clone_from: str = "") -> list[str]:
        raise NotImplementedError(f"{self.kind_label} environments cannot be created by the launcher")

    def remove_cmd(self) -> list[str] | None:
        """Command that deletes the underlying environment; None when there is nothing to delete."""
        return None

    def rename_cmd(self, new_name: str) -> list[str] | None:
        """Command that renames the underlying environment; None = rename the registry entry only."""
        return None

    # -- (de)serialisation ------------------------------------------------------

    def _extra_dict(self) -> dict[str, Any]:
        return {}

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"kind": self.kind, "name": self.name, **self._extra_dict()}
        if self.note:
            d["note"] = self.note
        if self.default:
            d["default"] = True
        return d

    @staticmethod
    def from_dict(server_id: str, d: dict[str, Any]) -> "Environment":
        cls = _ENV_KINDS[d.get("kind", "conda")]
        extras = {f.key: d[f.key] for f in cls.extra_fields}
        return cls(
            server_id=server_id,
            name=d["name"],
            note=d.get("note", ""),
            default=bool(d.get("default", False)),
            **extras,
        )

    @staticmethod
    def validate_name(name: str) -> str | None:
        """Return an error message when *name* is not a valid environment name."""
        if not name:
            return "Name must not be empty"
        if not _NAME_RE.match(name):
            return "Name may contain only letters, digits, '_', '.' and '-'"
        return None


class CondaEnvironment(Environment):
    """A named conda environment under CONFIG.conda_base/envs/."""

    kind            = "conda"
    kind_label      = "conda"
    supports_create = True

    @property
    def location(self) -> Path:
        return CONFIG.conda_base / "envs" / self.name

    @property
    def bin_dir(self) -> Path:
        return self.location / "bin"

    def exists(self) -> bool:
        return (self.location / "conda-meta").is_dir()

    @property
    def site_packages(self) -> Path | None:
        """The env's site-packages dir (conda also creates a python3.1 alias of python3.1x)."""
        found = {p.resolve() for p in (self.location / "lib").glob("python3.*/site-packages") if p.is_dir()}
        return sorted(found)[-1] if found else None

    def find_package_path(self, pattern: str) -> Path | None:
        sp = self.site_packages
        if sp is None:
            return None
        matches = sorted(sp.glob(pattern))
        return matches[-1] if matches else None

    def package_version(self, distribution: str) -> str:
        sp = self.site_packages
        if sp is None:
            return ""
        norm = re.sub(r"[-_.]+", "_", distribution).lower()
        for info in sp.glob("*.dist-info"):
            dist, _, version = info.name[: -len(".dist-info")].partition("-")
            if re.sub(r"[-_.]+", "_", dist).lower() == norm:
                return version
        return ""

    @staticmethod
    def _conda() -> str:
        return str(CONFIG.conda_base / "bin" / "conda")

    def create_cmd(self, python_version: str = "", clone_from: str = "") -> list[str]:
        if clone_from:
            return [self._conda(), "create", "-y", "-n", self.name, "--clone", clone_from]
        return [self._conda(), "create", "-y", "-n", self.name, f"python={python_version or '3.11'}"]

    def remove_cmd(self) -> list[str] | None:
        return [self._conda(), "env", "remove", "-y", "-n", self.name]

    def rename_cmd(self, new_name: str) -> list[str] | None:
        return [self._conda(), "rename", "-n", self.name, new_name]


class NativeEnvironment(Environment):
    """A directory of native binaries (e.g. a llama.cpp build's bin/ directory)."""

    kind       = "native"
    kind_label = "native"
    extra_fields = (
        EnvField("bin_dir", "Binary directory", "e.g. ~/llama.cpp/build/bin"),
    )

    def __init__(self, server_id: str, name: str, bin_dir: str, note: str = "", default: bool = False) -> None:
        super().__init__(server_id, name, note, default)
        self._bin_dir_raw = bin_dir

    @property
    def location(self) -> Path:
        return expand_path(self._bin_dir_raw)

    @property
    def bin_dir(self) -> Path:
        return self.location

    def exists(self) -> bool:
        return self.location.is_dir()

    def shares_underlying(self, other: Environment) -> bool:
        # Native entries are registry-only labels; renaming one never affects another.
        return other is self

    def _extra_dict(self) -> dict[str, Any]:
        return {"bin_dir": self._bin_dir_raw}


_ENV_KINDS: dict[str, type[Environment]] = {
    cls.kind: cls for cls in (CondaEnvironment, NativeEnvironment)
}


def env_kind(kind: str) -> type[Environment]:
    """Return the Environment subclass for a kind discriminator."""
    return _ENV_KINDS[kind]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class EnvironmentRegistry:
    """All registered environments, grouped by ServerSoftware.id (or tool env_group), persisted as JSON."""

    def __init__(self, path: Path, envs: dict[str, list[Environment]]) -> None:
        self._path = path
        self._envs = envs

    @classmethod
    def load(cls, path: Path) -> "EnvironmentRegistry":
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        envs = {
            sid: [Environment.from_dict(sid, d) for d in entries]
            for sid, entries in raw.get("environments", {}).items()
        }
        return cls(path, envs)

    def save(self) -> None:
        data = {
            "environments": {
                sid: [e.to_dict() for e in entries]
                for sid, entries in self._envs.items()
                if entries
            }
        }
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")

    # -- queries ----------------------------------------------------------------

    def for_server(self, server_id: str) -> list[Environment]:
        return list(self._envs.get(server_id, []))

    def get(self, server_id: str, name: str) -> Environment | None:
        return next((e for e in self._envs.get(server_id, []) if e.name == name), None)

    def default_for(self, server_id: str) -> Environment | None:
        """The server's default environment: the one flagged default, else the first."""
        entries = self._envs.get(server_id, [])
        return next((e for e in entries if e.default), entries[0] if entries else None)

    def sharing(self, env: Environment) -> list[Environment]:
        """All registry entries (across servers) backed by the same on-disk environment."""
        return [e for entries in self._envs.values() for e in entries if env.shares_underlying(e)]

    def all(self) -> list[Environment]:
        return [e for entries in self._envs.values() for e in entries]

    # -- mutations (each persists immediately) ---------------------------------

    def add(self, env: Environment) -> None:
        if err := Environment.validate_name(env.name):
            raise ValueError(err)
        if self.get(env.server_id, env.name) is not None:
            raise ValueError(f"Environment {env.name!r} is already registered for {env.server_id}")
        entries = self._envs.setdefault(env.server_id, [])
        env.default = not entries
        entries.append(env)
        self.save()

    def remove(self, env: Environment) -> None:
        entries = self._envs.get(env.server_id, [])
        entries.remove(env)
        if env.default and entries:
            entries[0].default = True
        self.save()

    def rename_conflict(self, env: Environment, new_name: str) -> str | None:
        """Error message when *env* cannot be renamed to *new_name*, else None."""
        if err := Environment.validate_name(new_name):
            return err
        for e in self.sharing(env):
            if self.get(e.server_id, new_name) is not None:
                return f"Environment {new_name!r} is already registered for {e.server_id}"
        return None

    def rename(self, env: Environment, new_name: str) -> list[Environment]:
        """Rename *env* and every entry sharing its underlying environment.

        Returns the renamed entries so callers can migrate data keyed by name.
        """
        if err := self.rename_conflict(env, new_name):
            raise ValueError(err)
        affected = self.sharing(env)
        for e in affected:
            e.name = new_name
        self.save()
        return affected

    def set_default(self, env: Environment) -> None:
        for e in self._envs.get(env.server_id, []):
            e.default = e is env
        self.save()

    def set_note(self, env: Environment, note: str) -> None:
        env.note = note
        self.save()


def _load_registry() -> EnvironmentRegistry:
    try:
        return EnvironmentRegistry.load(_ENV_PATH)
    except FileNotFoundError:
        print(
            f"\nERROR: {_ENV_PATH} not found.\n"
            f"Copy  {_EXAMPLE_PATH}\n"
            f"  →   {_ENV_PATH}\n"
            "and register the environment(s) for each server.\n",
            file=sys.stderr,
        )
        sys.exit(1)


ENVIRONMENTS = _load_registry()
