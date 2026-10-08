"""
gpu_power.py — Per-GPU power limits chosen on GpuSelectScreen.

User choices persist in user_data/gpu_power.json keyed by full GPU UUID, so a cap
follows the physical card across slot changes and applies to every model.  They are
applied as the first step of a launch command (scripts/gpu-power-limit), which keeps
the logged command a complete, paste-ready reproduction.  Nothing is applied outside
a launch: after a reboot or GPU reset the next launch re-applies the limits.
"""
from __future__ import annotations

import json
import shlex
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from launcher.gpu import GpuRow, fmt_power_limit_w

_STORE_PATH = Path(__file__).parent / "user_data" / "gpu_power.json"
SCRIPT_PATH = Path(__file__).parent / "scripts" / "gpu-power-limit"

DEFAULT_KEYWORD = "default"  # script argument for "factory default limit"


class PowerLimitStore:
    """User-chosen power limits (watts) keyed by full GPU UUID; absent = factory default."""

    def __init__(self, path: Path, limits: dict[str, int]) -> None:
        self._path = path
        self._limits = limits

    @classmethod
    def load(cls, path: Path) -> "PowerLimitStore":
        if not path.exists():
            return cls(path, {})
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        return cls(path, {uuid: int(w) for uuid, w in raw.get("power_limits", {}).items()})

    def save(self) -> None:
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump({"power_limits": dict(sorted(self._limits.items()))}, f, indent=2)
            f.write("\n")

    def get(self, uuid: str) -> int | None:
        return self._limits.get(uuid)

    def set_for(self, rows: Iterable[GpuRow], watts: int | None) -> None:
        """Set (or clear, with None) the limit for each card and persist.

        A value equal to a card's factory default is stored as "default" (cleared).
        """
        for row in rows:
            if watts is None or watts == round(row.power_default_w):
                self._limits.pop(row.uuid, None)
            else:
                self._limits[row.uuid] = watts
        self.save()


POWER_LIMITS = PowerLimitStore.load(_STORE_PATH)


def target_watts(row: GpuRow, store: PowerLimitStore = POWER_LIMITS) -> float:
    """The limit a launch applies to *row*'s card: the chosen one, else its factory default."""
    chosen = store.get(row.uuid)
    return float(chosen) if chosen is not None else row.power_default_w


def fmt_power_limit(row: GpuRow, store: PowerLimitStore = POWER_LIMITS) -> str:
    """Markup for *row*'s limit — ``current→target W`` when a launch would change it.

    Empty when the card does not report power limits.
    """
    if row.power_default_w <= 0:
        return ""
    current = fmt_power_limit_w(row, row.power_limit_w)
    target = target_watts(row, store)
    if round(target) == round(row.power_limit_w):
        return f"{current} W"
    return f"{current}→{fmt_power_limit_w(row, target)} W"


def power_limit_range(rows: Sequence[GpuRow]) -> tuple[int, int]:
    """Watt range settable on every card in *rows* (intersection of their constraints)."""
    return round(max(r.power_min_w for r in rows)), round(min(r.power_max_w for r in rows))


def parse_power_limit(text: str, rows: Sequence[GpuRow]) -> int | None:
    """Parse user input in watts for *rows*; blank → None (factory default).

    Raises ValueError with a user-facing message when invalid or out of range.
    """
    text = text.strip().lower().removesuffix("w").strip()
    if not text or text == DEFAULT_KEYWORD:
        return None
    try:
        watts = int(text)
    except ValueError:
        raise ValueError(f"Not a whole number of watts: {text!r}") from None
    lo, hi = power_limit_range(rows)
    if not lo <= watts <= hi:
        raise ValueError(f"{watts} W is outside the settable range {lo}–{hi} W")
    return watts


@dataclass(frozen=True)
class PowerLimitStep:
    """Launch-command step applying power limits to the launched GPUs."""

    targets: dict[int, int | None]  # NVML index → watts; None = factory default

    def args(self) -> list[str]:
        """Script arguments, grouping GPUs that share a target: ``["0,1,2=250", "3=default"]``."""
        groups: dict[str, list[int]] = {}
        for index, watts in sorted(self.targets.items()):
            groups.setdefault(DEFAULT_KEYWORD if watts is None else str(watts), []).append(index)
        return [f"{','.join(map(str, indices))}={value}" for value, indices in groups.items()]

    def command(self) -> str:
        return shlex.join([str(SCRIPT_PATH), *self.args()])


def power_limit_step(
    rows: Sequence[GpuRow],
    indices: Iterable[int],
    store: PowerLimitStore = POWER_LIMITS,
) -> PowerLimitStep | None:
    """The power step for launching on *indices*, or None when not needed.

    Needed when any launched GPU has a chosen limit or currently runs at a
    non-default one — the latter so the step resets it, keeping launches reproducible.
    """
    by_index = {r.index: r for r in rows if not r.error}
    targets: dict[int, int | None] = {}
    needed = False
    for index in indices:
        row = by_index.get(index)
        if row is None:
            continue
        targets[index] = store.get(row.uuid)
        needed |= targets[index] is not None or not row.power_limit_is_default
    return PowerLimitStep(targets) if needed else None
