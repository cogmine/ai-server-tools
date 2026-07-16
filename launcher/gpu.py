"""
gpu.py — NVML polling (via pynvml) and the GpuTable Textual widget.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

_nvml: ModuleType | None = None
try:
    import pynvml as _nvml
except ImportError:
    pass

# ---------------------------------------------------------------------------
# NVML lifecycle — lazy init, cached
# ---------------------------------------------------------------------------

_nvml_ready: bool | None = None  # None = not yet attempted


def _ensure_nvml() -> bool:
    """Initialise NVML on first call; return True if available and ready."""
    global _nvml_ready
    if _nvml_ready is None:
        if _nvml is None:
            _nvml_ready = False
        else:
            nvml = _nvml
            try:
                nvml.nvmlInit()
                _nvml_ready = True
            except nvml.NVMLError:
                _nvml_ready = False
    return _nvml_ready

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import DataTable
from textual import work


@dataclass
class GpuRow:
    index: int
    bus_id: str       # 2-char hex slot, e.g. "03"
    name: str         # short model name (last component)
    power_w: float
    power_limit_w: float
    mem_used_mib: float
    mem_total_mib: float
    util_pct: float
    temp_c: float
    pcie_gen: str     # e.g. "4/4"
    pcie_width: str   # e.g. "16/16"
    error: str = ""          # non-empty → row is in error state
    uuid: str = ""           # first 8 hex chars of GPU UUID (stable across slot rotations)
    compute_cap: str = ""    # CUDA compute capability, e.g. "8.6" for sm_86
    pci_root: str = ""       # root complex domain hex, e.g. "15" — GPUs sharing a root are on the same physical slot

    @property
    def mem_pct(self) -> float:
        if self.mem_total_mib <= 0:
            return 0.0
        return self.mem_used_mib / self.mem_total_mib

    @property
    def mem_bar(self) -> str:
        filled = round(self.mem_pct * 10)
        return "█" * filled + "░" * (10 - filled)

    @property
    def mem_used_gb(self) -> float:
        return self.mem_used_mib / 1024

    @property
    def mem_total_gb(self) -> float:
        return self.mem_total_mib / 1024


def _color(value: float, low: float, high: float) -> str:
    """Return a Rich markup color tag for a 0–100 value with low/high thresholds."""
    if value >= high:
        return "red"
    if value >= low:
        return "yellow"
    return "green"


def _fmt_mem(row: GpuRow) -> str:
    pct = row.mem_pct * 100
    color = _color(pct, 60, 90)
    bar = row.mem_bar
    used = f"{row.mem_used_gb:4.1f}"
    total = f"{row.mem_total_gb:g}"
    return f"[{color}]{bar}[/] {used}/{total}G"


def _fmt_util(row: GpuRow) -> str:
    color = _color(row.util_pct, 50, 90)
    return f"[{color}]{row.util_pct:.0f}%[/]"


def _fmt_temp(row: GpuRow) -> str:
    color = _color(row.temp_c, 60, 80)
    return f"[{color}]{row.temp_c:.0f}°C[/]"


def _fmt_power(row: GpuRow) -> str:
    pct = (row.power_w / row.power_limit_w * 100) if row.power_limit_w > 0 else 0
    color = _color(pct, 60, 90)
    return f"[{color}]{row.power_w:.0f}[/]/{row.power_limit_w:.0f}W"


def _pci_root(bus_addr: str) -> str:
    """Resolve the PCIe root complex domain for a GPU via sysfs.

    *bus_addr* is the NVML busId string, e.g. ``"00000000:16:00.0"``.
    Returns the root complex hex string (e.g. ``"15"``), or ``""`` on failure.
    The sysfs symlink path contains ``pci0000:XX`` where XX is the root.
    """
    # NVML uses 8-char domain; sysfs uses 4-char: 00000000:16:00.0 → 0000:16:00.0
    # Lowercase: NVML may return uppercase hex (e.g. 8F) but sysfs uses lowercase.
    sysfs_addr = bus_addr[4:].lower()
    link = Path(f"/sys/bus/pci/devices/{sysfs_addr}")
    try:
        target = link.resolve().as_posix()  # e.g. .../pci0000:15/0000:15:01.0/0000:16:00.0
        # Extract root complex from "pci0000:XX" segment
        idx = target.find("/pci")
        if idx >= 0:
            # "pci0000:XX/" → grab the hex part after the colon
            seg = target[idx + 4:]  # "0000:15/0000:15:01.0/..."
            colon = seg.find(":")
            slash = seg.find("/")
            if colon >= 0 and slash > colon:
                return seg[colon + 1 : slash].upper()
    except (OSError, ValueError):
        pass
    return ""


def assign_slot_groups(rows: list[GpuRow]) -> dict[str, int]:
    """Map each shared ``pci_root`` to a 1-based group number (ordered by bus_id).

    Returns ``{pci_root: group_number}``.  Only roots hosting 2+ GPUs get a
    group number; solitary roots are omitted (callers treat missing keys as
    ungrouped).
    """
    # Count GPUs per root
    root_count: dict[str, int] = {}
    for row in rows:
        if row.pci_root:
            root_count[row.pci_root] = root_count.get(row.pci_root, 0) + 1

    seen: dict[str, int] = {}
    group = 0
    for row in sorted(rows, key=lambda r: r.bus_id):
        if row.pci_root and row.pci_root not in seen and root_count.get(row.pci_root, 0) >= 2:
            group += 1
            seen[row.pci_root] = group
    return seen


def poll_gpus() -> list[GpuRow]:
    """Return one GpuRow per GPU via direct NVML calls (pynvml)."""
    if not _ensure_nvml():
        return []
    nvml = _nvml
    assert nvml is not None  # guaranteed by _ensure_nvml()

    try:
        count = nvml.nvmlDeviceGetCount()
    except nvml.NVMLError:
        return []

    rows: list[GpuRow] = []
    for i in range(count):
        try:
            handle = nvml.nvmlDeviceGetHandleByIndex(i)

            uuid_raw = nvml.nvmlDeviceGetUUID(handle)
            uid_short = uuid_raw[4:12] if uuid_raw.startswith("GPU-") else uuid_raw[:8]

            pci      = nvml.nvmlDeviceGetPciInfo(handle)
            bus_id   = pci.busId.decode() if isinstance(pci.busId, bytes) else pci.busId
            bus_hex  = bus_id[9:11]  # "00000000:16:00.0" → "16"

            mem         = nvml.nvmlDeviceGetMemoryInfo(handle)
            util        = nvml.nvmlDeviceGetUtilizationRates(handle)
            temp        = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
            power_draw  = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0          # mW → W
            power_limit = nvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0

            gen_cur   = nvml.nvmlDeviceGetCurrPcieLinkGeneration(handle)
            gen_max   = nvml.nvmlDeviceGetMaxPcieLinkGeneration(handle)
            width_cur = nvml.nvmlDeviceGetCurrPcieLinkWidth(handle)
            width_max = nvml.nvmlDeviceGetMaxPcieLinkWidth(handle)

            cap_major, cap_minor = nvml.nvmlDeviceGetCudaComputeCapability(handle)

            rows.append(GpuRow(
                index=i,
                bus_id=bus_hex,
                name=uid_short,
                power_w=power_draw,
                power_limit_w=power_limit,
                mem_used_mib=int(mem.used) / (1024 * 1024),
                mem_total_mib=int(mem.total) / (1024 * 1024),
                util_pct=float(util.gpu),
                temp_c=float(temp),
                pcie_gen=f"{gen_cur}/{gen_max}",
                pcie_width=f"{width_cur}/{width_max}",
                uuid=uid_short,
                compute_cap=f"{cap_major}.{cap_minor}",
                pci_root=_pci_root(bus_id),
            ))
        except nvml.NVMLError as exc:
            rows.append(GpuRow(
                index=i,
                bus_id="??",
                name="??",
                power_w=0, power_limit_w=0,
                mem_used_mib=0, mem_total_mib=0,
                util_pct=0, temp_c=0,
                pcie_gen="?/?", pcie_width="?/?",
                error=str(exc),
            ))

    return rows


# ---------------------------------------------------------------------------
# Hardware capability helpers
# ---------------------------------------------------------------------------

_cached_sm: int | None = None


def system_min_sm() -> int:
    """Return the minimum CUDA compute capability across all GPUs as an int.

    Example: sm_8.6 (RTX 3090) → 86.  Returns 0 if NVML is unavailable
    or returns no data.  Result is cached after the first successful call.
    """
    global _cached_sm
    if _cached_sm is not None:
        return _cached_sm
    if not _ensure_nvml():
        _cached_sm = 0
        return 0
    nvml = _nvml
    assert nvml is not None  # guaranteed by _ensure_nvml()
    try:
        count = nvml.nvmlDeviceGetCount()
        caps: list[int] = []
        for i in range(count):
            try:
                handle = nvml.nvmlDeviceGetHandleByIndex(i)
                major, minor = nvml.nvmlDeviceGetCudaComputeCapability(handle)
                caps.append(major * 10 + minor)
            except nvml.NVMLError:
                pass
        _cached_sm = min(caps) if caps else 0
    except nvml.NVMLError:
        _cached_sm = 0
    return _cached_sm


class GpuTable(Widget):
    """Live-updating GPU stats table."""

    BORDER_TITLE = "GPU Stats"

    gpu_rows: reactive[list[GpuRow]] = reactive(list, recompose=False)

    def compose(self) -> ComposeResult:
        table: DataTable = DataTable(show_cursor=False)
        table.add_columns(
            "#", "ID", "SM", "Memory", "Util", "Temp", "Power", "Gen", "Width"
        )
        yield table

    def on_mount(self) -> None:
        self.refresh_gpus()
        self.set_interval(0.5, self.refresh_gpus)

    @work(thread=True)
    def refresh_gpus(self) -> None:
        rows = poll_gpus()
        self.app.call_from_thread(self._update_table, rows)

    def _update_table(self, rows: list[GpuRow]) -> None:
        table = self.query_one(DataTable)
        table.clear()
        for row in rows:
            if row.error:
                table.add_row(
                    str(row.index),
                    row.bus_id,
                    row.compute_cap or "-",
                    f"[red]!ERROR: {row.error}[/]",
                    "-", "-", "-", "-", "-",
                )
            else:
                table.add_row(
                    str(row.index),
                    row.bus_id,
                    row.compute_cap or "-",
                    _fmt_mem(row),
                    _fmt_util(row),
                    _fmt_temp(row),
                    _fmt_power(row),
                    row.pcie_gen,
                    row.pcie_width,
                )
