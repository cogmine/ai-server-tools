"""
gpu.py — NVML polling (via pynvml) and the GpuTable Textual widget.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
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

_POWER_LIMIT_TOLERANCE_W = 0.5  # NVML reports mW; ignore rounding noise vs. default

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import DataTable
from textual import work


# Hardware widths of the saturating PCIe error counters (observed on GA102).
_RX_ERR_COUNTER_MAX = 0xFFFF
_LANE_ERR_COUNTER_MAX = 0xFF

# Total link errors (since driver load) at which the Link column turns red.
_LINK_ERR_BAD = 100


@dataclass(frozen=True)
class PcieLinkHealth:
    """Cumulative PCIe link error counters reported by a GPU since driver load.

    Each error forces a retransmission or a link retrain, so a non-zero count
    points at a marginal physical connection (riser, cable, connector, slot).
    """
    receiver_errors: int  # physical-layer symbol/framing errors (saturates at 16 bits)
    bad_tlp: int          # transaction packets that failed LCRC
    bad_dllp: int         # data-link packets that failed CRC
    replays: int          # packets the GPU had to resend (host-side receive errors)
    lane_errors: tuple[int, ...] = ()  # per-lane counts; only read when errors exist

    @property
    def total(self) -> int:
        return self.receiver_errors + self.bad_tlp + self.bad_dllp + self.replays

    @property
    def saturated(self) -> bool:
        """True when a counter is pinned at its hardware maximum (true count is higher)."""
        return (self.receiver_errors >= _RX_ERR_COUNTER_MAX
                or any(n >= _LANE_ERR_COUNTER_MAX for n in self.lane_errors))

    @property
    def bad_lanes(self) -> tuple[int, ...]:
        return tuple(lane for lane, n in enumerate(self.lane_errors) if n)


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
    pcie_width_cur: int  # negotiated link width
    pcie_width_cap: int  # widest attainable: min(GPU max, upstream port max); 0 = unknown
    error: str = ""          # non-empty → row is in error state
    uuid: str = ""           # full NVML UUID, e.g. "GPU-1a2b3c4d-…" (stable across slot rotations)
    compute_cap: str = ""    # CUDA compute capability, e.g. "8.6" for sm_86
    pci_root: str = ""       # root complex domain hex, e.g. "15" — GPUs sharing a root are on the same physical slot
    power_default_w: float = 0.0  # factory (VBIOS) default power limit
    power_min_w: float = 0.0      # lowest settable power limit
    power_max_w: float = 0.0      # highest settable power limit
    link: PcieLinkHealth | None = None  # None = not polled or unsupported by the GPU

    @property
    def pcie_width(self) -> str:
        """Negotiated / attainable link width, e.g. ``"4/4"``."""
        if not self.pcie_width_cap:
            return "?/?"
        return f"{self.pcie_width_cur}/{self.pcie_width_cap}"

    @property
    def pcie_width_degraded(self) -> bool:
        """True when the link trained narrower than both ends support (failed lanes)."""
        return 0 < self.pcie_width_cur < self.pcie_width_cap

    @property
    def short_uuid(self) -> str:
        """First 8 hex chars of the UUID — compact display identifier."""
        return self.uuid.removeprefix("GPU-")[:8]

    @property
    def power_limit_delta_w(self) -> float:
        """Current power limit minus the factory default (negative = capped)."""
        if self.power_default_w <= 0:
            return 0.0
        return self.power_limit_w - self.power_default_w

    @property
    def power_limit_is_default(self) -> bool:
        return abs(self.power_limit_delta_w) <= _POWER_LIMIT_TOLERANCE_W

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


def fmt_power_limit_w(row: GpuRow, watts: float) -> str:
    """*watts* as a power limit for *row*'s card: ``↓`` below / ``↑`` above its factory default."""
    text = f"{watts:.0f}"
    if row.power_default_w <= 0 or abs(watts - row.power_default_w) <= _POWER_LIMIT_TOLERANCE_W:
        return text
    if watts < row.power_default_w:
        return f"[cyan]{text}↓[/]"
    return f"[magenta]{text}↑[/]"


def _fmt_power(row: GpuRow) -> str:
    pct = (row.power_w / row.power_limit_w * 100) if row.power_limit_w > 0 else 0
    color = _color(pct, 60, 90)
    return f"[{color}]{row.power_w:.0f}[/]/{fmt_power_limit_w(row, row.power_limit_w)}W"


def _fmt_width(row: GpuRow) -> str:
    """Link width; red when lanes failed to train (narrower than both ends support)."""
    return f"[red]{row.pcie_width}[/]" if row.pcie_width_degraded else row.pcie_width


def _fmt_count(n: int) -> str:
    return f"{n / 1000:.0f}k" if n >= 10_000 else f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def _fmt_link(row: GpuRow) -> str:
    """PCIe link error total (since driver load) plus the lanes reporting errors."""
    link = row.link
    if link is None:
        return "-"
    if not link.total:
        return "[green]ok[/]"
    color = "red" if link.total >= _LINK_ERR_BAD else "yellow"
    text = _fmt_count(link.total) + ("+" if link.saturated else "")
    if link.bad_lanes:
        text += " L" + ",".join(map(str, link.bad_lanes))
    return f"[{color}]{text}[/]"


def _sysfs_device(bus_addr: str) -> Path:
    """sysfs path for an NVML busId: ``"00000000:16:00.0"`` → ``/sys/bus/pci/devices/0000:16:00.0``."""
    # NVML uses an 8-char domain and may return uppercase hex; sysfs uses 4-char lowercase.
    return Path(f"/sys/bus/pci/devices/{bus_addr[4:].lower()}")


@lru_cache(maxsize=None)
def _upstream_max_width(bus_addr: str) -> int:
    """Max link width of the PCIe port a GPU is attached to (e.g. 4 on a bifurcated x4 port).

    Returns 0 when unknown (no port above the device, or sysfs unreadable).
    Topology is fixed while the system runs, so the result is cached.
    """
    try:
        port = _sysfs_device(bus_addr).resolve().parent
        return int((port / "max_link_width").read_text())
    except (OSError, ValueError):
        return 0


# busId → (error totals at the last per-lane read, per-lane counts).  Lane
# counters only move alongside the totals, so they are re-read only on change.
_lane_cache: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}


def _read_link_health(handle, bus_addr: str, width_max: int) -> PcieLinkHealth | None:
    """Read the GPU's PCIe error counters; per-lane counts only when the totals change.

    The four totals cost about as much as one other NVML call; the per-lane
    breakdown is ~10x that, so it is read only for links with new errors.
    """
    nvml = _nvml
    assert nvml is not None
    try:
        totals = nvml.nvmlDeviceGetFieldValues(handle, [
            nvml.NVML_FI_DEV_PCIE_COUNT_RECEIVER_ERROR,
            nvml.NVML_FI_DEV_PCIE_COUNT_BAD_TLP,
            nvml.NVML_FI_DEV_PCIE_COUNT_BAD_DLLP,
            nvml.NVML_FI_DEV_PCIE_REPLAY_COUNTER,
        ])
    except nvml.NVMLError:
        return None
    if any(v.nvmlReturn != nvml.NVML_SUCCESS for v in totals):
        return None
    counts = tuple(int(v.value.ullVal) for v in totals)
    rx, tlp, dllp, replays = counts
    lanes: tuple[int, ...] = ()
    if any(counts):
        cached = _lane_cache.get(bus_addr)
        if cached and cached[0] == counts:
            lanes = cached[1]
        else:
            try:
                per_lane = nvml.nvmlDeviceGetFieldValues(
                    handle, [(nvml.NVML_FI_DEV_PCIE_COUNT_LANE_ERROR, lane) for lane in range(width_max)]
                )
                lanes = tuple(int(v.value.ullVal) if v.nvmlReturn == nvml.NVML_SUCCESS else 0
                              for v in per_lane)
                _lane_cache[bus_addr] = (counts, lanes)
            except nvml.NVMLError:
                pass
    return PcieLinkHealth(receiver_errors=rx, bad_tlp=tlp, bad_dllp=dllp,
                          replays=replays, lane_errors=lanes)


def _pci_root(bus_addr: str) -> str:
    """Resolve the PCIe root complex domain for a GPU via sysfs.

    *bus_addr* is the NVML busId string, e.g. ``"00000000:16:00.0"``.
    Returns the root complex hex string (e.g. ``"15"``), or ``""`` on failure.
    The sysfs symlink path contains ``pci0000:XX`` where XX is the root.
    """
    try:
        target = _sysfs_device(bus_addr).resolve().as_posix()  # e.g. .../pci0000:15/0000:15:01.0/0000:16:00.0
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


def poll_gpus(link_health: bool = False) -> list[GpuRow]:
    """Return one GpuRow per GPU via direct NVML calls (pynvml).

    *link_health* also reads the PCIe error counters into ``GpuRow.link``.
    """
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
            power_default = nvml.nvmlDeviceGetPowerManagementDefaultLimit(handle) / 1000.0
            power_min_mw, power_max_mw = nvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)

            gen_cur   = nvml.nvmlDeviceGetCurrPcieLinkGeneration(handle)
            gen_max   = nvml.nvmlDeviceGetMaxPcieLinkGeneration(handle)
            width_cur = nvml.nvmlDeviceGetCurrPcieLinkWidth(handle)
            width_max = nvml.nvmlDeviceGetMaxPcieLinkWidth(handle)

            cap_major, cap_minor = nvml.nvmlDeviceGetCudaComputeCapability(handle)
            port_width = _upstream_max_width(bus_id)

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
                pcie_width_cur=width_cur,
                pcie_width_cap=min(width_max, port_width) if port_width else width_max,
                uuid=uuid_raw,
                compute_cap=f"{cap_major}.{cap_minor}",
                pci_root=_pci_root(bus_id),
                power_default_w=power_default,
                power_min_w=power_min_mw / 1000.0,
                power_max_w=power_max_mw / 1000.0,
                link=_read_link_health(handle, bus_id, width_max) if link_health else None,
            ))
        except nvml.NVMLError as exc:
            rows.append(GpuRow(
                index=i,
                bus_id="??",
                name="??",
                power_w=0, power_limit_w=0,
                mem_used_mib=0, mem_total_mib=0,
                util_pct=0, temp_c=0,
                pcie_gen="?/?", pcie_width_cur=0, pcie_width_cap=0,
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
                    _fmt_width(row),
                )
