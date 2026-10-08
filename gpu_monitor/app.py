"""
gpu_monitor/app.py — Full-screen live GPU stats Textual app.
Shares the GpuRow / poll_gpus / formatting helpers from launcher.gpu.
"""
from __future__ import annotations

import time
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import DataTable, Footer, Header, Label
from textual import work

from launcher.gpu import GpuRow, poll_gpus, assign_slot_groups, _fmt_mem, _fmt_util, _fmt_temp, _fmt_power


# Static identity/topology columns come first, dynamic metrics after
COLUMNS = ("#", "ID", "Grp", "Slot", "SM", "Gen", "Width", "Memory", "Util", "Temp", "Power")

# ---------------------------------------------------------------------------
# Adaptive polling parameters
# ---------------------------------------------------------------------------

POLL_FAST_S   = 0.1  # interval while any GPU is active (~167 ms hardware util sampling rate)
POLL_SLOW_S   = 1.0  # interval while all GPUs are idle
ACTIVE_COOL_S = 2.0  # hold fast rate for this many seconds after activity stops


class GpuMonitorApp(App):
    """Live GPU stats — full screen, adaptive refresh rate."""

    CSS_PATH = Path(__file__).parent / "gpu_monitor.tcss"
    TITLE = "GPU Monitor"
    BINDINGS = [
        Binding("q,escape", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(show_cursor=False, id="gpu-table")
        yield Label("", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        self._last_active_at: float = 0.0
        self._slot_groups: dict[str, int] = {}  # pci_root → group number
        self._grp_col_visible: bool = True   # determined on first render with data
        self._columns_ready: bool = False    # True once columns have been added
        self._refresh()

    def _schedule_next(self, delay: float) -> None:
        self.set_timer(delay, self._refresh)

    @work(thread=True)
    def _refresh(self) -> None:
        try:
            rows = poll_gpus()
        except Exception:
            rows = []
        self.app.call_from_thread(self._update, rows)

    def _is_active(self, rows: list[GpuRow]) -> bool:
        """Return True if any GPU is currently active (util > 0)."""
        return any(row.util_pct > 0 for row in rows if not row.error)

    def _update(self, rows: list[GpuRow]) -> None:
        # Determine the next interval first so _schedule_next can always fire
        # via finally, even if the UI update raises.
        now = time.monotonic()
        if self._is_active(rows):
            self._last_active_at = now
        fast_mode = (now - self._last_active_at) < ACTIVE_COOL_S
        next_interval = POLL_FAST_S if fast_mode else POLL_SLOW_S
        try:
            self._render(rows, fast_mode)
        finally:
            self._schedule_next(next_interval)

    def _render(self, rows: list[GpuRow], fast_mode: bool = False) -> None:
        table = self.query_one(DataTable)
        table.clear()

        # Assign slot groups once (stable across refreshes as long as GPU set is constant)
        if not self._slot_groups and rows:
            self._slot_groups = assign_slot_groups(rows)
        groups = self._slot_groups

        # Set up columns on the first render that has data.  Grp is omitted when
        # all GPUs would show the same value (all ungrouped or all in one group).
        if not self._columns_ready and rows:
            grp_values = {groups.get(r.pci_root, 0) for r in rows}
            self._grp_col_visible = len(grp_values) > 1
            cols = [c for c in COLUMNS if c != "Grp" or self._grp_col_visible]
            table.add_columns(*cols)
            self._columns_ready = True

        total_mem_used = 0.0
        total_mem_total = 0.0
        total_power = 0.0
        total_power_limit = 0.0
        error_count = 0
        group_counts: dict[int, int] = {}  # group → GPU count

        for row in rows:
            grp = groups.get(row.pci_root, 0)
            grp_str = str(grp) if grp else "-"
            group_counts[grp] = group_counts.get(grp, 0) + 1
            if row.error:
                error_count += 1
                vals = [str(row.index), row.short_uuid or row.bus_id]
                if self._grp_col_visible:
                    vals.append(grp_str)
                vals += [
                    row.bus_id, row.compute_cap or "-",
                    row.pcie_gen, row.pcie_width,
                    f"[red]!ERROR: {row.error}[/]",
                    "-", "-", "-",
                ]
                table.add_row(*vals)
            else:
                total_mem_used  += row.mem_used_mib
                total_mem_total += row.mem_total_mib
                total_power       += row.power_w
                total_power_limit += row.power_limit_w
                vals = [str(row.index), row.short_uuid]
                if self._grp_col_visible:
                    vals.append(grp_str)
                vals += [
                    row.bus_id, row.compute_cap or "-",
                    row.pcie_gen, row.pcie_width,
                    _fmt_mem(row), _fmt_util(row), _fmt_temp(row), _fmt_power(row),
                ]
                table.add_row(*vals)

        # Update summary status bar
        status = self.query_one("#status-bar", Label)
        if rows:
            used_gb  = total_mem_used  / 1024
            total_gb = total_mem_total / 1024
            free_gb  = total_gb - used_gb
            free_pct = (free_gb / total_gb * 100) if total_gb else 0
            free_color = "green" if free_pct > 50 else ("yellow" if free_pct > 20 else "red")
            err_str = f"  [red]{error_count} error(s)[/]" if error_count else ""
            live_str = "  ·  [bold yellow]⚡[/]" if fast_mode else ""
            grp_str = "  ·  [dim]Slots:[/] " + " ".join(
                f"{g}:{c}" for g, c in sorted(group_counts.items()) if g
            ) if group_counts else ""
            status.update(
                f"[dim]VRAM:[/]  [bold]{total_gb:.0f}[/] GB total"
                f"  ·  {used_gb:.0f} GB used"
                f"  ·  [{free_color}][bold]{free_gb:.0f}[/] GB free[/]"
                f"  ·  [dim]Power:[/] {total_power / 1000:.1f}/{total_power_limit / 1000:.1f} kW"
                f"{err_str}"
                f"{grp_str}"
                f"{live_str}"
            )
        else:
            status.update("[red]NVML not available[/]")
