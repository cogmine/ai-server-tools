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

from launcher.gpu import GpuRow, poll_gpus, _fmt_mem, _fmt_util, _fmt_temp, _fmt_power


# Static identity/topology columns come first, dynamic metrics after
COLUMNS = ("#", "ID", "Slot", "SM", "Gen", "Width", "Memory", "Util", "Temp", "Power")

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
        table = self.query_one(DataTable)
        table.add_columns(*COLUMNS)
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

        total_mem_used = 0.0
        total_mem_total = 0.0
        error_count = 0

        for row in rows:
            if row.error:
                error_count += 1
                table.add_row(
                    str(row.index), row.uuid or row.bus_id, row.bus_id,
                    row.compute_cap or "-",
                    row.pcie_gen, row.pcie_width,
                    f"[red]!ERROR: {row.error}[/]",
                    "-", "-", "-",
                )
            else:
                total_mem_used  += row.mem_used_mib
                total_mem_total += row.mem_total_mib
                table.add_row(
                    str(row.index),
                    row.uuid,
                    row.bus_id,
                    row.compute_cap or "-",
                    row.pcie_gen,
                    row.pcie_width,
                    _fmt_mem(row),
                    _fmt_util(row),
                    _fmt_temp(row),
                    _fmt_power(row),
                )

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
            status.update(
                f"[dim]VRAM:[/]  [bold]{total_gb:.0f}[/] GB total"
                f"  ·  {used_gb:.0f} GB used"
                f"  ·  [{free_color}][bold]{free_gb:.0f}[/] GB free[/]"
                f"{err_str}"
                f"{live_str}"
            )
        else:
            status.update("[red]NVML not available[/]")
