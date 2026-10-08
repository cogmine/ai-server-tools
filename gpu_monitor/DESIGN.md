# GPU Monitor — Design Document

## Purpose

A standalone full-screen Textual TUI that shows live per-GPU statistics for all NVIDIA GPUs in the system. It is intended to be run in a second terminal alongside the launcher — watching VRAM fill as a large model loads, or monitoring utilisation and temperature during inference.

---

## Design Principles

**Minimal retained state.** The only state carried between refreshes is the previous poll's `GpuRow` list (used solely to detect significant change for adaptive rate control) and a stable-poll counter. There is no aggregation, no history, no alerting. Every refresh replaces the table entirely. This keeps the implementation simple and avoids drift between display state and hardware state.

**No dependency on launcher screens or config.** `gpu_monitor` imports exactly one thing from the `launcher` package: `launcher.gpu`. It does not read `config.toml`, does not know about models, and does not import Textual screens from the launcher. The two applications share polling logic but are otherwise fully independent.

**Why share `launcher.gpu` rather than duplicate it?** The formatting helpers (`_fmt_mem`, `_fmt_temp`, etc.) and the `GpuRow` dataclass would otherwise need to be kept in sync. A single source prevents divergence in colour thresholds, number formatting, and column ordering.

---

## Shared Dependency: `launcher.gpu`

`launcher/gpu.py` contains:

- `GpuRow` — dataclass holding one GPU's snapshot: index, bus_id, pci_root, full UUID (`short_uuid` = first 8 hex chars for display), PCIe gen/width, memory used/total, utilisation, temperature, power used/limit, power default/min/max limits, error flag. `pci_root` is the root complex domain hex (e.g. `"15"`) resolved from sysfs — GPUs sharing a root complex sit on the same physical PCIe slot (possibly via bifurcation). `power_limit_delta_w` = current limit − factory default (negative = capped).
- `poll_gpus() → list[GpuRow]` — queries all GPUs via direct NVML calls (pynvml). On per-device error sets `row.error` and fills what it can. Returns `[]` if NVML is unavailable. Resolves `pci_root` via `/sys/bus/pci/devices/` symlinks (no root required).
- `assign_slot_groups(rows) → dict[str, int]` — maps each *shared* `pci_root` (2+ GPUs) to a 1-based group number ordered by bus_id. Solitary roots are omitted so the UI shows "-" for ungrouped GPUs. Used by the GPU monitor to display a "Grp" column showing which GPUs share a physical slot.
- Formatting helpers: `_fmt_mem`, `_fmt_util`, `_fmt_temp`, `_fmt_power` — each returns a Textual-markup string with colour thresholds. `_fmt_power` marks a non-default power limit (`↓` cyan = capped, `↑` magenta = raised).

**Why pynvml over nvidia-smi subprocess?** Each `subprocess` spawn costs 5–50 ms (fork + exec + CSV parse). At the fast polling rate of 0.1 s that is 5–50% of the entire interval just spawning a process. pynvml calls the same underlying NVML C library directly; the cost per poll is sub-millisecond. pynvml also returns typed values (ints/floats), eliminating all string parsing and `[N/A]` edge-case handling.

---

## Polling Architecture

```
GpuMonitorApp.on_mount
    └─ _refresh()                   — immediate first poll

_refresh() [@work(thread=True)]
    └─ poll_gpus()                  — NVML calls via pynvml (non-blocking, sub-ms)
    └─ call_from_thread(_update)    — safe UI update from worker thread

_update(rows)
    └─ DataTable.clear() + add_row per GPU
    └─ update summary Label
    └─ _is_active(rows)             — compare against _prev_rows
    └─ _schedule_next(delay)        — one-shot timer → _refresh()
```

`@work(thread=True)` runs `_refresh` in a thread pool managed by Textual. `call_from_thread` marshals the result back to the UI thread. This ensures the NVML poll never blocks the event loop and the UI stays responsive during polling.

**One-shot timer chain instead of `set_interval`**: each `_update` call schedules the *next* `_refresh` via `set_timer`. This guarantees sequential polling with no overlap — a new poll is only started after the previous one has completed and the UI has been updated. It also makes the interval trivially variable between calls.

### Adaptive refresh rate

| State | Interval | Condition |
|---|---|---|
| Active | 0.1 s (`POLL_FAST_S`) | any GPU has `util_pct > 0` |

**Why 0.1 s and not faster?** NVML's utilization counter is a hardware-sampled value updated by the GPU at ~167 ms intervals. Polling faster than that returns the same utilization value repeatedly. Memory (`mem_used_mib`) is real-time, and power (~10–50 ms) and temperature (~1–2 s) have their own rates, but utilization is the most watched metric and sets the practical floor. 0.1 s is already right at that limit. Going significantly below it wastes CPU and floods the Textual event loop with UI updates that contain no new data.
| Idle | 1.0 s (`POLL_SLOW_S`) | all GPUs have `util_pct == 0` |

**Why activity-based rather than rate-of-change?** A direct activity check (`util_pct > 0`) is a clear, unambiguous signal that the GPU is doing work right now and the display should be responsive. Rate-of-change detection required tracking previous values, delta thresholds, and a hysteresis counter to avoid flapping — all complexity that isn't needed when the underlying signal (utilisation) is already binary in practice.

**Status bar indicator**: a `⚡` symbol appears in the summary bar while polling at the fast rate.

**Why `DataTable.clear()` on every refresh rather than updating rows in place?** GPU count is fixed once the machine boots, so row count never changes. A full clear + re-add is simpler than per-cell updates and eliminates any possibility of stale values surviving a query error.

---

## Column Layout

Static identity/topology columns are listed first so they never shift while dynamic metrics change:

| Column | Type | Notes |
|---|---|---|
| `#` | Static | NVML device index — used for CUDA_VISIBLE_DEVICES ordering |
| `ID` | Static | First 8 hex chars of UUID — stable across PCIe slot changes (slot can change after reboot if other cards are added/removed) |
| `Grp` | Static | Physical slot group — GPUs with the same group number share a root complex (same physical x16 slot, possibly bifurcated). Derived from sysfs root complex mapping. Useful for diagnosing missing GPUs: a group with fewer cards than expected pinpoints the problem slot. |
| `Slot` | Static | PCIe bus ID — useful for physical slot identification |
| `SM` | Static | CUDA compute capability (e.g. `8.6` for RTX 3090). Used by the launcher to gate hw-incompatible models. |
| `Gen` | Static | PCIe link generation current/max — diagnoses PCIe downgrade (important on x570 boards where x16 slots share bandwidth) |
| `Width` | Static | PCIe link width current/max — flags accidentally x8-slotted cards |
| `Memory` | Dynamic | Block-char progress bar + used/total GB; green → yellow → red at 60%/90% |
| `Util` | Dynamic | Compute utilisation %; colour-coded |
| `Temp` | Dynamic | Core temperature °C; colour-coded |
| `Power` | Dynamic | Current draw / power limit watts; draw colour-coded vs. limit. Limit marked `↓`/`↑` when it differs from the factory default, so manually capped cards stand out |

The summary bar at the bottom aggregates total VRAM used vs. total across all GPUs — the single number most relevant when loading a large model. It also shows total power draw vs. the sum of power limits (the worst-case PSU load), and a per-slot GPU count (e.g. `Slots: 1:1 2:2 3:3 4:3 5:1 6:1 7:1`) so a missing GPU is immediately visible as a group with fewer GPUs than expected.

---

## Error Handling

If NVML raises `NVMLError` for a specific GPU (e.g. during driver recovery), `poll_gpus` sets `row.error = True` and `_update` renders the row with an "ERR" marker for dynamic columns rather than crashing. The error count is appended to the summary bar so failures are visible even when the table is scrolled.

---

## Separation from the Launcher

`gpu_monitor` was deliberately kept as a separate `python -m gpu_monitor` invocation rather than an embedded panel inside the launcher. Reasons:

1. **Terminal flexibility.** GPU stats are most useful in a wide side-by-side layout. Embedding them in the launcher would require a fixed split that doesn't suit all terminal sizes.
2. **Lifecycle independence.** The launcher exits when a server launches; GPU stats should outlive it.
3. **Simplicity.** The launcher does not need to own two independent refresh loops.
