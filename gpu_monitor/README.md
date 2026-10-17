# GPU Monitor

GPU Monitor is a full-screen Textual dashboard for watching all NVIDIA GPUs in a machine in real time. It surfaces the hardware signals that matter when loading or serving models: VRAM usage, compute utilisation, temperature, power draw, and PCIe topology. The refresh rate stays fast while any GPU is active and backs off when the system is idle, so you can leave it running alongside the launcher or a server process and see changes as they happen.

![GPU Monitor screenshot](../.github/assets/gpumonitor.png)

## Requirements

- The `ai-tools` conda environment (defined in `environment.yml` at the repo root — shared with the launcher)

```bash
conda env create -f ~/ai-server-tools/environment.yml  # skip if already created
```

## Usage

```bash
cd ~/ai-server-tools
~/miniconda3/envs/ai-tools/bin/python -m gpu_monitor
```

Press `q` or `Escape` to exit.

## Display columns

| Column | Description |
|---|---|
| `#` | GPU index (NVML order — matches `CUDA_VISIBLE_DEVICES` numbering) |
| `ID` | First 8 hex chars of GPU UUID — stable across PCIe slot changes |
| `Slot` | PCIe bus ID (2-char hex) |
| `SM` | CUDA compute capability (e.g. `8.6` for RTX 3090) |
| `Gen` | PCIe link generation: current / max (current drops to 1 while idle — normal power saving) |
| `Width` | PCIe link width: negotiated / attainable (the narrower of the GPU and the port it sits on, so `4/4` on a bifurcated x4 port). **Red** when the link trained narrower — failed lanes in a riser, cable or connector |
| `Memory` | Bar + used / total GB (green → yellow → red at 60% / 90%) |
| `Util` | GPU compute utilisation % |
| `Temp` | Core temperature °C |
| `Power` | Current draw / power limit (W). The limit is marked `↓` (cyan) when capped below the card's factory default, `↑` (magenta) when raised above it |
| `Link` | PCIe link errors reported by the GPU since the driver loaded (boot): `ok`, or the error count (yellow; red at 100+) followed by the lanes reporting errors, e.g. `6.7k L2`. A `+` means a hardware counter saturated, so the true count is higher. See [PCIe link health](#pcie-link-health) |

A summary bar at the bottom shows total VRAM used vs. total across all GPUs, total power draw vs. the sum of the power limits, and the number of GPUs with PCIe link issues (degraded width or link errors).

## PCIe link health

Every corrupted PCIe packet forces a resend or a brief link retrain. The GPU counts these (receiver errors, bad TLP/DLLP CRCs, replays) and records which lanes they occurred on. On a healthy connection the counters stay at zero, even under load, so any non-zero `Link` value points at a marginal physical connection: a riser, splitter cable, connector, or poorly seated card.

- **Locate the fault**: the `L` lanes identify the failing wire pair(s). If the fault follows a cable or riser when you swap it with a known-good one, replace that part; if it stays with the slot/splitter port, the port is at fault.
- **Width red, e.g. `2/4`**: lanes failed during link training. The `Link` lanes usually show which ones.
- **Mitigation**: forcing the slot to a lower PCIe generation in BIOS (e.g. Gen3) often eliminates errors on marginal riser links at little cost for layer-split inference.
- Counters reset only when the NVIDIA driver reloads (reboot), so compare against the time since boot. Watch whether a count is still *growing* after reseating or replacing a part.

## File map

```
gpu_monitor/
  __main__.py        Entry point: python -m gpu_monitor
  app.py             Textual App + live DataTable update loop
  gpu_monitor.tcss   CSS layout
```

GPU polling logic lives in `../launcher/gpu.py` and is shared with the launcher.

## Dependency note

`gpu_monitor` imports `launcher.gpu` — both directories must sit under the same parent so that `launcher` is importable. No other launcher modules are used.
