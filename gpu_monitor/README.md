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
| `Gen` | PCIe link generation: current / max |
| `Width` | PCIe link width: current / max |
| `Memory` | Bar + used / total GB (green → yellow → red at 60% / 90%) |
| `Util` | GPU compute utilisation % |
| `Temp` | Core temperature °C |
| `Power` | Current / TDP watts |

A summary bar at the bottom shows total VRAM used vs. total across all GPUs.

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
