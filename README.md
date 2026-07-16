# ai-server-tools

> [!WARNING]
> **This project is under active development and is not yet stable.** Interfaces, data formats, and configuration may change without notice. Not recommended for production use.

> [!NOTE]
> **Note: Much of the code in this repository is AI-generated under close human direction. Design decisions are human-driven.**

`ai-server-tools` is a small toolkit for operating a local NVIDIA multi-GPU AI machine. It combines a keyboard-driven launcher for browsing models and starting inference servers with a standalone monitor for watching VRAM, utilisation, thermals, power, and PCIe layout in real time. The goal is to make day-to-day server bring-up, model switching, and hardware visibility faster and less error-prone.

## Tools

### [Launcher](launcher/README.md)
A keyboard-driven Textual TUI for browsing a model catalog, selecting a serving backend, choosing GPUs, and launching inference servers such as vLLM, llama.cpp, tabbyAPI / ExLlamaV2, SGLang, and ComfyUI. It keeps the machine-specific launch details in one place, with live VRAM readout, per-model/per-server test-result tracking, remote-to-local caching, and in-TUI quantization.

![Launcher screenshot](.github/assets/launcher.png)

### [GPU Monitor](gpu_monitor/README.md)
A full-screen live dashboard for seeing what every NVIDIA GPU in the machine is doing right now. It shows per-GPU VRAM usage, compute utilisation, temperature, power, and PCIe topology, polling quickly under load and backing off at idle so you can watch a model load or serve requests without opening multiple tools.

![GPU Monitor screenshot](.github/assets/gpumonitor.png)

## Requirements

- Linux, NVIDIA GPUs with a compatible driver (NVML required)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda

## Shared setup

Both tools use a single shared conda environment defined at the repo root:

```bash
conda env create -f ~/ai-server-tools/environment.yml
```

This installs Python 3.11, `textual`, and `nvidia-ml-py`. The AI model servers (vLLM, tabbyAPI, etc.) each have their own separate conda environments — see [launcher/README.md](launcher/README.md) for details.

For IDE type-checking (Pyright / Pylance), copy the example config and set your conda path:

```bash
cp pyrightconfig.json.example pyrightconfig.json
# Edit pyrightconfig.json and set venvPath to your miniconda3/envs directory
```

For Copilot context, copy the system instructions template and fill in your machine details:

```bash
cp .github/instructions/system-local.instructions.md.example .github/instructions/system-local.instructions.md
# Edit system-local.instructions.md with your host, GPU specs, and storage paths
```

## Quick start

```bash
# Launcher
cd ~/ai-server-tools
~/miniconda3/envs/ai-tools/bin/python -m launcher

# GPU Monitor (run alongside the launcher in a separate terminal)
cd ~/ai-server-tools
~/miniconda3/envs/ai-tools/bin/python -m gpu_monitor
```

Or with convenience scripts — see each tool's README.

## Repo layout

```
environment.yml          Shared conda environment (launcher + gpu_monitor)
launcher/                TUI model launcher — see launcher/README.md
gpu_monitor/             Live GPU stats — see gpu_monitor/README.md
user_data/               Gitignored — personal local data (caches, audit log)
logs/                    Gitignored — server session logs
```

## License

This project is licensed under the [MIT License](LICENSE.md).
