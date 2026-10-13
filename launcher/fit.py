"""
fit.py — Pre-launch memory-fit estimates.

estimate_fit(server, env, model, cuda_devices, param_overrides) → FitEstimate | None
    Asks the server's own tooling whether a launch with the given settings fits in
    the currently free memory of the selected GPUs, without loading any weights.
    Returns None when the server has no fit probe.

Probes are registered per server id in _PROBES (mirroring launch._BUILDERS).
Adding one: subclass FitProbe and register it — the GPU selection screen picks it up.
"""
from __future__ import annotations

import dataclasses
import os
import re
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

from launcher.environments import Environment
from launcher.launch import launch_params, llamacpp_model_args
from launcher.models import AIModel, ServerSoftware


@dataclass(frozen=True)
class FitEstimate:
    """Projected device memory use for one launch configuration.  Sizes in MiB; None = not reported."""

    fits: bool = False               # the requested settings fit entirely in free GPU memory
    gpu_layers: int | None = None    # layers placed on GPU when the model must be partially offloaded
    projected_mib: int | None = None  # total device memory the launch would use
    free_mib: int | None = None       # free memory across the selected devices
    model_mib: int | None = None      # weights
    context_mib: int | None = None    # KV cache (and recurrent state)
    compute_mib: int | None = None    # compute / scratch buffers
    shortfall_mib: int | None = None  # memory to shed to fit (incl. per-device headroom); 0 = enough in total
    searching: bool = False           # provisional: the fitter is still searching for a layer placement
    search_timed_out: bool = False    # the placement search did not finish; the projection still stands
    error: str = ""                   # non-empty → no estimate available


class FitProbe(ABC):
    """Server-specific memory-fit estimator."""

    @abstractmethod
    def available(self, env: Environment) -> bool:
        """True when *env* provides what the probe needs."""

    @abstractmethod
    def estimate(
        self, server: ServerSoftware, env: Environment, model: AIModel, params: dict[str, Any],
        cuda_devices: str, cancelled: Callable[[], bool], progress: Callable[[FitEstimate], None],
    ) -> FitEstimate:
        """Estimate memory use for a launch of *model* with resolved builder *params*.

        Polls *cancelled* while waiting and abandons the estimate when it returns True.
        May call *progress* with a provisional estimate (searching=True) before returning.
        """


class LlamaCppFitProbe(FitProbe):
    """Runs llama-fit-params (same build as llama-server) with the launch's model/context arguments.

    llama.cpp's own fitter (also used by llama-server's default --fit) loads only the model
    metadata, so the estimate is exact for every architecture (sliding-window, MLA, hybrid)
    and accounts for compute buffers and memory already in use on the devices.
    Not covered: --mmproj vision projectors, MTP draft heads, and anything passed via extra_flags.

    The verdict for the requested settings is logged within seconds.  When they do not fit,
    the fitter goes on to search partial-CPU-offload placements, testing allocations one by
    one — minutes of single-core CPU for large MoE models.  The log is therefore streamed:
    the verdict is reported via *progress* as soon as it appears, and survives a search timeout.
    """

    executable = "llama-fit-params"
    timeout_s = 120
    poll_s = 0.2

    # stdout contract: the fitted CLI arguments, e.g. "-c 655360 -ngl -1" (-1 = all layers on GPU).
    _NGL_RE = re.compile(r"(?:^|\s)-ngl\s+(-?\d+)")
    # Log lines (verbosity ≥ 4).  Parsed best-effort: when the format changes, only the sizes are lost.
    _PROJECTED_RE = re.compile(r"projected to use (\d+) MiB of device memory vs\. (\d+) MiB of free device memory")
    _INITIAL_MARK = "projected memory use with initial parameters"
    # "- CUDA0 (NVIDIA GeForce RTX 3090) :  24372 total,  30396 used,  -6317 free vs. target of   1024"
    _DEVICE_FREE_RE = re.compile(r"- CUDA\d+[^:]*:\s*-?\d+ total,\s*-?\d+ used,\s*(-?\d+) free vs\. target of\s*(\d+)")
    # "| - CUDA0 (RTX 3090) | 24372 = 24108 + (19844 = 8415 + 8160 + 3268) + -19579 |"
    _DEVICE_ROW_RE = re.compile(r"\|\s*- CUDA\d+[^|]*\|[^(]*\(\s*\d+\s*=\s*(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)\s*\)")
    _ERROR_RE = re.compile(r"^\S+ E (.+)$", re.MULTILINE)

    def available(self, env: Environment) -> bool:
        return env.executable(self.executable).is_file()

    def _initial_estimate(self, log: str) -> FitEstimate | None:
        """Projection for the requested settings, once the fitter has logged it (else None).

        Mirrors llama.cpp's rule: the settings fit when every device keeps its target margin
        free; the shortfall is the summed deficit (Σ target − free), which no redistribution
        across devices can recover — only offloading to CPU can.
        """
        before, mark, after = log.partition(self._INITIAL_MARK)
        projected = self._PROJECTED_RE.search(after) if mark else None
        if projected is None:
            return None
        devices = [(int(free), int(target)) for free, target in self._DEVICE_FREE_RE.findall(after[:projected.start()])]
        if not devices:
            return None
        # The breakdown table before the mark describes the requested (initial) parameters.
        rows = [tuple(map(int, row)) for row in self._DEVICE_ROW_RE.findall(before)]
        model_mib, context_mib, compute_mib = (sum(col) for col in zip(*rows)) if rows else (None, None, None)
        return FitEstimate(
            fits=all(free >= target for free, target in devices),
            projected_mib=int(projected.group(1)),
            free_mib=int(projected.group(2)),
            model_mib=model_mib,
            context_mib=context_mib,
            compute_mib=compute_mib,
            shortfall_mib=max(0, sum(target - free for free, target in devices)),
        )

    def estimate(
        self, server: ServerSoftware, env: Environment, model: AIModel, params: dict[str, Any],
        cuda_devices: str, cancelled: Callable[[], bool], progress: Callable[[FitEstimate], None],
    ) -> FitEstimate:
        cmd = [str(env.executable(self.executable)), *llamacpp_model_args(server, model, params), "-lv", "4"]
        try:
            proc = subprocess.Popen(
                cmd,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": cuda_devices},
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
        except OSError as e:
            return FitEstimate(error=str(e))
        # Both pipes are drained continuously: the search can log tens of thousands of lines.
        out: list[str] = []
        err: list[str] = []
        readers = [
            threading.Thread(target=lambda s=stream, buf=buf: buf.extend(s), daemon=True)
            for stream, buf in ((proc.stdout, out), (proc.stderr, err))
        ]
        for r in readers:
            r.start()

        initial: FitEstimate | None = None
        deadline = time.monotonic() + self.timeout_s
        while proc.poll() is None:
            if cancelled() or time.monotonic() > deadline:
                proc.kill()
                proc.wait()
                if cancelled():
                    return FitEstimate(error="cancelled")
                if initial is not None:
                    return dataclasses.replace(initial, search_timed_out=True)
                return FitEstimate(error=f"{self.executable} timed out")
            if initial is None:
                initial = self._initial_estimate("".join(err))
                if initial is not None and not initial.fits:
                    progress(dataclasses.replace(initial, searching=True))
            time.sleep(self.poll_s)
        for r in readers:
            r.join()
        stdout, stderr = "".join(out), "".join(err)
        initial = initial or self._initial_estimate(stderr)

        if proc.returncode != 0:
            # Fails when nothing it may adjust (e.g. an explicit layer count) makes the settings
            # fit; the projection for the requested settings is still valid.
            if initial is not None:
                return initial
            errors = self._ERROR_RE.findall(stderr)
            return FitEstimate(error=errors[0] if errors else f"{self.executable} exited with code {proc.returncode}")
        ngl_match = self._NGL_RE.search(stdout)
        if ngl_match is None:
            return FitEstimate(error=f"unexpected {self.executable} output: {stdout.strip()!r}")
        fitted_ngl = int(ngl_match.group(1))
        if str(params["n_gpu_layers"]) != "auto":
            # The fitter leaves an explicit layer count alone, so the projection decides.
            return initial or FitEstimate(error=f"unrecognised {self.executable} log")
        # Auto layers: the fitter may rebalance across devices; -ngl -1 = all layers on GPU.
        fits = fitted_ngl == -1
        gpu_layers = None if fits or fitted_ngl < 0 else fitted_ngl
        if initial is None:
            return FitEstimate(fits=fits, gpu_layers=gpu_layers)
        return dataclasses.replace(initial, fits=fits, gpu_layers=gpu_layers)


_PROBES: dict[str, FitProbe] = {
    "llamacpp": LlamaCppFitProbe(),
}


def fit_probe_for(server: ServerSoftware, env: Environment) -> FitProbe | None:
    """The fit probe for *server*, when one is registered and usable in *env*."""
    probe = _PROBES.get(server.id)
    return probe if probe is not None and probe.available(env) else None


def estimate_fit(
    server: ServerSoftware,
    env: Environment,
    model: AIModel,
    cuda_devices: str,
    param_overrides: dict[str, Any] | None = None,
    cancelled: Callable[[], bool] = lambda: False,
    progress: Callable[[FitEstimate], None] = lambda _estimate: None,
) -> FitEstimate | None:
    """Estimate whether launching *model* with these settings fits on *cuda_devices*.  Blocking (seconds).

    *progress* may receive a provisional estimate (searching=True) before the final one is returned.
    """
    probe = fit_probe_for(server, env)
    if probe is None:
        return None
    params = launch_params(server, model, param_overrides=param_overrides)
    return probe.estimate(server, env, model, params, cuda_devices, cancelled, progress)
