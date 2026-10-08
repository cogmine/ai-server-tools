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

import os
import re
import subprocess
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
    error: str = ""                   # non-empty → no estimate available


class FitProbe(ABC):
    """Server-specific memory-fit estimator."""

    @abstractmethod
    def available(self, env: Environment) -> bool:
        """True when *env* provides what the probe needs."""

    @abstractmethod
    def estimate(
        self, env: Environment, model: AIModel, params: dict[str, Any], cuda_devices: str,
        cancelled: Callable[[], bool],
    ) -> FitEstimate:
        """Estimate memory use for a launch of *model* with resolved builder *params*.

        Polls *cancelled* while waiting and abandons the estimate when it returns True.
        """


class LlamaCppFitProbe(FitProbe):
    """Runs llama-fit-params (same build as llama-server) with the launch's model/context arguments.

    llama.cpp's own fitter (also used by llama-server's default --fit) loads only the model
    metadata, so the estimate is exact for every architecture (sliding-window, MLA, hybrid)
    and accounts for compute buffers and memory already in use on the devices.
    Not covered: --mmproj vision projectors, MTP draft heads, and anything passed via extra_flags.
    """

    executable = "llama-fit-params"
    timeout_s = 120
    poll_s = 0.2

    # stdout contract: the fitted CLI arguments, e.g. "-c 655360 -ngl -1" (-1 = all layers on GPU).
    _NGL_RE = re.compile(r"(?:^|\s)-ngl\s+(-?\d+)")
    # Log lines (verbosity ≥ 4).  Parsed best-effort: when the format changes, only the sizes are lost.
    _PROJECTED_RE = re.compile(r"projected to use (\d+) MiB of device memory vs\. (\d+) MiB of free device memory")
    _INITIAL_MARK = "projected memory use with initial parameters"
    # "| - CUDA0 (RTX 3090) | 24372 = 24108 + (19844 = 8415 + 8160 + 3268) + -19579 |"
    _DEVICE_ROW_RE = re.compile(r"\|\s*- CUDA\d+[^|]*\|[^(]*\(\s*\d+\s*=\s*(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)\s*\)")
    _ERROR_RE = re.compile(r"^\S+ E (.+)$", re.MULTILINE)

    def available(self, env: Environment) -> bool:
        return env.executable(self.executable).is_file()

    def estimate(
        self, env: Environment, model: AIModel, params: dict[str, Any], cuda_devices: str,
        cancelled: Callable[[], bool],
    ) -> FitEstimate:
        cmd = [str(env.executable(self.executable)), *llamacpp_model_args(model, params), "-lv", "4"]
        try:
            proc = subprocess.Popen(
                cmd,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": cuda_devices},
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
        except OSError as e:
            return FitEstimate(error=str(e))
        deadline = time.monotonic() + self.timeout_s
        while True:
            try:
                stdout, stderr = proc.communicate(timeout=self.poll_s)
                break
            except subprocess.TimeoutExpired:
                if cancelled() or time.monotonic() > deadline:
                    proc.kill()
                    proc.communicate()
                    return FitEstimate(error="cancelled" if cancelled() else f"{self.executable} timed out")
        if proc.returncode != 0:
            errors = self._ERROR_RE.findall(stderr)
            return FitEstimate(error=errors[0] if errors else f"{self.executable} exited with code {proc.returncode}")

        ngl_match = self._NGL_RE.search(stdout)
        if ngl_match is None:
            return FitEstimate(error=f"unexpected {self.executable} output: {stdout.strip()!r}")
        fitted_ngl = int(ngl_match.group(1))
        projected = self._PROJECTED_RE.search(stderr)
        projected_mib, free_mib = (int(projected.group(1)), int(projected.group(2))) if projected else (None, None)

        if str(params["n_gpu_layers"]) == "auto":
            fits = fitted_ngl == -1
        else:
            # Layer count set explicitly: the fitter leaves it alone, so compare totals instead.
            fits = projected_mib is not None and free_mib is not None and projected_mib <= free_mib

        # The first breakdown table describes the requested (initial) parameters.
        initial = stderr.split(self._INITIAL_MARK, 1)[0]
        rows = [tuple(map(int, row)) for row in self._DEVICE_ROW_RE.findall(initial)]
        model_mib, context_mib, compute_mib = (sum(col) for col in zip(*rows)) if rows else (None, None, None)

        return FitEstimate(
            fits=fits,
            gpu_layers=None if fits or fitted_ngl < 0 else fitted_ngl,
            projected_mib=projected_mib,
            free_mib=free_mib,
            model_mib=model_mib,
            context_mib=context_mib,
            compute_mib=compute_mib,
        )


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
) -> FitEstimate | None:
    """Estimate whether launching *model* with these settings fits on *cuda_devices*.  Blocking (seconds)."""
    probe = fit_probe_for(server, env)
    if probe is None:
        return None
    params = launch_params(server, model, param_overrides=param_overrides)
    return probe.estimate(env, model, params, cuda_devices, cancelled)
