"""
app.py — Top-level Textual App.

Navigation entry point is AITypeScreen (screens.py), which drives the
flow: AI Type → Model → Model settings → Server → [Environment] → [Quant] → GPUs → ConfirmLaunch.
"""
from __future__ import annotations

import fcntl
import os
import shutil
import signal
import struct
import sys
import termios
import threading
from pathlib import Path
from typing import ClassVar

from textual.app import App
from textual.binding import Binding

from launcher.data import AI_MODELS, SERVERS
from launcher.environments import ENVIRONMENTS
from launcher.launch import LaunchJob, QuantizeJob, finalize_quantization
from launcher.models import AIModel, available_ai_types, uses_hub_flow
from launcher.screens import (
    AITypeScreen, EnvSelectScreen, ImageGenHubScreen, ModelPropertiesScreen, ModelSelectScreen,
    ServerSelectScreen,
)


class ModelLauncherApp(App):
    """Model Server Launcher — Textual TUI."""

    CSS_PATH = Path(__file__).parent / "model_launcher.tcss"
    TITLE = "Model Server Launcher"
    BINDINGS: ClassVar = [
        Binding("q", "quit_app", "Quit", show=False),
    ]

    def __init__(
        self,
        preselect: "LaunchJob | None" = None,
        preselect_model: "AIModel | None" = None,
    ) -> None:
        super().__init__()
        self._preselect = preselect
        self._preselect_model = preselect_model

    def on_mount(self) -> None:
        if self._preselect is not None:
            job = self._preselect
            model, server = job.model, job.server
            # Navigate straight to the server (and environment) list with the
            # launched selection focused so the user can immediately mark the result.
            # Only applicable for text-gen (model is always set); image-gen hub
            # flow passes model=None so we fall through to normal startup.
            if model is not None:
                ai_type = next(iter(model.ai_types))
                self.push_screen(ModelSelectScreen(ai_type, preselect_model=model))
                self.push_screen(ModelPropertiesScreen(model))
                self.push_screen(ServerSelectScreen(model, preselect_server_id=server.id))
                if len(ENVIRONMENTS.for_server(server.id)) > 1:
                    self.push_screen(EnvSelectScreen(model, server, preselect_env=job.env.name))
                return
        if self._preselect_model is not None:
            ai_type = next(iter(self._preselect_model.ai_types))
            self.push_screen(ModelSelectScreen(ai_type, preselect_model=self._preselect_model))
            return
        types = available_ai_types(AI_MODELS)
        if len(types) == 1:
            if uses_hub_flow(SERVERS, types[0]):
                self.push_screen(ImageGenHubScreen(types[0]))
            else:
                self.push_screen(ModelSelectScreen(types[0]))
        else:
            self.push_screen(AITypeScreen())

    def action_quit_app(self) -> None:
        self.exit(None)

    def action_exit_clean(self) -> None:
        self.exit(None)


def _set_pty_size(master_fd: int) -> None:
    """Stamp the PTY with the current terminal dimensions."""
    try:
        cols, rows = shutil.get_terminal_size()
        fcntl.ioctl(master_fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
    except Exception:
        pass


def _tee_output(master_fd: int, log_path: Path) -> None:
    """Read from PTY master, write to terminal and log simultaneously.

    Uses os.read() directly on the fd.  When the child exits the PTY raises
    OSError (EIO on Linux) which we treat as clean EOF.
    """
    with open(log_path, "ab") as log_f:
        while True:
            try:
                chunk = os.read(master_fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            log_f.write(chunk)
            log_f.flush()


def run(no_menu: bool = False, default: int | None = None) -> None:
    """Entry point called from __main__.py."""
    preselect: "LaunchJob | None" = None
    preselect_model: "AIModel | None" = None
    while True:
        app = ModelLauncherApp(preselect=preselect, preselect_model=preselect_model)
        result = app.run()
        preselect = None  # consumed; clear for any subsequent loops
        preselect_model = None

        quantize_job = result if isinstance(result, QuantizeJob) else None
        launch_job = result if isinstance(result, LaunchJob) else None
        if quantize_job is not None:
            factory = quantize_job.factory
        elif launch_job is not None:
            factory = launch_job.factory
        else:
            # User quit the menu without launching — exit to bash.
            sys.exit(0)
        proc, log_path, master_fd = factory()
        print(f"Logging to: {log_path}", flush=True)
        tee = threading.Thread(target=_tee_output, args=(master_fd, log_path), daemon=True)
        tee.start()
        # Forward terminal resize events to the PTY so width-sensitive output
        # (progress bars etc.) stays correct if the user resizes mid-run.
        def _winch(signum: int, frame: object) -> None:
            _set_pty_size(master_fd)
        old_winch = signal.signal(signal.SIGWINCH, _winch)
        user_interrupted = False
        try:
            proc.wait()
        except KeyboardInterrupt:
            user_interrupted = True
            proc.terminate()
            proc.wait()
        finally:
            signal.signal(signal.SIGWINCH, old_winch)
        tee.join()
        os.close(master_fd)

        # Append exit status footer so crashes are always documented in the log.
        # Bash uses 128+N for a child killed by signal N.
        rc = proc.returncode
        with open(log_path, "ab") as log_f:
            log_f.write(b"\n")
            if rc is not None and rc > 128:
                sig_num = rc - 128
                try:
                    sig_name = signal.Signals(sig_num).name
                except ValueError:
                    sig_name = f"signal {sig_num}"
                log_f.write(f"# exit: killed by {sig_name} ({sig_num})\n".encode())
            else:
                log_f.write(f"# exit: code {rc}\n".encode())

        if user_interrupted:
            # Ctrl-C on a server launch: reopen the menu with model+server+env still
            # selected so the user can immediately mark the test result.
            # Ctrl-C on a quantize job, or a stage_all_models server (model=None):
            # just reopen the top-level menu (no preselect).
            if launch_job is not None and launch_job.model is not None:
                preselect = launch_job
            continue

        if quantize_job is not None:
            if rc == 0:
                try:
                    preselect_model = finalize_quantization(quantize_job)
                    print(f"Added generated model: {preselect_model.id}", flush=True)
                except Exception as exc:
                    print(f"Failed to finalize quantized model: {exc}", file=sys.stderr)
            continue

        sys.exit(rc or 0)
