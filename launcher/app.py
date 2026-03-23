"""
app.py — Top-level Textual App.

Navigation entry point is AITypeScreen (screens.py), which drives the
3-level flow: AI Type → Model → Server → ConfirmLaunch.
"""
from __future__ import annotations

import fcntl
import os
import shutil
import signal
import struct
import subprocess
import sys
import termios
import threading
from pathlib import Path
from typing import Callable, ClassVar, cast

from textual.app import App
from textual.binding import Binding

from launcher.data import AI_MODELS, SERVERS
from launcher.models import AIModel, ServerSoftware, available_ai_types, uses_hub_flow
from launcher.screens import AITypeScreen, ImageGenHubScreen, ModelSelectScreen, ServerSelectScreen


class ModelLauncherApp(App):
    """Model Server Launcher — Textual TUI."""

    CSS_PATH = Path(__file__).parent / "model_launcher.tcss"
    TITLE = "Model Server Launcher"
    BINDINGS: ClassVar = [
        Binding("q", "quit_app", "Quit", show=False),
    ]

    def __init__(self, preselect: "tuple[AIModel, ServerSoftware] | None" = None) -> None:
        super().__init__()
        self._preselect = preselect

    def on_mount(self) -> None:
        if self._preselect is not None:
            model, server = self._preselect
            # Navigate straight to the server list with the correct model and
            # server pre-selected so the user can immediately mark the result.
            # Only applicable for text-gen (model is always set); image-gen hub
            # flow passes model=None so we fall through to normal startup.
            if model is not None:
                ai_type = next(iter(model.ai_types))
                self.push_screen(ModelSelectScreen(ai_type))
                self.push_screen(ServerSelectScreen(model, preselect_server_id=server.id))
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
    preselect: "tuple[AIModel, ServerSoftware] | None" = None
    while True:
        app = ModelLauncherApp(preselect=preselect)
        result = app.run()
        preselect = None  # consumed; clear for any subsequent loops

        # result is a (factory, model, server) tuple set by ConfirmLaunch/ConfirmQuantize.
        # server is None for quantize jobs (ConfirmQuantize passes None as the third element).
        if not isinstance(result, tuple) or len(result) != 3:
            # User quit the menu without launching — exit to bash.
            sys.exit(0)

        factory, model, server = cast(
            "tuple[Callable[[], tuple[subprocess.Popen, Path, int]], AIModel, ServerSoftware | None]",
            result,
        )
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
            # Ctrl-C on a server launch: reopen the menu with model+server still selected
            # so the user can immediately mark the test result.
            # Ctrl-C on a quantize job, or a stage_all_models server (model=None):
            # just reopen the top-level menu (no preselect).
            if server is not None and model is not None:
                preselect = (model, server)
            continue

        sys.exit(rc or 0)

