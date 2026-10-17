
"""screens.py — Navigation: AI Type → Model → Server → [Environment] → [Quant] → GPUs → Confirm.

Navigation stack:
  AITypeScreen        — Level 1: choose what kind of AI (text-gen, image-gen, …)
  ModelSelectScreen   — Level 2: choose a model (filtered by AIType)
  ServerSelectScreen  — Level 3: choose a server (filtered by model format + type)
  EnvSelectScreen     — Level 3a (optional): choose the environment, when the
                          server has more than one registered
  QuantSelectScreen   — Level 3b (optional): choose runtime quantization, when the
                          server supports on-load quantization for the model’s format
  GpuSelectScreen     — Level 3c: choose GPUs (CUDA_VISIBLE_DEVICES)
  ConfirmLaunch       — Show the command and confirm before spawning

EnvManageScreen (`e` on the server / environment lists) registers, creates,
renames and removes environments using the reusable TextPromptScreen,
ChoiceScreen and CommandRunScreen dialogs.
"""
from __future__ import annotations

import dataclasses
import functools
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, ClassVar

from rich.markup import escape
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Log,
    ProgressBar,
    SelectionList,
    Static,
    TextArea,
)
from textual.containers import Horizontal, Vertical
from textual.worker import get_current_worker

from launcher.gpu import GpuRow, poll_gpus, system_min_sm
from launcher.gpu_power import (
    POWER_LIMITS, PowerLimitStep, fmt_power_limit, parse_power_limit, power_limit_range, power_limit_step,
)
from launcher.models import (
    AIModel,
    AIType,
    AI_TYPE_LABELS,
    CacheMount,
    FORMAT_SHORT,
    ModelFormat,
    QuantizationTool,
    RopeScaling,
    RuntimeQuant,
    RUNTIME_QUANT_LABEL,
    RUNTIME_QUANT_SHORT,
    RUNTIME_QUANT_VRAM_RATIO,
    ServerSoftware,
    available_ai_types,
    fmt_ctx,
    models_for_type,
    servers_for_model,
    servers_for_type,
    tools_for_model,
    uses_hub_flow,
)
from launcher.data import AI_MODELS, CACHE_MOUNTS, QUANTIZATION_TOOLS, SERVERS
from launcher.environments import ENVIRONMENTS, Environment, env_kind
from launcher.fit import FitEstimate, estimate_fit, fit_probe_for
import launcher.launch as launcher


STATUS_COLOR: dict[str, str] = {
    "good":    "bright_green",
    "bad":     "bright_red",
    "unknown": "white",
}

SERVER_STATUS_DOT: dict[str, str] = {
    "good":    "[bright_green]●[/]",
    "bad":     "[bright_red]●[/]",
    "unknown": "[dim]○[/]",
}


# Sort rank for composite status symbols: lower = better / shown first.
_STATUS_RANK: dict[str, int] = {
    "\u2713":  0,   # ✓  — all good
    "\u2713?": 1,   # ✓? — some good, none bad
    "\u2713\u2717": 2,  # ✓✗ — mixed
    "\u2717?": 3,   # ✗? — some bad, none good
    "\u2717":  4,   # ✗  — all bad
    "?":       5,   # all unknown
    "N/A":     6,
}

# ---------------------------------------------------------------------------
# Model-list multi-column sort
#
# _SORT_KEY is the SINGLE SOURCE OF TRUTH for which columns are sortable and
# how each column maps an AIModel to a comparable sort value.
#
# MAINTENANCE CONTRACT — when adding a new column to the model table:
#   1. Add a  key → extractor  entry here.  That is ALL that is needed:
#      _sort_models() and on_header() both derive their behaviour from this
#      dict automatically.  No second place to update.
#   2. If the column should NOT be sortable (e.g. a pure visual derivative
#      like "warn"), deliberately omit it — clicking its header becomes a
#      no-op without any extra code.
#
# Sort-direction conventions used below:
#   • Scalars (size, params, ctx, moe): ascending = smallest first.
#   • Boolean flags (fav, vision, trc):  expressed as `0 if flag else 1` so
#       ascending (the default) puts flagged/"interesting" rows at the top.
#   • Cache: tuple (priority, label) — cached rows (0, ...) before uncached
#       (1, ""); cached entries sub-sorted A–Z by mount label.
#   • Text (name, format, note): lower-cased string, A–Z ascending.
#
# "warn" is intentionally absent: it is a derived visual icon whose
# underlying signals (model size vs VRAM, min_sm) are already captured by
# the "size" and "format" columns.
# ---------------------------------------------------------------------------
_SORT_KEY: dict[str, Callable[["AIModel"], Any]] = {
    "fav":    lambda m: 0 if m.favorite          else 1,
    "name":   lambda m: m.name.lower(),
    "status": lambda m: _STATUS_RANK.get(m.composite_status(SERVERS)[1], 5),
    "format": lambda m: FORMAT_SHORT.get(m.format, m.format.value).lower(),
    "quant":  lambda m: m.quant.lower(),
    "arch":   lambda m: m.arch.lower(),
    "size":   lambda m: m.size_gb,
    "params": lambda m: m.params_m,
    "ctx":    lambda m: m.ctx,
    "moe":    lambda m: m.moe_experts,
    "vision": lambda m: 0 if m.vision            else 1,
    "trc":    lambda m: 0 if m.trust_remote_code else 1,
    "tool":   lambda m: 0 if m.tool_call_parser    else 1,
    "cache":  lambda m: (0, c.label) if (c := m.cached_in(CACHE_MOUNTS)) is not None else (1, ""),
    "note":   lambda m: m.note.lower(),
}


# ---------------------------------------------------------------------------
# Base screen classes
# ---------------------------------------------------------------------------

class DataTableScreen(Screen):
    """Base class for screens driven by a DataTable or SelectionList.

    Provides hidden priority bindings for left/right/enter so these keys are
    intercepted before a focused DataTable can consume them.  The bindings
    call ``action_dt_back()`` (default: pop_screen) and ``action_dt_select()``
    (default: no-op — must be overridden).

    Subclasses should:
    - Define ``action_dt_select()`` with their selection logic.
    - Override ``action_dt_back()`` for root screens that call ``app.exit()``
      instead of ``app.pop_screen()``.
    - Add their *visible* bindings (Quit, Back label, contextual keys, Select f5)
      to their own BINDINGS list — Textual merges all BINDINGS from the MRO.
    """

    BINDINGS: ClassVar = [
        Binding("left",  "dt_back",   "", show=False, priority=True),
        Binding("right", "dt_select", "", show=False, priority=True),
        Binding("enter", "dt_select", "", show=False, priority=True),
    ]

    def action_dt_back(self) -> None:
        self.app.pop_screen()

    def action_dt_select(self) -> None:
        pass  # override in subclasses


class ButtonMenuScreen(Screen):
    """Base class for screens whose primary navigation is via Button widgets.

    Provides up/down focus cycling and Enter dispatch to the focused button.
    Subclasses should call ``.focus()`` on the first button in ``on_mount()``.
    """

    BINDINGS: ClassVar = [
        Binding("up",   "app.focus_previous", "", show=False, priority=True),
        Binding("down", "app.focus_next",     "", show=False, priority=True),
    ]

    def action_press_focused(self) -> None:
        if isinstance(self.focused, Button):
            self.focused.press()


# ---------------------------------------------------------------------------
# ConfirmLaunch — show the full command, ask to proceed
# ---------------------------------------------------------------------------

class ConfirmLaunch(ButtonMenuScreen):
    """Let the user edit the exact shell command before launching it."""

    BINDINGS: ClassVar = [
        Binding("ctrl+q",        "quit_app", "Quit"),
        Binding("escape",        "cancel",   "Back",   key_display="esc"),
        Binding("ctrl+enter",    "confirm",  "Launch", key_display="Ctrl+Enter"),
    ]

    def __init__(self, command: str, proc_factory, model: "AIModel | None", server: ServerSoftware, env: Environment) -> None:
        super().__init__()
        self._command = command
        self._proc_factory = proc_factory
        self._model  = model
        self._server = server
        self._env    = env

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Static("[bold]Ready to launch:[/]", id="confirm-title"),
            TextArea(self._command, language="bash", soft_wrap=True, id="confirm-command"),
            Horizontal(
                Button("Launch", variant="success", id="btn-launch"),
                Button("Back",   variant="default", id="btn-back"),
                id="confirm-buttons",
            ),
            id="confirm-body",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#confirm-command", TextArea).focus()

    def action_confirm(self) -> None:
        command = self.query_one("#confirm-command", TextArea).text
        if not command.strip():
            self.notify("Command cannot be empty", severity="warning", timeout=3)
            return
        factory = functools.partial(self._proc_factory, command)
        self.app.exit(launcher.LaunchJob(factory, self._model, self._server, self._env))

    @on(Button.Pressed, "#btn-launch")
    def on_launch_pressed(self) -> None:
        self.action_confirm()

    @on(Button.Pressed, "#btn-back")
    def action_cancel(self) -> None:
        self.app.pop_screen()


# ---------------------------------------------------------------------------
# Level 3 — Server selection (shown after a model is chosen)
# ---------------------------------------------------------------------------

def _env_brief(server: ServerSoftware, env: Environment) -> str:
    """One-line environment description: name, installed versions, missing flag."""
    if not env.exists():
        return f"{env.name}  [red](missing)[/]"
    versions = server.env_version_label(env)
    return f"{env.name}  [dim]{versions}[/]" if versions else env.name


def _continue_launch(screen: Screen, model: "AIModel | None", server: ServerSoftware, env: Environment) -> None:
    """Advance the launch flow once server + environment are chosen: [Quant] → GPUs."""
    if not env.exists():
        screen.notify(
            f"Environment {env.name!r} not found at {env.location} — press e to manage environments",
            severity="error", timeout=5,
        )
        return
    if model is not None and server.available_runtime_quants(model.format):
        # Let the user optionally choose a runtime quantization mode first.
        screen.app.push_screen(QuantSelectScreen(model, server, env))
        return
    screen.app.push_screen(GpuSelectScreen(server, env, model))


class ServerSelectScreen(DataTableScreen):
    """Choose which server software to use.

    When model is provided (text-gen flow): filters servers by model compatibility,
    shows per-server status dots (best across environments), and supports g/x/?
    status marking for servers with a single environment and no runtime quants
    (otherwise marking happens on the deeper EnvSelectScreen / QuantSelectScreen).

    When model is None (hub flow for stage_all_models types): filters servers by
    ai_type, suppresses status marking (no model to mark against), and skips
    path validation before launching.

    Servers with more than one registered environment continue to EnvSelectScreen;
    `e` opens EnvManageScreen for the focused server.
    """

    BINDINGS: ClassVar = [
        Binding("q",             "quit_app",        "Quit"),
        Binding("escape,b",      "app.pop_screen",  "Back",   key_display="esc/←"),
        Binding("?",             "mark_unknown",    "Clear"),
        Binding("g",             "mark_good",       "Mark good"),
        Binding("x",             "mark_bad",        "Mark bad"),
        Binding("e",             "manage_envs",     "Environments"),
        Binding("f5",            "dt_select",       "Select", key_display="↵/→/dbl-click"),
    ]

    def __init__(
        self,
        model: "AIModel | None",
        ai_type: "AIType | None" = None,
        preselect_server_id: str | None = None,
    ) -> None:
        super().__init__()
        self._model   = model
        self._ai_type = ai_type
        self._servers = (
            servers_for_model(SERVERS, model)
            if model is not None
            else servers_for_type(SERVERS, ai_type)  # type: ignore[arg-type]
        )
        self._preselect_server_id = preselect_server_id

    def _focused_server(self) -> "ServerSoftware | None":
        table = self.query_one("#server-list", DataTable)
        idx = table.cursor_row
        if 0 <= idx < len(self._servers):
            return self._servers[idx]
        return None

    def _single_env(self, server: ServerSoftware) -> Environment | None:
        """The server's only environment, or None when it has zero or several."""
        envs = ENVIRONMENTS.for_server(server.id)
        return envs[0] if len(envs) == 1 else None

    def on_mount(self) -> None:
        self._populate()
        if self._preselect_server_id is not None:
            try:
                idx = next(i for i, s in enumerate(self._servers) if s.id == self._preselect_server_id)
                self.query_one("#server-list", DataTable).move_cursor(row=idx)
            except StopIteration:
                pass

    def on_screen_resume(self) -> None:
        # Environments or results may have changed on a deeper screen.
        self._populate()
        self._refresh_title()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action in ("mark_good", "mark_bad", "mark_unknown"):
            if self._model is None:
                return False  # no model → status marking not applicable
            server = self._focused_server()
            if server is None:
                return False
            env = self._single_env(server)
            # Multi-env or quant-capable servers are marked on the deeper screens.
            if env is None or server.available_runtime_quants(self._model.format):
                return False
            current = self._model.status_for(server.id, env.name)
            if action == "mark_good":    return current != "good"
            if action == "mark_bad":     return current != "bad"
            if action == "mark_unknown": return current != "unknown"
        if action == "manage_envs":
            return self._focused_server() is not None
        return True

    def _set_status(self, status: str) -> None:
        if self._model is None:
            return
        server = self._focused_server()
        env = self._single_env(server) if server is not None else None
        if server is None or env is None:
            return
        try:
            launcher.save_server_status(self._model, server.id, env.name, status, quant=None)
        except Exception as e:
            self.notify(str(e), severity="error", timeout=5)
            return
        self._populate()
        self._refresh_title()
        self.notify(
            f"{self._model.name} / {server.label} / {env.name} → {status}",
            timeout=2,
        )

    def _server_row(self, s: ServerSoftware) -> tuple[str, str, str]:
        """Return (status_dot, label, environment) cells for the server table row."""
        envs = ENVIRONMENTS.for_server(s.id)
        if not envs:
            env_cell = "[red]no environment — press e[/]"
        elif len(envs) == 1:
            env_cell = _env_brief(s, envs[0])
        else:
            env_cell = f"[dim]{len(envs)} environments[/]"
        dot = SERVER_STATUS_DOT[self._model.best_status_for(s.id)] if self._model is not None else ""
        return dot, s.label, env_cell

    def _populate(self) -> None:
        table = self.query_one("#server-list", DataTable)
        saved_row = table.cursor_row
        table.clear()
        for s in self._servers:
            table.add_row(*self._server_row(s), key=s.id)
        table.move_cursor(row=min(saved_row, max(0, len(self._servers) - 1)))
        self.refresh_bindings()

    def _refresh_title(self) -> None:
        self.query_one("#screen-title", Label).update(self._title_markup())

    def _title_markup(self) -> str:
        if self._model is not None:
            color, _ = self._model.composite_status(self._servers)
            return (
                f"[bold]Select server for:[/]  "
                f"[{color}]{self._model.name}  \u00b7  {self._model.format_tag}[/]"
            )
        label = AI_TYPE_LABELS.get(self._ai_type, str(self._ai_type)) if self._ai_type else "Unknown"
        return f"[bold]Select server:[/]  {label}"

    def action_mark_good(self)    -> None: self._set_status("good")
    def action_mark_bad(self)     -> None: self._set_status("bad")
    def action_mark_unknown(self) -> None: self._set_status("unknown")

    def action_manage_envs(self) -> None:
        server = self._focused_server()
        if server is not None:
            self.app.push_screen(EnvManageScreen(server))

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label(self._title_markup(), id="screen-title")
        table = DataTable(id="server-list", cursor_type="row", show_header=False)
        table.add_column("", key="status")
        table.add_column("Server", key="label")
        table.add_column("Environment", key="env")
        yield table
        yield Footer()

    def action_dt_select(self) -> None:
        server = self._focused_server()
        if server is None:
            return
        model = self._model

        if model is not None:
            # Validate model path exists before trying to launch
            eff_path = model.resolve_path()
            p = Path(eff_path)
            exists = p.is_file() if model.format == ModelFormat.GGUF else p.is_dir()
            if not exists:
                self.notify(f"Path not found:\n{eff_path}", severity="error", timeout=4)
                return

        envs = ENVIRONMENTS.for_server(server.id)
        if not envs:
            self.notify(f"No environment registered for {server.label} — press e to add one", severity="warning", timeout=4)
            return
        if len(envs) > 1:
            self.app.push_screen(EnvSelectScreen(model, server))
            return
        _continue_launch(self, model, server, envs[0])

    @on(DataTable.RowSelected)
    def on_row_selected(self, event: DataTable.RowSelected) -> None:
        self.action_dt_select()


# ---------------------------------------------------------------------------
# Level 3a — Environment selection (servers with more than one environment)
# ---------------------------------------------------------------------------

class EnvSelectScreen(DataTableScreen):
    """Choose which registered environment (conda env / native build) to launch from.

    Only pushed when the server has more than one environment.  The cursor starts
    on the default environment (or preselect_env).  g/x/? mark the focused
    environment for servers without runtime quants for this model; quant-capable
    servers are marked per quant on QuantSelectScreen.
    """

    BINDINGS: ClassVar = [
        Binding("q",             "quit_app",        "Quit"),
        Binding("escape,b",      "app.pop_screen",  "Back",   key_display="esc/←"),
        Binding("?",             "mark_unknown",    "Clear"),
        Binding("g",             "mark_good",       "Mark good"),
        Binding("x",             "mark_bad",        "Mark bad"),
        Binding("e",             "manage_envs",     "Environments"),
        Binding("f5",            "dt_select",       "Select", key_display="↵/→/dbl-click"),
    ]

    def __init__(self, model: "AIModel | None", server: ServerSoftware, preselect_env: str | None = None) -> None:
        super().__init__()
        self._model  = model
        self._server = server
        self._envs: list[Environment] = []
        self._preselect_env = preselect_env

    def compose(self) -> ComposeResult:
        model_part = f"  ·  {self._model.name}  ·  {self._model.format_tag}" if self._model else ""
        yield Header()
        yield Label(f"[bold]Select environment for:[/]  {self._server.label}{model_part}", id="screen-title")
        table = DataTable(id="env-list", cursor_type="row", show_header=False)
        table.add_column("", key="status")
        table.add_column("", key="default")
        table.add_column("Environment", key="env")
        table.add_column("Note", key="note")
        yield table
        yield Footer()

    def on_mount(self) -> None:
        self._populate()
        target = self._preselect_env or getattr(ENVIRONMENTS.default_for(self._server.id), "name", None)
        idx = next((i for i, e in enumerate(self._envs) if e.name == target), 0)
        self.query_one("#env-list", DataTable).move_cursor(row=idx)

    def on_screen_resume(self) -> None:
        self._populate()

    def _populate(self) -> None:
        table = self.query_one("#env-list", DataTable)
        saved_row = table.cursor_row
        table.clear()
        self._envs = ENVIRONMENTS.for_server(self._server.id)
        for env in self._envs:
            dot = (
                SERVER_STATUS_DOT[self._model.env_best_status(self._server.id, env.name)]
                if self._model is not None else ""
            )
            table.add_row(
                dot,
                "[yellow]★[/]" if env.default else "",
                _env_brief(self._server, env),
                f"[dim]{env.note}[/]",
                key=env.name,
            )
        table.move_cursor(row=min(saved_row, max(0, len(self._envs) - 1)))
        self.refresh_bindings()

    def _focused_env(self) -> Environment | None:
        idx = self.query_one("#env-list", DataTable).cursor_row
        return self._envs[idx] if 0 <= idx < len(self._envs) else None

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action in ("mark_good", "mark_bad", "mark_unknown"):
            env = self._focused_env()
            if self._model is None or env is None:
                return False
            if self._server.available_runtime_quants(self._model.format):
                return False
            current = self._model.status_for(self._server.id, env.name)
            if action == "mark_good":    return current != "good"
            if action == "mark_bad":     return current != "bad"
            if action == "mark_unknown": return current != "unknown"
        return True

    def _set_status(self, status: str) -> None:
        env = self._focused_env()
        if self._model is None or env is None:
            return
        try:
            launcher.save_server_status(self._model, self._server.id, env.name, status, quant=None)
        except Exception as e:
            self.notify(str(e), severity="error", timeout=5)
            return
        self._populate()
        self.notify(f"{self._model.name} / {self._server.label} / {env.name} → {status}", timeout=2)

    def action_mark_good(self)    -> None: self._set_status("good")
    def action_mark_bad(self)     -> None: self._set_status("bad")
    def action_mark_unknown(self) -> None: self._set_status("unknown")

    def action_manage_envs(self) -> None:
        self.app.push_screen(EnvManageScreen(self._server))

    def action_dt_select(self) -> None:
        env = self._focused_env()
        if env is not None:
            _continue_launch(self, self._model, self._server, env)

    @on(DataTable.RowSelected)
    def on_row_selected(self, event: DataTable.RowSelected) -> None:
        self.action_dt_select()

# ---------------------------------------------------------------------------
# Reusable dialogs — text prompt, button choice, streamed external command
# ---------------------------------------------------------------------------

class TextPromptScreen(Screen["str | None"]):
    """Ask for one line of text.  Dismisses with the stripped value, or None on cancel.

    validate(value) returns an error message to reject the value, or None to accept it.
    """

    BINDINGS: ClassVar = [
        Binding("escape", "cancel", "Cancel", key_display="esc"),
    ]

    def __init__(
        self,
        title: str,
        prompt: str = "",
        initial: str = "",
        placeholder: str = "",
        validate: "Callable[[str], str | None] | None" = None,
    ) -> None:
        super().__init__()
        self._title       = title
        self._prompt      = prompt
        self._initial     = initial
        self._placeholder = placeholder
        self._validate    = validate

    def compose(self) -> ComposeResult:
        prompt = f"\n[dim]{self._prompt}[/]" if self._prompt else ""
        yield Header()
        yield Vertical(
            Label(f"[bold]{self._title}[/]{prompt}", id="screen-title"),
            Input(value=self._initial, placeholder=self._placeholder, id="prompt-input"),
            Horizontal(
                Button("OK",     variant="success", id="btn-ok"),
                Button("Cancel", variant="default", id="btn-cancel"),
                id="confirm-buttons",
            ),
            id="confirm-body",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#prompt-input", Input).focus()

    def _submit(self, value: str) -> None:
        value = value.strip()
        if self._validate is not None and (err := self._validate(value)):
            self.notify(err, severity="error", timeout=4)
            return
        self.dismiss(value)

    @on(Input.Submitted)
    def on_submitted(self, event: Input.Submitted) -> None:
        self._submit(event.value)

    @on(Button.Pressed, "#btn-ok")
    def on_ok(self) -> None:
        self._submit(self.query_one("#prompt-input", Input).value)

    @on(Button.Pressed, "#btn-cancel")
    def action_cancel(self) -> None:
        self.dismiss(None)


class ChoiceScreen(ButtonMenuScreen):
    """Present a message and a vertical list of buttons.

    choices: (choice_id, label, button_variant) tuples.  Dismisses with the
    chosen choice_id, or None on Esc.
    """

    BINDINGS: ClassVar = [
        Binding("escape,left",  "cancel",        "Cancel", key_display="esc/←"),
        Binding("enter,right",  "press_focused", "Select", key_display="↵/→/Space"),
    ]

    def __init__(self, title: str, message: str, choices: "list[tuple[str, str, str]]") -> None:
        super().__init__()
        self._title   = title
        self._message = message
        self._choices = choices

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label(f"[bold]{self._title}[/]", id="menu-title")
        yield Static(self._message, id="choice-message")
        yield Vertical(
            *(
                Button(label, id=f"choice-{i}", variant=variant)  # type: ignore[arg-type]
                for i, (_, label, variant) in enumerate(self._choices)
            ),
            id="main-menu-buttons",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.query("Button").first().focus()

    @on(Button.Pressed)
    def on_button(self, event: Button.Pressed) -> None:
        idx = int((event.button.id or "choice-0").rsplit("-", 1)[-1])
        self.dismiss(self._choices[idx][0])

    def action_cancel(self) -> None:
        self.dismiss(None)


class CommandRunScreen(Screen[bool]):
    """Run an external command (e.g. conda) with live output.  Dismisses with success.

    The screen cannot be left while the command runs — interrupting conda
    mid-operation could leave a half-created or half-renamed environment.
    """

    BINDINGS: ClassVar = [
        Binding("escape", "close", "Close", key_display="esc"),
    ]

    def __init__(self, title: str, argv: list[str]) -> None:
        super().__init__()
        self._title   = title
        self._argv    = argv
        self._busy = True
        self._ok      = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label(f"[bold]{self._title}[/]", id="screen-title")
        yield Static(f"[dim]$ {launcher.format_cmd(self._argv)}[/]", id="cmd-line")
        yield Log(id="cmd-log", highlight=False)
        yield Label("[yellow]Running…[/]", id="cmd-status")
        yield Horizontal(
            Button("Close", variant="default", id="btn-close", disabled=True),
            id="confirm-buttons",
        )
        yield Footer()

    def on_mount(self) -> None:
        self._run()

    @work(thread=True)
    def _run(self) -> None:
        log = self.query_one("#cmd-log", Log)
        try:
            # Text mode translates the \r of progress bars into line breaks.
            proc = subprocess.Popen(
                self._argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, text=True, bufsize=1,
            )
        except OSError as e:
            self.app.call_from_thread(log.write_line, str(e))
            self.app.call_from_thread(self._done, -1)
            return
        assert proc.stdout is not None
        for line in proc.stdout:
            self.app.call_from_thread(log.write_line, line.rstrip("\n"))
        self.app.call_from_thread(self._done, proc.wait())

    def _done(self, rc: int) -> None:
        self._busy = False
        self._ok = rc == 0
        status = "[bright_green]Completed successfully.[/]" if self._ok else f"[bright_red]Failed (exit code {rc}).[/]"
        self.query_one("#cmd-status", Label).update(status)
        btn = self.query_one("#btn-close", Button)
        btn.disabled = False
        btn.focus()

    @on(Button.Pressed, "#btn-close")
    def action_close(self) -> None:
        if self._busy:
            self.notify("Still running — wait for the command to finish", severity="warning", timeout=3)
            return
        self.dismiss(self._ok)


# ---------------------------------------------------------------------------
# Environment management — register / create / rename / remove per server
# ---------------------------------------------------------------------------

class EnvManageScreen(DataTableScreen):
    """Manage the environments registered for one server.

    a  — register an environment (offers to create it when it does not exist)
    c  — create the focused environment when it is missing (kinds that support it)
    r  — rename (renames the underlying conda env and migrates recorded results)
    d  — unregister, optionally deleting the underlying conda env
    s  — make the focused environment the server's default
    n  — edit the note shown next to the environment
    """

    BINDINGS: ClassVar = [
        Binding("q",        "quit_app",       "Quit"),
        Binding("escape,b", "app.pop_screen", "Back", key_display="esc/←"),
        Binding("a",        "add",            "Add"),
        Binding("c",        "create",         "Create"),
        Binding("r",        "rename",         "Rename"),
        Binding("d",        "delete",         "Remove"),
        Binding("s",        "set_default",    "Set default"),
        Binding("n",        "edit_note",      "Note"),
    ]

    def __init__(self, server: ServerSoftware) -> None:
        super().__init__()
        self._server = server
        self._kind   = env_kind(server.env_kind)
        self._envs: list[Environment] = []

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label(
            f"[bold]Environments for:[/]  {self._server.label}  [dim]({self._kind.kind_label})[/]",
            id="screen-title",
        )
        table = DataTable(id="env-manage-list", cursor_type="row")
        table.add_column("★",        key="default")
        table.add_column("Name",     key="name")
        table.add_column("Versions", key="versions")
        table.add_column("Note",     key="note")
        yield table
        yield Label("", id="env-detail")
        yield Footer()

    def on_mount(self) -> None:
        self._populate()

    def _populate(self, focus_name: str | None = None) -> None:
        table = self.query_one("#env-manage-list", DataTable)
        saved_row = table.cursor_row
        table.clear()
        self._envs = ENVIRONMENTS.for_server(self._server.id)
        for env in self._envs:
            exists = env.exists()
            table.add_row(
                "[yellow]★[/]" if env.default else "",
                env.name if exists else f"[red]{env.name}  (missing)[/]",
                self._server.env_version_label(env) if exists else "",
                env.note,
                key=env.name,
            )
        row = next((i for i, e in enumerate(self._envs) if e.name == focus_name), saved_row)
        table.move_cursor(row=min(row, max(0, len(self._envs) - 1)))
        self._update_detail()
        self.refresh_bindings()

    def _focused_env(self) -> Environment | None:
        idx = self.query_one("#env-manage-list", DataTable).cursor_row
        return self._envs[idx] if 0 <= idx < len(self._envs) else None

    def _update_detail(self) -> None:
        env = self._focused_env()
        label = self.query_one("#env-detail", Label)
        if env is None:
            label.update("[dim]No environments registered — press a to add one.[/]")
            return
        shared = [e.server_id for e in ENVIRONMENTS.sharing(env) if e is not env]
        parts = [f"[dim]{env.kind_label}:[/] {env.location}"]
        if not env.exists():
            parts.append("[red]not found on disk[/]")
        if shared:
            parts.append(f"[dim]also used by:[/] {', '.join(shared)}")
        label.update("   ·   ".join(parts))

    @on(DataTable.RowHighlighted)
    def on_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self._update_detail()
        self.refresh_bindings()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        env = self._focused_env()
        if action in ("rename", "delete", "edit_note"):
            return env is not None
        if action == "set_default":
            return env is not None and not env.default
        if action == "create":
            return env is not None and env.supports_create and not env.exists()
        return True

    # -- helpers ---------------------------------------------------------------

    def _validate_new_name(self, name: str) -> str | None:
        if err := Environment.validate_name(name):
            return err
        if ENVIRONMENTS.get(self._server.id, name) is not None:
            return f"{name!r} is already registered for {self._server.label}"
        return None

    async def _create(self, env: Environment, allow_register_only: bool) -> bool | None:
        """Offer to create a missing environment.

        Returns True when created, False when the user chose to register it
        without creating it, None when cancelled or creation failed.
        """
        seen: set[Path] = set()
        sources: list[Environment] = []
        # Same-server environments first: cloning a sibling is the usual way to try an upgrade.
        for src in sorted(ENVIRONMENTS.all(), key=lambda e: e.server_id != self._server.id):
            if type(src) is type(env) and src.exists() and src.location not in seen:
                seen.add(src.location)
                sources.append(src)
        choices = [("empty", "Create an empty environment", "primary")]
        choices += [(f"clone-{i}", f"Clone {src.name}  ({src.server_id})", "primary") for i, src in enumerate(sources)]
        if allow_register_only:
            choices.append(("register", "Register without creating", "warning"))
        choices.append(("cancel", "Cancel", "default"))
        choice = await self.app.push_screen_wait(ChoiceScreen(
            f"Create environment {env.name!r}?",
            f"No {env.kind_label} environment exists at {env.location}.",
            choices,
        ))
        if choice in (None, "cancel"):
            return None
        if choice == "register":
            return False
        if choice == "empty":
            version = await self.app.push_screen_wait(TextPromptScreen(
                f"Create {env.name}", "Python version:", initial="3.11",
                validate=lambda v: None if re.fullmatch(r"\d+(\.\d+)*", v) else "Enter a version such as 3.11",
            ))
            if version is None:
                return None
            argv = env.create_cmd(python_version=version)
        else:
            argv = env.create_cmd(clone_from=sources[int(choice.rsplit("-", 1)[-1])].name)
        ok = await self.app.push_screen_wait(CommandRunScreen(f"Creating {env.name}", argv))
        return True if ok else None

    # -- actions -----------------------------------------------------------------

    @work(exclusive=True)
    async def action_add(self) -> None:
        name = await self.app.push_screen_wait(TextPromptScreen(
            f"Add {self._kind.kind_label} environment for {self._server.label}", "Name:",
            placeholder="e.g. vllm-next", validate=self._validate_new_name,
        ))
        if name is None:
            return
        extras: dict[str, str] = {}
        for fld in self._kind.extra_fields:
            value = await self.app.push_screen_wait(TextPromptScreen(
                f"Add {name}", f"{fld.label}:", placeholder=fld.placeholder,
                validate=lambda v: None if v else "A value is required",
            ))
            if value is None:
                return
            extras[fld.key] = value
        env = Environment.from_dict(self._server.id, {"kind": self._kind.kind, "name": name, **extras})
        if not env.exists():
            if env.supports_create:
                if await self._create(env, allow_register_only=True) is None:
                    return
            else:
                choice = await self.app.push_screen_wait(ChoiceScreen(
                    f"{env.location} not found",
                    "The environment does not exist yet.  Register it anyway?",
                    [("register", "Register anyway", "warning"), ("cancel", "Cancel", "default")],
                ))
                if choice != "register":
                    return
        try:
            ENVIRONMENTS.add(env)
        except ValueError as e:
            self.notify(str(e), severity="error", timeout=5)
            return
        self._populate(focus_name=env.name)
        self.notify(f"Registered {env.name} for {self._server.label}", timeout=3)

    @work(exclusive=True)
    async def action_create(self) -> None:
        env = self._focused_env()
        if env is None:
            return
        if await self._create(env, allow_register_only=False):
            self.notify(f"Created {env.name}", timeout=3)
        self._populate(focus_name=env.name)

    @work(exclusive=True)
    async def action_rename(self) -> None:
        env = self._focused_env()
        if env is None:
            return
        new_name = await self.app.push_screen_wait(TextPromptScreen(
            f"Rename {env.name}", "New name:", initial=env.name,
            validate=lambda v: None if v == env.name else ENVIRONMENTS.rename_conflict(env, v),
        ))
        if new_name is None or new_name == env.name:
            return
        argv = env.rename_cmd(new_name) if env.exists() else None
        if argv is not None:
            shared = [e.server_id for e in ENVIRONMENTS.sharing(env) if e is not env]
            message = f"This renames the {env.kind_label} environment on disk."
            if shared:
                message += f"\nIts registrations for {', '.join(shared)} are renamed too."
            choice = await self.app.push_screen_wait(ChoiceScreen(
                f"Rename {env.name} → {new_name}?", message,
                [("rename", "Rename", "warning"), ("cancel", "Cancel", "default")],
            ))
            if choice != "rename":
                return
            if not await self.app.push_screen_wait(CommandRunScreen(f"Renaming {env.name} → {new_name}", argv)):
                return
        try:
            launcher.apply_env_rename(env, new_name)
        except Exception as e:
            self.notify(str(e), severity="error", timeout=5)
            return
        self._populate(focus_name=new_name)
        self.notify(f"Renamed to {new_name}", timeout=3)

    @work(exclusive=True)
    async def action_delete(self) -> None:
        env = self._focused_env()
        if env is None:
            return
        shared = [e.server_id for e in ENVIRONMENTS.sharing(env) if e is not env]
        n_results = launcher.count_env_status(self._server.id, env.name)
        lines = [f"Unregister {env.name!r} from {self._server.label}?"]
        if n_results:
            lines.append(f"Recorded test results for {n_results} model(s) will be removed.")
        if shared:
            lines.append(f"The {env.kind_label} environment is also used by {', '.join(shared)} and will be kept.")
        argv = env.remove_cmd() if env.exists() and not shared else None
        choices = [("unregister", "Unregister only", "warning")]
        if argv is not None:
            choices.append(("delete", f"Unregister and delete the {env.kind_label} environment", "error"))
        choices.append(("cancel", "Cancel", "default"))
        choice = await self.app.push_screen_wait(ChoiceScreen(f"Remove {env.name}", "\n".join(lines), choices))
        if choice == "delete":
            assert argv is not None
            if not await self.app.push_screen_wait(CommandRunScreen(f"Deleting {env.name}", argv)):
                return
        elif choice != "unregister":
            return
        try:
            launcher.apply_env_removal(env)
        except Exception as e:
            self.notify(str(e), severity="error", timeout=5)
            return
        self._populate()
        self.notify(f"Removed {env.name}", timeout=3)

    def action_set_default(self) -> None:
        env = self._focused_env()
        if env is None:
            return
        ENVIRONMENTS.set_default(env)
        self._populate(focus_name=env.name)
        self.notify(f"{env.name} is now the default for {self._server.label}", timeout=3)

    @work(exclusive=True)
    async def action_edit_note(self) -> None:
        env = self._focused_env()
        if env is None:
            return
        note = await self.app.push_screen_wait(TextPromptScreen(
            f"Note for {env.name}", "Shown next to the environment (blank to clear):", initial=env.note,
        ))
        if note is None:
            return
        ENVIRONMENTS.set_note(env, note)
        self._populate(focus_name=env.name)


# ---------------------------------------------------------------------------
# Level 3b (optional) — Runtime quantization selection
# ---------------------------------------------------------------------------

class QuantSelectScreen(DataTableScreen):
    """Optional step: choose a runtime quantization mode for the selected model+server.

    Only pushed when server.available_runtime_quants(model.format) is non-empty.
    The first option is always "None (full precision)".
    Use g/x/? to mark the focused option good/bad/unknown without launching.
    """

    BINDINGS: ClassVar = [
        Binding("q",             "quit_app",        "Quit"),
        Binding("escape,b",      "app.pop_screen",  "Back",   key_display="esc/←"),
        Binding("?",             "mark_unknown",    "Clear"),
        Binding("g",             "mark_good",       "Mark good"),
        Binding("x",             "mark_bad",        "Mark bad"),
        Binding("f5",            "dt_select",       "Select", key_display="↵/→/dbl-click"),
    ]

    def __init__(self, model: AIModel, server: ServerSoftware, env: Environment) -> None:
        super().__init__()
        self._model  = model
        self._server = server
        self._env    = env
        # None at index 0 = load at full precision; then the available quant modes.
        quants = server.available_runtime_quants(model.format)
        self._options: list[RuntimeQuant | None] = [None, *quants]
        self._total_gb: float | None = None   # populated by _fetch_vram

    def _quant_at_cursor(self) -> "tuple[bool, RuntimeQuant | None]":
        """Return (valid, quant) for the current DataTable cursor.

        valid is False when the cursor is not on a valid option.
        quant is None when 'full precision' (index 0) is focused.
        """
        idx = self.query_one("#quant-list", DataTable).cursor_row
        if 0 <= idx < len(self._options):
            return True, self._options[idx]
        return False, None

    def compose(self) -> ComposeResult:
        color, _ = self._model.composite_status(SERVERS)
        yield Header()
        yield Label(
            f"[bold]Select runtime quantization for:[/]  "
            f"[{color}]{self._model.name}  \u00b7  {self._model.format_tag}"
            f"  \u00b7  {self._server.label}  \u00b7  {self._env.name}[/]",
            id="screen-title",
        )
        table = DataTable(id="quant-list", cursor_type="row", show_header=False)
        table.add_column("", key="status")
        table.add_column("Quantization", key="label")
        yield table
        yield Footer()

    def on_mount(self) -> None:
        self._populate()
        self.query_one("#quant-list", DataTable).move_cursor(row=0)
        self._fetch_vram()

    @work(thread=True)
    def _fetch_vram(self) -> None:
        rows = poll_gpus()
        total = sum(r.mem_total_mib for r in rows if not r.error) / 1024
        if total > 0:
            self.app.call_from_thread(self._on_vram, total)

    def _on_vram(self, total: float) -> None:
        self._total_gb = total
        self._populate()

    def _populate(self) -> None:
        table = self.query_one("#quant-list", DataTable)
        saved_row = table.cursor_row
        table.clear()

        # Determine fit status for each option (None = unknown, not yet loaded)
        if self._total_gb is not None and self._model.size_gb:
            fits_full = self._model.size_gb <= self._total_gb
            fit_quants = {rq for rq, _ in self._model.vram_fit_hints(self._total_gb, SERVERS)}
        else:
            fits_full = None
            fit_quants: set[RuntimeQuant] = set()

        def _label(text: str, fits: bool | None) -> str:
            return f"[bright_green]{text}[/]" if fits is True else text

        table.add_row(
            SERVER_STATUS_DOT[self._model.status_for(self._server.id, self._env.name, None)],
            _label("None (load at full precision)", fits_full),
            key="none",
        )
        for q in self._options[1:]:
            if q is not None:
                table.add_row(
                    SERVER_STATUS_DOT[self._model.status_for(self._server.id, self._env.name, q)],
                    _label(RUNTIME_QUANT_LABEL[q], True if q in fit_quants else (None if self._total_gb is None else False)),
                    key=q.value,
                )
        table.move_cursor(row=min(saved_row, max(0, len(self._options) - 1)))
        self.refresh_bindings()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action in ("mark_good", "mark_bad", "mark_unknown"):
            valid, quant = self._quant_at_cursor()
            if not valid:
                return False
            current = self._model.status_for(self._server.id, self._env.name, quant)
            if action == "mark_good":    return current != "good"
            if action == "mark_bad":     return current != "bad"
            if action == "mark_unknown": return current != "unknown"
        return True

    def _set_status(self, status: str) -> None:
        valid, quant = self._quant_at_cursor()
        if not valid:
            return
        try:
            launcher.save_server_status(self._model, self._server.id, self._env.name, status, quant=quant)
        except Exception as e:
            self.notify(str(e), severity="error", timeout=5)
            return
        self._populate()
        quant_label = RUNTIME_QUANT_LABEL[quant] if quant else "None (full precision)"
        self.notify(
            f"{self._model.name} / {self._server.label} / {self._env.name} / {quant_label} \u2192 {status}",
            timeout=2,
        )

    def action_mark_good(self)    -> None: self._set_status("good")
    def action_mark_bad(self)     -> None: self._set_status("bad")
    def action_mark_unknown(self) -> None: self._set_status("unknown")

    def action_dt_select(self) -> None:
        idx = self.query_one("#quant-list", DataTable).cursor_row
        if not (0 <= idx < len(self._options)):
            return
        quant = self._options[idx]
        self.app.push_screen(GpuSelectScreen(self._server, self._env, self._model, quant))

    @on(DataTable.RowSelected)
    def on_row_selected(self, event: DataTable.RowSelected) -> None:
        self.action_dt_select()


# ---------------------------------------------------------------------------
# Level 3c — GPU selection (always shown before ConfirmLaunch)
# ---------------------------------------------------------------------------

class GpuSelectionScreen(DataTableScreen):
    """Shared GPU list, selection controls, and device validation."""

    BINDINGS: ClassVar = [
        Binding("q",        "quit_app",       "Quit"),
        Binding("escape,b", "app.pop_screen", "Back", key_display="esc/←"),
        Binding("a",        "select_all",     "All GPUs"),
        Binding("f5",       "dt_select",      "Confirm", key_display="↵/→"),
    ]

    def __init__(self, title: str, max_gpus: int = 0) -> None:
        super().__init__()
        self._gpu_title = title
        self._max_gpus = max_gpus
        self._valid_rows: list[GpuRow] = []

    def compose(self) -> ComposeResult:
        mode_note = "  [dim](single GPU only)[/]" if self._max_gpus == 1 else ""
        yield Header()
        yield Label(f"[bold]Select GPUs for:[/]  {self._gpu_title}{mode_note}", id="screen-title")
        yield SelectionList(id="gpu-list")
        yield Label("", id="gpu-status")
        yield Label("", id="gpu-fit", classes="-empty")
        yield Footer()

    def on_mount(self) -> None:
        self._valid_rows = [r for r in poll_gpus() if not r.error]
        if not self._valid_rows:
            self.app.call_later(self._skip_to_confirm)
            return
        pre_select = self._initial_gpu_count()
        sl = self.query_one(SelectionList)
        for i, row in enumerate(self._valid_rows):
            sl.add_option((self._gpu_label(row), row.index, i < pre_select))
        sl.highlighted = 0
        self._refresh_status()

    def _initial_gpu_count(self) -> int:
        return 1 if self._max_gpus == 1 else len(self._valid_rows)

    @staticmethod
    def _gpu_label(row: GpuRow) -> str:
        free_gb = (row.mem_total_mib - row.mem_used_mib) / 1024
        free_ratio = free_gb / max(row.mem_total_gb, 1)
        free_color = "bright_green" if free_ratio > 0.8 else ("yellow" if free_ratio > 0.3 else "red")
        used_label = f"[{free_color}]{row.mem_used_gb:.1f}[/] / {row.mem_total_gb:.0f} GB"
        return f"GPU {row.index}  ·  {row.short_uuid}  ·  {used_label}"

    def _skip_to_confirm(self) -> None:
        self.notify(
            "GPU polling unavailable; cannot select GPUs. Go back and retry once GPU polling is available.",
            severity="error", timeout=8,
        )

    @on(SelectionList.SelectionToggled)
    def on_selection_toggled(self, event: SelectionList.SelectionToggled) -> None:
        if self._max_gpus == 1:
            sl = self.query_one(SelectionList)
            if event.selection_index < len(self._valid_rows):
                toggled_gpu = self._valid_rows[event.selection_index].index
                if toggled_gpu in sl.selected:
                    for v in list(sl.selected):
                        if v != toggled_gpu:
                            sl.deselect(v)
        self._refresh_status()

    def _refresh_status(self) -> None:
        selected = sorted(self.query_one(SelectionList).selected)
        if selected:
            devices = ",".join(str(i) for i in selected)
            text = f"[dim]CUDA_VISIBLE_DEVICES=[/][bold]{devices}[/]  ({len(selected)} GPUs)"
        else:
            text = "[red]No GPUs selected — select at least one before confirming.[/]"
        self.query_one("#gpu-status", Label).update(text)

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action == "select_all":
            return self._max_gpus != 1
        return True

    def action_select_all(self) -> None:
        self.query_one(SelectionList).select_all()
        self._refresh_status()


class GpuSelectScreen(GpuSelectionScreen):
    """Select which GPUs to expose to the server via CUDA_VISIBLE_DEVICES.

    Shown between server/quant selection and ConfirmLaunch for every server launch.
    Defaults to all GPUs selected (multi-GPU servers) or GPU 0 (max_gpus=1 servers).

    For servers with gpu_count_param set (vLLM, SGLang), the param is auto-injected
    into the launch command so parallelism flags stay consistent with GPU count.
    For servers with max_gpus=1, selection is enforced as radio-button style.

    For servers with concurrency_param set (llama.cpp slots), +/- adjusts the number of
    concurrent request slots; the value is remembered per model on launch.  When the
    server has a fit probe (fit.py), a memory-fit estimate for the current selection is
    computed in the background and shown below the status bar.

    Each GPU line shows its power limit (``current→target`` when a launch would change
    it).  ``p`` sets the limit for the highlighted GPU, ``P`` for all selected GPUs; the
    choice is remembered per card (gpu_power.py) and applied as the first step of the
    launch command.
    """

    _MAX_SLOTS = 64
    _FIT_DEBOUNCE_S = 0.4   # settle time before re-running the (multi-second) fit probe

    BINDINGS: ClassVar = [
        Binding("q",             "quit_app",        "Quit"),
        Binding("escape,b",      "app.pop_screen",  "Back",    key_display="esc/←"),
        Binding("a",             "select_all",      "All GPUs"),
        Binding("plus,equals_sign", "slots(1)",     "Slots+",  key_display="+"),
        Binding("minus",         "slots(-1)",       "Slots−",  key_display="-"),
        Binding("p",             "power_limit(False)", "Power limit"),
        Binding("P",             "power_limit(True)",  "Limit selected"),
        Binding("f5",            "dt_select",       "Confirm", key_display="↵/→"),
    ]

    def __init__(
        self,
        server: ServerSoftware,
        env: Environment,
        model: "AIModel | None",
        runtime_quant: "RuntimeQuant | None" = None,
    ) -> None:
        model_part = f"  ·  {model.name}" if model else ""
        super().__init__(f"{server.label}  ·  {env.name}{model_part}", server.max_gpus)
        self._server       = server
        self._env          = env
        self._model        = model
        self._runtime_quant = runtime_quant
        # Parallelism plan for TP/PP servers (vLLM, SGLang); recomputed on
        # every selection change so the status bar stays current.
        self._plan: launcher.ParallelismPlan | None = None
        # Concurrent request slots (server.concurrency_param); None = not adjustable.
        self._slots: int | None = None
        if server.concurrency_param and model is not None:
            self._slots = int(launcher.launch_params(server, model)[server.concurrency_param])
        self._fit_probe = fit_probe_for(server, env) if model is not None else None
        # Bumped on every settings change; results from older estimates are discarded.
        self._fit_generation = 0
        self._fit: FitEstimate | None = None

    def _initial_gpu_count(self) -> int:
        n_available = len(self._valid_rows)
        if self._server.gpu_count_param == "pipeline_parallel" and self._model:
            plan = launcher.compute_parallelism(n_available, self._model)
            return plan.gpu_count
        return super()._initial_gpu_count()

    @staticmethod
    def _gpu_label(row: GpuRow) -> str:
        power = fmt_power_limit(row)
        power_label = f"  ·  {power}" if power else ""
        return GpuSelectionScreen._gpu_label(row) + power_label

    def _skip_to_confirm(self) -> None:
        """Fall back to ConfirmLaunch with the default CUDA_VISIBLE_DEVICES when no GPUs found."""
        self.app.pop_screen()
        server, env, model, quant = self._server, self._env, self._model, self._runtime_quant
        command = launcher.build_launch_command(server, env, model, quant)
        factory = lambda edited, s=server, e=env, m=model, q=quant: launcher.launch(s, e, m, q, command=edited)
        self.app.push_screen(
            ConfirmLaunch(command, factory, model, server, env)
        )

    def _refresh_status(self) -> None:
        self._render_status()
        self._refresh_fit()

    def _render_status(self) -> None:
        sl = self.query_one(SelectionList)
        selected = sorted(sl.selected)
        n = len(selected)
        label = self.query_one("#gpu-status", Label)
        if not selected:
            label.update("[red]No GPUs selected — select at least one before confirming.[/]")
            self._plan = None
            return
        devices_str = ",".join(str(i) for i in selected)
        parts = [
            f"[dim]CUDA_VISIBLE_DEVICES=[/][bold]{devices_str}[/]"
            f"  [dim]({n} GPU{'s' if n != 1 else ''})[/]"
        ]
        if self._server.gpu_count_param == "pipeline_parallel" and self._model:
            self._plan = launcher.compute_parallelism(n, self._model)
            if self._plan.gpu_count < n:
                adj = selected[:self._plan.gpu_count]
                adj_str = ",".join(str(i) for i in adj)
                parts = [
                    f"[dim]CUDA_VISIBLE_DEVICES=[/][bold]{adj_str}[/]"
                    f"  [yellow]({self._plan.gpu_count} of {n} GPUs)[/]"
                ]
            tp_pp = (
                f"TP=[bold]{self._plan.tp}[/] × PP=[bold]{self._plan.pp}[/]"
                if self._plan.tp > 1
                else f"PP=[bold]{self._plan.pp}[/]"
            )
            parts.append(tp_pp)
            if self._plan.note:
                parts.append(f"[yellow]{self._plan.note}[/]")
        elif self._server.gpu_count_param:
            self._plan = None
            parts.append(f"[dim]{self._server.gpu_count_param}[/] → [bold]{n}[/]")
        else:
            self._plan = None
        power_step = self._power_step()
        if power_step is not None:
            parts.append(f"[dim]power limits[/] {' '.join(power_step.args())}")
        label.update("   ·   ".join(parts))

    def _launch_indices(self) -> list[int]:
        """GPU indices the launch will use — the selection, trimmed to the parallelism plan."""
        selected = sorted(self.query_one(SelectionList).selected)
        if self._plan is not None:
            selected = selected[:self._plan.gpu_count]
        return selected

    def _launch_settings(self) -> tuple[str, dict[str, Any]] | None:
        """(CUDA_VISIBLE_DEVICES, param_overrides) for the current selection; None when nothing is selected."""
        indices = self._launch_indices()
        if not indices:
            return None
        overrides: dict[str, Any] = {}
        if self._plan is not None:
            overrides = {"tensor_parallel": self._plan.tp, "pipeline_parallel": self._plan.pp}
        elif self._server.gpu_count_param:
            overrides = {self._server.gpu_count_param: len(indices)}
        if self._slots is not None and self._server.concurrency_param:
            overrides[self._server.concurrency_param] = self._slots
        return ",".join(str(i) for i in indices), overrides

    def _power_step(self) -> PowerLimitStep | None:
        return power_limit_step(self._valid_rows, self._launch_indices())

    # -- Power limits -----------------------------------------------------

    def action_power_limit(self, selected: bool) -> None:
        sl = self.query_one(SelectionList)
        if selected:
            chosen = set(sl.selected)
            rows = [r for r in self._valid_rows if r.index in chosen]
        elif sl.highlighted is not None:
            rows = [self._valid_rows[sl.highlighted]]
        else:
            rows = []
        if not rows:
            self.notify(f"No GPU {'selected' if selected else 'highlighted'}", severity="warning", timeout=3)
            return
        lo, hi = power_limit_range(rows)
        defaults = sorted({round(r.power_default_w) for r in rows})
        default_str = f"{defaults[0]} W" if len(defaults) == 1 else f"{defaults[0]}–{defaults[-1]} W per card"
        stored = {POWER_LIMITS.get(r.uuid) for r in rows}
        initial = str(next(iter(stored))) if len(stored) == 1 and None not in stored else ""
        ids = ",".join(str(r.index) for r in rows)

        def validate(text: str) -> str | None:
            try:
                parse_power_limit(text, rows)
            except ValueError as e:
                return str(e)
            return None

        def on_result(text: str | None) -> None:
            if text is None:
                return
            watts = parse_power_limit(text, rows)
            try:
                POWER_LIMITS.set_for(rows, watts)
            except OSError as e:
                self.notify(f"Could not save power limit: {e}", severity="error", timeout=5)
                return
            for i, row in enumerate(self._valid_rows):
                sl.replace_option_prompt_at_index(i, self._gpu_label(row))
            self._render_status()
            target = "factory default" if watts is None else f"{watts} W"
            self.notify(f"GPU {ids} power limit → {target} (applied at launch)", timeout=3)

        self.app.push_screen(
            TextPromptScreen(
                f"Power limit for GPU{'s' if len(rows) > 1 else ''} {ids}",
                prompt=(
                    f"Watts, {lo}–{hi}.  Blank = factory default ({default_str}).\n"
                    "Remembered per card; applied (via sudo, only when it differs) when a launch starts."
                ),
                initial=initial,
                placeholder=f"{lo}–{hi}",
                validate=validate,
            ),
            on_result,
        )

    # -- Slots and memory-fit estimate ------------------------------------

    def _refresh_fit(self) -> None:
        """Invalidate the current estimate, re-render, and schedule a new estimate."""
        self._fit_generation += 1
        self._fit = None
        self._render_fit()
        settings = self._launch_settings()
        if self._fit_probe is not None and settings is not None:
            self._estimate_fit(self._fit_generation, *settings)

    @work(thread=True, exclusive=True, group="fit")
    def _estimate_fit(self, generation: int, cuda_devices: str, overrides: dict[str, Any]) -> None:
        # exclusive=True cancels the previous estimate's worker, whose probe then kills its subprocess.
        worker = get_current_worker()
        time.sleep(self._FIT_DEBOUNCE_S)
        if worker.is_cancelled or self._model is None:
            return
        estimate = estimate_fit(
            self._server, self._env, self._model, cuda_devices, overrides,
            cancelled=lambda: worker.is_cancelled,
            progress=lambda provisional: (
                None if worker.is_cancelled
                else self.app.call_from_thread(self._show_fit, generation, provisional)
            ),
        )
        if not worker.is_cancelled:
            self.app.call_from_thread(self._show_fit, generation, estimate)

    def _show_fit(self, generation: int, estimate: FitEstimate | None) -> None:
        if generation == self._fit_generation:
            self._fit = estimate
            self._render_fit()

    def _render_fit(self) -> None:
        parts: list[str] = []
        if self._slots is not None and self._model is not None:
            parts.append(f"Slots [bold]{self._slots}[/] × {fmt_ctx(self._server.launch_ctx(self._model))} ctx  [dim](+/-)[/]")
        if self._fit_probe is not None and self.query_one(SelectionList).selected:
            parts.append(_fit_markup(self._fit, self._slots))
        label = self.query_one("#gpu-fit", Label)
        label.update("   ·   ".join(parts))
        label.set_class(not parts, "-empty")

    def action_slots(self, delta: int) -> None:
        if self._slots is None:
            return
        slots = max(1, min(self._MAX_SLOTS, self._slots + delta))
        if slots != self._slots:
            self._slots = slots
            self._refresh_fit()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action == "slots":
            return self._slots is not None
        return super().check_action(action, parameters)

    def action_dt_select(self) -> None:
        settings = self._launch_settings()
        if settings is None:
            self.notify("Select at least one GPU", severity="warning", timeout=3)
            return
        cuda_devices, param_overrides = settings
        try:
            command = launcher.build_launch_command(
                self._server,
                self._env,
                self._model,
                self._runtime_quant,
                cuda_devices=cuda_devices,
                param_overrides=param_overrides,
                power_step=self._power_step(),
            )
        except Exception as e:
            self.notify(str(e), severity="error", timeout=5)
            return
        server, env, model, quant, slots = self._server, self._env, self._model, self._runtime_quant, self._slots

        def factory(edited: str):
            if slots is not None and model is not None and server.concurrency_param:
                launcher.save_server_param(model, server, server.concurrency_param, slots)
            return launcher.launch(
                server, env, model, quant,
                cuda_devices=cuda_devices, param_overrides=param_overrides, command=edited,
            )

        self.app.push_screen(ConfirmLaunch(command, factory, model, server, env))


def _gib(mib: int) -> str:
    return f"{mib / 1024:.0f}"


def _fit_markup(estimate: FitEstimate | None, slots: int | None) -> str:
    """Status-bar markup for a memory-fit estimate (None = still estimating)."""
    if estimate is None:
        return "[dim]estimating memory fit…[/]"
    if estimate.error:
        return f"[dim]fit estimate unavailable: {escape(estimate.error)}[/]"
    if estimate.fits:
        verdict = "[bright_green]✓ fits on GPU[/]"
    elif estimate.gpu_layers is not None:
        verdict = f"[yellow]⚠ only {estimate.gpu_layers} layers fit on GPU — rest offloaded to CPU (slow)[/]"
    elif estimate.shortfall_mib == 0 and (estimate.searching or estimate.search_timed_out):
        # Enough memory in total, but the default layer split overloads some devices.
        verdict = "[yellow]⚠ fits in total GPU memory, but needs layers rebalanced across GPUs[/]"
        if estimate.searching:
            verdict += "  [dim]searching for a placement…[/]"
        else:
            verdict += "  [dim](placement search timed out — llama-server will attempt it at launch)[/]"
    else:
        verdict = "[bright_red]✗ does not fit in free GPU memory[/]"
        if estimate.shortfall_mib:
            verdict += f" [bright_red]— {_gib(estimate.shortfall_mib)} GiB short[/]"
        if estimate.searching:
            verdict += "  [dim]searching for a GPU/CPU split…[/]"
        elif estimate.search_timed_out:
            verdict += "  [dim](GPU/CPU split search timed out)[/]"
    details: list[str] = []
    if estimate.projected_mib is not None and estimate.free_mib is not None:
        details.append(f"needs {_gib(estimate.projected_mib)} of {_gib(estimate.free_mib)} GiB free")
    if estimate.model_mib is not None and estimate.context_mib is not None and estimate.compute_mib is not None:
        kv = f"KV {_gib(estimate.context_mib)}"
        if slots and slots > 1:
            kv += f" ({estimate.context_mib / 1024 / slots:.1f}/slot)"
        details.append(f"weights {_gib(estimate.model_mib)} · {kv} · compute {_gib(estimate.compute_mib)} GiB")
    return verdict + (f"  [dim]{' · '.join(details)}[/]" if details else "")


# ---------------------------------------------------------------------------
# Quantize — Screen 1 (optional): choose the quantization tool
# ---------------------------------------------------------------------------

class QuantizeToolScreen(DataTableScreen):
    """Select which quantization tool to use.

    Only pushed when more than one tool is applicable for the model's format.
    Skipped automatically when exactly one tool applies.
    """

    BINDINGS: ClassVar = [
        Binding("q",             "quit_app",       "Quit"),
        Binding("escape,b",      "app.pop_screen", "Back",   key_display="esc/←"),
        Binding("f5",            "dt_select",      "Select", key_display="↵/→/dbl-click"),
    ]

    def __init__(self, model: AIModel, tools: list[QuantizationTool]) -> None:
        super().__init__()
        self._model = model
        self._tools = tools

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label(
            f"[bold]Select quantization tool for:[/]  "
            f"{self._model.name}  ·  [dim]{self._model.format_tag}[/]",
            id="screen-title",
        )
        table = DataTable(id="tool-list", cursor_type="row", show_header=False)
        table.add_column("Tool", key="label")
        for t in self._tools:
            label = t.label if launcher.quant_tool_available(t) else f"{t.label}  [dim](not installed)[/]"
            table.add_row(label)
        yield table
        yield Footer()

    def action_dt_select(self) -> None:
        idx = self.query_one("#tool-list", DataTable).cursor_row
        if not (0 <= idx < len(self._tools)):
            return
        tool = self._tools[idx]
        if not launcher.quant_tool_available(tool):
            if tool.pip_package:
                self.app.push_screen(QuantToolInstallScreen(self._model, tool))
            else:
                self.notify(tool.install_note or f"{tool.label} is not available.", timeout=8)
            return
        self.app.push_screen(QuantizeOptionScreen(self._model, tool))

    @on(DataTable.RowSelected)
    def on_row_selected(self, event: DataTable.RowSelected) -> None:
        self.action_dt_select()


# ---------------------------------------------------------------------------
# Quantize — Install screen: auto-install a missing tool dependency
# ---------------------------------------------------------------------------

class QuantToolInstallScreen(ButtonMenuScreen):
    """Offer to auto-install a missing quantization tool dependency.

    Used when a tool declares a pip_package (e.g. exllamav2), installed into the
    tool's environment (the default environment of tool.env_server_id).
    The install runs in a background thread so the TUI remains responsive.
    On success the screen pops itself and pushes QuantizeOptionScreen.
    """

    BINDINGS: ClassVar = [
        Binding("q",             "quit_app",        "Quit"),
        Binding("escape,b,left", "cancel",          "Cancel",  key_display="esc/←"),
        Binding("enter,right",   "press_focused",   "Install", key_display="↵/→/Space"),
    ]

    def __init__(self, model: AIModel, tool: QuantizationTool) -> None:
        super().__init__()
        self._model = model
        self._tool  = tool
        self._installing = False

    def compose(self) -> ComposeResult:
        pkg = self._tool.pip_package or "(unknown)"
        env = launcher.quant_tool_env(self._tool)
        env_name = env.name if env is not None else f"(no {self._tool.env_server_id} environment)"
        yield Header()
        yield Vertical(
            Static(
                f"[bold]{self._tool.label}[/] requires [bold]{pkg}[/],\n"
                f"which is not installed in the [dim]{env_name}[/] environment.\n\n"
                "Press [bold]Enter[/] to install it now, or [bold]Esc[/] to cancel.",
                id="install-text",
            ),
            Horizontal(
                Button("Install", variant="success", id="btn-install"),
                Button("Cancel",  variant="default", id="btn-cancel"),
                id="install-buttons",
            ),
            id="confirm-body",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#btn-install", Button).focus()

    @on(Button.Pressed, "#btn-install")
    def on_install(self) -> None:
        if self._installing:
            return
        self._installing = True
        self.query_one("#btn-install", Button).disabled = True
        self.query_one("#btn-cancel", Button).disabled = True
        pkg = self._tool.pip_package
        self.query_one("#install-text", Static).update(
            f"Installing [bold]{pkg}[/]…\n"
            "(This may take 1–2 minutes. Do not close this window.)"
        )
        self._run_install()

    @on(Button.Pressed, "#btn-cancel")
    def action_cancel(self) -> None:
        if not self._installing:
            self.app.pop_screen()

    @work(thread=True)
    def _run_install(self) -> None:
        success, log_path = launcher.install_quant_tool(self._tool)
        self.call_from_thread(self._on_install_done, success, log_path)  # type: ignore[attr-defined]

    def _on_install_done(self, success: bool, log_path: str) -> None:
        self._installing = False
        if success:
            self.app.pop_screen()  # back to QuantizeToolScreen
            self.app.push_screen(QuantizeOptionScreen(self._model, self._tool))
        else:
            self.query_one("#install-text", Static).update(
                "[red bold]Installation failed.[/]\n"
                f"See log for details: [dim]{log_path}[/]\n\n"
                "Press [bold]Esc[/] to go back."
            )
            self.query_one("#btn-cancel", Button).disabled = False


# ---------------------------------------------------------------------------
# Quantize — Screen 2: choose the quant type / bpw
# ---------------------------------------------------------------------------

class QuantizeOptionScreen(DataTableScreen):
    """Select a quantization option (e.g. Q4_K_M or 4.0 bpw)."""

    BINDINGS: ClassVar = [
        Binding("q",             "quit_app",       "Quit"),
        Binding("escape,b",      "app.pop_screen", "Back",   key_display="esc/←"),
        Binding("f5",            "dt_select",      "Select", key_display="↵/→/dbl-click"),
    ]

    def __init__(self, model: AIModel, tool: QuantizationTool) -> None:
        super().__init__()
        self._model = model
        self._tool  = tool

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label(
            f"[bold]Select quantization for:[/]  "
            f"{self._model.name}  ·  [dim]{self._model.format_tag}[/]"
            f"  →  [dim]{self._tool.label}[/]",
            id="screen-title",
        )
        table = DataTable(id="option-list", cursor_type="row", show_header=False)
        table.add_column("Option", key="label")
        for opt in self._tool.quant_options:
            table.add_row(self._tool.quant_option_labels[opt])
        yield table
        yield Footer()

    def action_dt_select(self) -> None:
        idx = self.query_one("#option-list", DataTable).cursor_row
        if not (0 <= idx < len(self._tool.quant_options)):
            return
        quant_option = self._tool.quant_options[idx]
        self.app.push_screen(QuantizeDestScreen(self._model, self._tool, quant_option))

    @on(DataTable.RowSelected)
    def on_row_selected(self, event: DataTable.RowSelected) -> None:
        self.action_dt_select()


# ---------------------------------------------------------------------------
# Quantize — Screen 3: choose output destination (cache mount)
# ---------------------------------------------------------------------------

class QuantizeDestScreen(ButtonMenuScreen):
    """Select which local storage mount to write the quantized output to."""

    BINDINGS: ClassVar = [
        Binding("q",             "quit_app",       "Quit"),
        Binding("escape,b,left", "app.pop_screen", "Back",   key_display="esc/←"),
        Binding("enter,right",   "press_focused",  "Select", key_display="↵/→/Space"),
    ]

    def __init__(self, model: AIModel, tool: QuantizationTool, quant_option: str) -> None:
        super().__init__()
        self._model        = model
        self._tool         = tool
        self._quant_option = quant_option

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label(
            f"[bold]Select output destination for:[/]  "
            f"{self._model.name}  ·  [dim]{self._tool.quant_option_labels[self._quant_option]}[/]",
            id="screen-title",
        )
        yield Vertical(id="dest-ops")
        yield Footer()

    def on_mount(self) -> None:
        self._build_buttons()

    def _build_buttons(self) -> None:
        ops = self.query_one("#dest-ops", Vertical)
        for mount in sorted(CACHE_MOUNTS, key=lambda m: m.priority):
            total, free = launcher.cache_disk_info(mount)
            fits = self._model.size_gb <= free if self._model.size_gb else True
            fit_color = "green" if fits else "red"
            output_path = launcher.compute_quant_output_path(
                self._tool, self._model, self._quant_option, mount,
            )
            ops.mount(Button(
                f"{mount.label}  "
                f"[dim](output: {output_path.name}  ·  "
                f"[{fit_color}]{free:.0f} GB free[/]  ·  {total:.0f} GB total)[/]",
                id=f"btn-dest-{mount.priority}",
                variant="primary" if fits else "default",
            ))
        ops.mount(Button("Back", id="btn-back", variant="default"))
        self.call_after_refresh(lambda: self.query("#dest-ops Button").first().focus())

    @on(Button.Pressed)
    def on_btn(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "btn-back":
            self.app.pop_screen()
        elif bid.startswith("btn-dest-"):
            priority = int(bid.rsplit("-", 1)[-1])
            mount = next(m for m in CACHE_MOUNTS if m.priority == priority)
            self.app.push_screen(
                QuantizeGpuScreen(self._model, self._tool, self._quant_option, mount)
            )


class QuantizeGpuScreen(GpuSelectionScreen):
    """Choose devices for an offline quantization job before confirmation."""

    def __init__(
        self, model: AIModel, tool: QuantizationTool, quant_option: str, dest: CacheMount,
    ) -> None:
        super().__init__(f"{tool.label}  ·  {model.name}")
        self._model = model
        self._tool = tool
        self._quant_option = quant_option
        self._dest = dest

    def action_dt_select(self) -> None:
        selected = sorted(self.query_one(SelectionList).selected)
        if not selected:
            self.notify("Select at least one GPU", severity="warning", timeout=3)
            return
        cuda_devices = ",".join(str(i) for i in selected)
        try:
            cmd, output_path = launcher.build_quantize_cmd(
                self._tool, self._model, self._quant_option, self._dest, cuda_devices,
            )
        except (ValueError, OSError) as e:
            self.notify(str(e), severity="error", timeout=5)
            return
        self.app.push_screen(
            ConfirmQuantize(
                launcher.format_cmd(cmd),
                lambda t=self._tool, m=self._model, q=self._quant_option, d=self._dest:
                    launcher.quantize(t, m, q, d, cuda_devices),
                self._model,
                self._tool,
                self._quant_option,
                output_path,
            )
        )


# ---------------------------------------------------------------------------
# Quantize — Screen 5: confirm and launch the quantization job
# ---------------------------------------------------------------------------

class ConfirmQuantize(ButtonMenuScreen):
    """Show the quantization command and ask for confirmation."""

    BINDINGS: ClassVar = [
        Binding("q",             "quit_app",       "Quit"),
        Binding("escape,b,left", "cancel",         "Back",     key_display="esc/←"),
        Binding("enter,right",   "press_focused",  "Quantize", key_display="↵/→/Space"),
    ]

    def __init__(
        self,
        description: str,
        proc_factory,
        model: AIModel,
        tool: QuantizationTool,
        quant_option: str,
        output_path: "Path",
    ) -> None:
        super().__init__()
        self._description  = description
        self._proc_factory = proc_factory
        self._model        = model
        self._tool         = tool
        self._quant_option = quant_option
        self._output_path  = output_path

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Static(
                f"[bold]Ready to quantize:[/]\n"
                f"[dim]Output:[/]  {self._output_path}\n\n"
                f"{self._description}",
                id="confirm-text",
            ),
            Horizontal(
                Button("Quantize", variant="success", id="btn-launch"),
                Button("Back",     variant="default", id="btn-back"),
                id="confirm-buttons",
            ),
            id="confirm-body",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#btn-launch", Button).focus()

    @on(Button.Pressed, "#btn-launch")
    def action_confirm(self) -> None:
        self.app.exit(launcher.QuantizeJob(
            self._proc_factory,
            self._model,
            self._tool,
            self._quant_option,
            self._output_path,
        ))

    @on(Button.Pressed, "#btn-back")
    def action_cancel(self) -> None:
        self.app.pop_screen()


# ---------------------------------------------------------------------------
# Level 2 — Model selection (sortable DataTable, live VRAM check)
# ---------------------------------------------------------------------------

class ModelSelectScreen(DataTableScreen):
    """Choose a model from a sortable table. VRAM is checked live on mount."""

    BINDINGS: ClassVar = [
        Binding("q",             "quit_app",          "Quit"),
        Binding("escape,b",      "dt_back",           "Back",   key_display="esc/←"),
        Binding("c",             "cache_selected",    "Cache"),
        Binding("c",             "uncache_selected",  "Uncache"),
        Binding("f",             "mark_favorite",     "Favorite"),
        Binding("f",             "unmark_favorite",   "Unfavorite"),
        Binding("l",             "set_ctx_override",  "Context / YaRN"),
        Binding("z",             "quantize_selected", "Quantize"),
        Binding("f5",            "dt_select",         "Select", key_display="↵/→/dbl-click"),
    ]

    def __init__(self, ai_type: AIType, preselect_model: "AIModel | None" = None) -> None:
        super().__init__()
        self._ai_type         = ai_type
        self._preselect_model = preselect_model
        self._models          = models_for_type(AI_MODELS, ai_type)
        # Sort priority list: each entry is (column_key, reverse).  Index 0 is primary.
        # Default: favorites first → good/bad status → alphabetical by name.
        self._sort_keys: list[tuple[str, bool]] = [
            ("fav",    False),
            ("status", False),
            ("name",   False),
        ]
        self._avail_gb: float | None = None  # None while still fetching
        self._total_gb: float = 0.0
        self._system_sm: int = system_min_sm()  # cached; 0 if NVML unavailable

    def _title_text(self) -> str:
        return f"[bold]{AI_TYPE_LABELS[self._ai_type]} — select model:[/]"

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label(
            self._title_text(),
            id="screen-title",
        )
        table = DataTable(id="model-table", cursor_type="row")
        table.add_column("★",      key="fav")    # ★ — favorite indicator (leftmost)
        table.add_column("Name",   key="name")
        table.add_column("👍",     key="status")  # composite status symbol
        table.add_column("Format", key="format")
        table.add_column("Quant",  key="quant")
        table.add_column("Arch",   key="arch")
        table.add_column("",       key="warn")    # VRAM warning icon
        table.add_column("Size",   key="size")
        table.add_column("Params", key="params")
        table.add_column("Ctx",    key="ctx")
        table.add_column("MoE",    key="moe")    # expert count (blank for dense)
        table.add_column("👁",     key="vision")  # vision/multimodal flag
        table.add_column("TRC",    key="trc")    # --trust-remote-code required
        table.add_column("🔧",     key="tool")   # tool-call parser configured
        table.add_column("Cache",  key="cache")
        table.add_column("Note",   key="note")   # user note (rightmost)
        yield table
        with Horizontal(id="status-bar"):
            yield Label("[dim]Querying VRAM…[/]", id="status-bar-left")
            yield Label("", id="status-bar-right")
        yield Footer()

    def on_mount(self) -> None:
        self._sort_models()
        self._populate(snap_to=self._preselect_model)
        self._refresh_vram()                       # immediate first poll
        self.set_interval(2.0, self._refresh_vram) # re-poll every 2 s automatically

    @work(thread=True)
    def _refresh_vram(self) -> None:
        rows = poll_gpus()
        total = sum(r.mem_total_mib for r in rows if not r.error) / 1024
        avail = sum((r.mem_total_mib - r.mem_used_mib) for r in rows if not r.error) / 1024
        self.app.call_from_thread(self._on_vram, total, avail)

    def _on_vram(self, total: float, avail: float) -> None:
        prev_avail = self._avail_gb
        self._total_gb = total
        self._avail_gb = avail
        used = total - avail
        free_pct = (avail / total * 100) if total else 0
        free_color = "green" if free_pct > 50 else ("yellow" if free_pct > 20 else "red")
        self.query_one("#status-bar-left", Label).update(
            f"[dim]VRAM:[/]  [bold]{total:.0f}[/] GB total"
            f"  ·  {used:.0f} GB used"
            f"  ·  [{free_color}][bold]{avail:.0f}[/] GB free[/]"
        )
        # Only repopulate if this is the first result or available VRAM shifted ≥1 GB
        if prev_avail is None or abs(avail - prev_avail) >= 1.0:
            self._populate()

    def _arch_cell(self, model: AIModel) -> str:
        """Short architecture class name — strips common HF suffixes like ForCausalLM."""
        arch = model.arch
        if not arch:
            return ""
        for suffix in ("ForCausalLM", "ForConditionalGeneration", "ForSequenceClassification", "Model"):
            if arch.endswith(suffix):
                arch = arch[: -len(suffix)]
                break
        return arch

    def _warn_cell(self, model: AIModel) -> str:
        """Warning icon (with colour) if the model won't run or fit, else empty string."""
        if self._system_sm and not model.hw_compat(self._system_sm):
            return "[red]⛔[/]"                  # hardware incompatibility — highest priority
        if self._avail_gb is None:
            return ""                              # still loading
        if model.size_gb > self._total_gb:
            return "[red]⚠[/]"                    # exceeds total installed VRAM
        if model.size_gb > self._avail_gb:
            return "[yellow]⚠[/]"                 # fits in total but not currently free
        return ""

    def _size_cell(self, model: AIModel) -> str:
        """Plain size string, dimmed while VRAM info is still loading."""
        size = f"{model.size_gb:.0f} GB" if model.size_gb else "?"
        if self._avail_gb is None:
            return f"[dim]{size}[/]"
        return size

    def _moe_cell(self, model: AIModel) -> str:
        """Expert count string for MoE models, blank for dense."""
        if not model.moe_experts:
            return ""
        if model.moe_active:
            return f"{model.moe_experts}/{model.moe_active}"
        return str(model.moe_experts)

    def _vision_cell(self, model: AIModel) -> str:
        """Eye emoji when model accepts vision input, blank otherwise."""
        return "👁" if model.vision else ""

    def _trc_cell(self, model: AIModel) -> str:
        """Checkmark when --trust-remote-code is required, blank otherwise."""
        return "[yellow]✓[/]" if model.trust_remote_code else ""

    def _tool_cell(self, model: AIModel) -> str:
        """Wrench emoji when model has a tool-call parser configured, blank otherwise."""
        return "🔧" if model.tool_call_parser else ""

    def _params_cell(self, model: AIModel) -> str:
        """Parameter count formatted as XB (≥1000M) or XM (<1000M); blank if unknown."""
        m = model.params_m
        if not m:
            return ""
        if m >= 1000:
            return f"{m / 1000:g}B"
        return f"{m:g}M"

    def _cache_cell(self, model: AIModel) -> str:
        """Short label showing first cache mount where model lives, or dim dash."""
        cached = model.cached_in(CACHE_MOUNTS)
        if cached is None:
            return "[dim]—[/]"
        return f"[bright_green]{cached.label.split()[0]}[/]"

    def _focused_model(self) -> "AIModel | None":
        """Return the model at the current DataTable cursor row."""
        table = self.query_one("#model-table", DataTable)
        idx = table.cursor_row
        if 0 <= idx < len(self._models):
            return self._models[idx]
        return None

    def _sort_models(self) -> None:
        """Sort self._models in-place using the multi-column sort key list.

        self._sort_keys is an ordered list of (column_key, reverse) pairs;
        index 0 is the primary sort, index 1 secondary, and so on.  The list
        is maintained by on_header().

        Sort values are defined in the module-level _SORT_KEY dict.  To make
        a new column sortable, add an entry there — no changes needed here.
        """
        def _cmp(a: AIModel, b: AIModel) -> int:
            for col, rev in self._sort_keys:
                fn = _SORT_KEY.get(col)
                if fn is None:
                    continue
                va, vb = fn(a), fn(b)
                if   va < vb: result = -1
                elif va > vb: result =  1
                else:         result =  0
                if rev:
                    result = -result
                if result != 0:
                    return result
            return 0

        self._models.sort(key=functools.cmp_to_key(_cmp))

    def _populate(self, snap_to: "AIModel | None" = None) -> None:
        table = self.query_one("#model-table", DataTable)
        saved_row = table.cursor_row
        table.clear()
        for m in self._models:
            color, symbol = m.composite_status(SERVERS)
            table.add_row(
                "\u2605" if m.favorite else "",   # ★ or blank
                m.name,
                f"[{color}]{symbol}[/]",
                FORMAT_SHORT.get(m.format, m.format.value),
                m.quant,
                self._arch_cell(m),
                self._warn_cell(m),
                self._size_cell(m),
                self._params_cell(m),
                fmt_ctx(m.effective_ctx) if m.effective_ctx == m.ctx
                    else f"{fmt_ctx(m.effective_ctx)}[dim]/{fmt_ctx(m.ctx)}[/]",
                self._moe_cell(m),
                self._vision_cell(m),
                self._trc_cell(m),
                self._tool_cell(m),
                self._cache_cell(m),
                m.note,
                key=m.id,
            )
        if not self._models:
            return
        if snap_to is not None:
            try:
                target = next(i for i, m in enumerate(self._models) if m.id == snap_to.id)
            except StopIteration:
                target = min(saved_row, len(self._models) - 1)
        else:
            target = min(saved_row, len(self._models) - 1)
        table.move_cursor(row=target)
        self.refresh_bindings()
        self._refresh_note_bar()

    def _note_bar_text(self, model: AIModel) -> str:
        """Return the note bar text for a model, matching _warn_cell priority."""
        if self._system_sm and not model.hw_compat(self._system_sm):
            return f"[red]⛔  {model.hw_compat_note}[/]"
        if self._avail_gb is not None and model.size_gb:
            if model.size_gb > self._total_gb:
                base = f"[red]⚠  {model.size_gb:.0f}/{self._total_gb:.0f} GB VRAM[/]"
                hints = model.vram_fit_hints(self._total_gb, SERVERS)
                if hints:
                    hint_str = "  [dim]·[/]  ".join(
                        f"[bright_green]{RUNTIME_QUANT_SHORT[rq]} ~{est:.0f} GB[/]"
                        for rq, est in hints
                    )
                    base += f"  [dim]·[/]  {hint_str}"
                return base
            if model.size_gb > self._avail_gb:
                return (
                    f"[yellow]⚠  {model.size_gb:.0f} GB"
                    f"  ·  {self._total_gb:.0f} GB total"
                    f"  ·  {self._avail_gb:.0f} GB free[/]"
                )
        # Build info line from structured fields + user note
        parts: list[str] = []
        return "  [dim]·[/]  ".join(parts)

    def _refresh_note_bar(self) -> None:
        """Update the note bar below the table for the focused model."""
        model = self._focused_model()
        bar = self.query_one("#status-bar-right", Label)
        bar.update(self._note_bar_text(model) if model is not None else "")

    @on(DataTable.HeaderSelected)
    def on_header(self, event: DataTable.HeaderSelected) -> None:
        """Promote a column to primary sort key, or toggle its direction.

        Only columns present in _SORT_KEY are sortable.  Clicking the header
        of any absent column (e.g. 'warn') is silently ignored.
        """
        col = event.column_key.value
        if col not in _SORT_KEY:
            return
        if self._sort_keys and self._sort_keys[0][0] == col:
            # Already primary — toggle direction.
            self._sort_keys[0] = (col, not self._sort_keys[0][1])
        else:
            # Promote to primary (ascending); retain others in existing order.
            self._sort_keys = [(col, False)] + [
                (c, r) for c, r in self._sort_keys if c != col
            ]
        self._sort_models()
        self._populate()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Enable cache/uncache/favorite/quantize only when a model is focused and state matches."""
        if action in ("cache_selected", "uncache_selected", "mark_favorite", "unmark_favorite",
                      "quantize_selected"):
            model = self._focused_model()
            if model is None:
                return False
            if action == "cache_selected":    return model.cached_in(CACHE_MOUNTS) is None
            if action == "uncache_selected":  return model.cached_in(CACHE_MOUNTS) is not None
            if action == "mark_favorite":     return not model.favorite
            if action == "unmark_favorite":   return model.favorite
            if action == "quantize_selected": return bool(tools_for_model(QUANTIZATION_TOOLS, model))
        return True

    def on_screen_resume(self) -> None:
        """Refresh table whenever this screen becomes active again."""
        self._sort_models()
        self._populate()

    @on(DataTable.RowHighlighted)
    def on_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self.refresh_bindings()
        self._refresh_note_bar()

    def action_cache_selected(self) -> None:
        model = self._focused_model()
        if model is None:
            return
        self.app.push_screen(CacheActionScreen(model))

    def action_uncache_selected(self) -> None:
        model = self._focused_model()
        if model is None:
            return
        cached = model.cached_in(CACHE_MOUNTS)
        if not cached:
            self.notify("Not cached locally", severity="warning", timeout=3)
            return
        try:
            launcher.uncache_model(model, cached)
            launcher.save_cached_path(model, "")
            self.notify(f"Removed from {cached.label}", timeout=3)
            self._populate()
        except Exception as e:
            self.notify(str(e), severity="error", timeout=5)

    def action_dt_back(self) -> None:
        if len(self.app.screen_stack) > 2:
            self.app.pop_screen()
        else:
            self.app.exit(None)

    def _set_favorite(self, value: bool) -> None:
        model = self._focused_model()
        if model is None:
            return
        try:
            launcher.save_favorite(model, value)
        except Exception as e:
            self.notify(str(e), severity="error", timeout=5)
            return
        verb = "added to" if value else "removed from"
        self.notify(f"\u2605 {model.name} {verb} favorites", timeout=2)
        self._sort_models()
        self._populate(snap_to=model)

    def action_mark_favorite(self) -> None:
        self._set_favorite(True)

    def action_unmark_favorite(self) -> None:
        self._set_favorite(False)

    def action_set_ctx_override(self) -> None:
        model = self._focused_model()
        if model is not None:
            self.app.push_screen(
                ContextSettingsScreen(model),
                callback=lambda _: self._sort_models() or self._populate(),
            )

    def _open_model(self, model: AIModel) -> None:
        """Navigate to the next screen for the selected model. Override in subclasses."""
        self.app.push_screen(ServerSelectScreen(model))

    def action_dt_select(self) -> None:
        model = self._focused_model()
        if model is not None:
            self._open_model(model)

    def action_quantize_selected(self) -> None:
        model = self._focused_model()
        if model is None:
            return
        applicable = tools_for_model(QUANTIZATION_TOOLS, model)
        if not applicable:
            self.notify("No quantization tools available for this format", timeout=3)
            return
        if len(applicable) == 1:
            tool = applicable[0]
            if not launcher.quant_tool_available(tool):
                if tool.pip_package:
                    self.app.push_screen(QuantToolInstallScreen(model, tool))
                else:
                    self.notify(tool.install_note or f"{tool.label} is not available.", timeout=8)
                return
            self.app.push_screen(QuantizeOptionScreen(model, tool))
        else:
            self.app.push_screen(QuantizeToolScreen(model, applicable))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        model_id = event.row_key.value
        model = next((m for m in self._models if m.id == model_id), None)
        if model is not None:
            self._open_model(model)


# ---------------------------------------------------------------------------
# Context settings — inline editor for a model's ctx_override and rope_scaling
# ---------------------------------------------------------------------------

class ContextSettingsScreen(Screen):
    """Edit the focused model's launch context: a YaRN factor and/or a ctx override.

    YaRN factor: blank or 1 = off; otherwise RoPE-scales the native ctx (e.g. 4 →
    256K × 4 = 1M) on servers that support it.  Ctx override: a plain integer or a
    K-suffixed value (32K); blank or 0 = use the native ctx (scaled when YaRN is on).
    """

    BINDINGS: ClassVar = [
        Binding("escape", "app.pop_screen", "Cancel", key_display="esc"),
    ]

    def __init__(self, model: AIModel) -> None:
        super().__init__()
        self._model = model

    def compose(self) -> ComposeResult:
        m = self._model
        native = fmt_ctx(m.ctx)
        yield Header()
        yield Vertical(
            Label(
                f"[bold]Context settings for:[/]  {m.name}  ·  {m.format_tag}  ·  native {native}",
                id="screen-title",
            ),
            Label("YaRN factor  [dim](blank = off; e.g. 2 or 4 — static scaling may cost quality on short prompts)[/]"),
            Input(
                value=f"{m.rope_scaling.factor:g}" if m.rope_scaling else "",
                placeholder="e.g. 4",
                id="yarn-input",
            ),
            Label("Ctx override  [dim](blank = native, or native × YaRN factor)[/]"),
            Input(
                value=fmt_ctx(m.ctx_override) if m.ctx_override else "",
                placeholder="e.g. 32768 or 32K",
                id="ctx-input",
            ),
            Label("", id="ctx-preview"),
            Horizontal(
                Button("Set",    variant="success", id="btn-set"),
                Button("Clear",  variant="warning", id="btn-clear"),
                Button("Cancel", variant="default", id="btn-cancel"),
                id="confirm-buttons",
            ),
            id="confirm-body",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#yarn-input", Input).focus()
        self._update_preview()

    @staticmethod
    def _parse_ctx(text: str) -> int:
        """Parse '32768', '32K', '32k' → int. Returns 0 for empty/zero."""
        text = text.strip()
        if not text or text == "0":
            return 0
        if text.lower().endswith("k"):
            return int(float(text[:-1]) * 1024)
        return int(text)

    @staticmethod
    def _parse_factor(text: str) -> float:
        """Parse a YaRN factor; 0 = off (blank or 1).  Raises ValueError when below 1."""
        text = text.strip()
        if not text:
            return 0.0
        factor = float(text)
        if factor < 1:
            raise ValueError("YaRN factor must be at least 1")
        return 0.0 if factor == 1 else factor

    def _read(self) -> tuple[RopeScaling | None, int]:
        """The rope_scaling and ctx_override currently entered (raises ValueError when invalid)."""
        factor = self._parse_factor(self.query_one("#yarn-input", Input).value)
        limit = self._parse_ctx(self.query_one("#ctx-input", Input).value)
        current = self._model.rope_scaling
        if not factor:
            rope = None
        elif current is not None:
            rope = dataclasses.replace(current, factor=factor)  # keep method / original_ctx
        else:
            rope = RopeScaling(factor=factor)
        return rope, limit

    def _update_preview(self) -> None:
        label = self.query_one("#ctx-preview", Label)
        try:
            rope, limit = self._read()
        except ValueError as e:
            label.update(f"[red]{e}[/]")
            return
        m = self._model
        scaled = round(m.ctx * rope.factor) if rope else m.ctx
        launch = limit or scaled
        basis = f"{fmt_ctx(m.ctx)} × {rope.factor:g} YaRN" if rope else "native"
        text = f"Launch context: [bold]{fmt_ctx(launch)}[/]  [dim]({'override' if limit else basis})[/]"
        if launch > scaled:
            text += f"\n[yellow]⚠ exceeds {fmt_ctx(scaled)} ({basis}) — quality degrades past it[/]"
        if rope:
            servers = servers_for_model(SERVERS, m)
            scaling = [s.label for s in servers if m.format in s.rope_scaling_formats]
            native_only = [s.label for s in servers if m.format not in s.rope_scaling_formats]
            text += f"\n[dim]YaRN applied by: {', '.join(scaling) or 'none'}"
            if native_only:
                text += f"  ·  native ctx on: {', '.join(native_only)}"
            text += "[/]"
        label.update(text)

    @on(Input.Changed)
    def on_changed(self, event: Input.Changed) -> None:
        self._update_preview()

    def _apply(self) -> None:
        try:
            rope, limit = self._read()
        except ValueError as e:
            self.notify(str(e), severity="error", timeout=4)
            return
        try:
            launcher.save_rope_scaling(self._model, rope)
            launcher.save_ctx_override(self._model, limit)
        except Exception as e:
            self.notify(str(e), severity="error", timeout=5)
            return
        self.notify(f"Launch context: {fmt_ctx(self._model.effective_ctx)}", timeout=3)
        self.app.pop_screen()

    @on(Input.Submitted)
    def on_submitted(self, event: Input.Submitted) -> None:
        self._apply()

    @on(Button.Pressed, "#btn-set")
    def on_set(self) -> None:
        self._apply()

    @on(Button.Pressed, "#btn-clear")
    def on_clear(self) -> None:
        self.query_one("#yarn-input", Input).value = ""
        self.query_one("#ctx-input", Input).value = ""
        self._apply()

    @on(Button.Pressed, "#btn-cancel")
    def on_cancel(self) -> None:
        self.app.pop_screen()


# ---------------------------------------------------------------------------
# ModelManageScreen — catalog management without model selection / launch
# ---------------------------------------------------------------------------

class ModelManageScreen(ModelSelectScreen):
    """Model catalog management view for stage_all_models AI types (e.g. Image Generation).

    Identical to ModelSelectScreen except:
    - No model-select / launch action (→ / Enter are suppressed).
    - Title reads "Manage Models" instead of "Select Model".
    - Backing out returns to the ImageGenHubScreen.

    Catalog management bindings (c/f/g/x/?) work; l/z are suppressed (text-gen only).
    """

    # Drop set_ctx_override (text-gen concept, ctx=0 for all image-gen models) and
    # quantize_selected (exl2-convert matches BF16/FP16 formats but produces text-gen
    # EXL2 output — not applicable to diffusion models).
    # Textual merges BINDINGS across the full MRO, so filtering the list has no effect.
    # check_action is the correct mechanism to hide bindings inherited from the parent.
    _MANAGE_HIDDEN: ClassVar[frozenset[str]] = frozenset(
        {"set_ctx_override", "quantize_selected"}
    )

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action in self._MANAGE_HIDDEN:
            return False
        return super().check_action(action, parameters)

    def _open_model(self, model: AIModel) -> None:
        self.app.push_screen(CacheActionScreen(model))

    def _title_text(self) -> str:
        return f"[bold]{AI_TYPE_LABELS[self._ai_type]} \u2014 manage models:[/]"


# ---------------------------------------------------------------------------
# Level 1 — AI type selection
# ---------------------------------------------------------------------------

class AITypeScreen(ButtonMenuScreen):
    """Choose which kind of AI to work with (Text Generation, Image, …)."""

    BINDINGS: ClassVar = [
        Binding("q",             "quit_app",       "Quit"),
        Binding("escape,b,left", "back",           "Back",   key_display="esc/←"),
        Binding("enter,right",   "press_focused",  "Select", key_display="↵/→/Space"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._types = available_ai_types(AI_MODELS)

    def action_back(self) -> None:                   # AITypeScreen root-screen back
        if len(self.app.screen_stack) > 2:
            self.app.pop_screen()
        else:
            self.app.exit(None)

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("[bold]Select AI type:[/]", id="menu-title")
        buttons = [
            Button(AI_TYPE_LABELS[t], id=f"btn-{t.value}", variant="primary")
            for t in self._types
        ]
        yield Vertical(*buttons, id="main-menu-buttons")
        yield Footer()

    def on_mount(self) -> None:
        self.query("Button").first().focus()

    @on(Button.Pressed)
    def on_button(self, event: Button.Pressed) -> None:
        bid = event.button.id
        for t in self._types:
            if bid == f"btn-{t.value}":
                if uses_hub_flow(SERVERS, t):
                    self.app.push_screen(ImageGenHubScreen(t))
                else:
                    self.app.push_screen(ModelSelectScreen(t))
                return


# ---------------------------------------------------------------------------
# ImageGenHubScreen — hub for AI types that use stage_all_models
# ---------------------------------------------------------------------------

class ImageGenHubScreen(ButtonMenuScreen):
    """Hub shown when an AI type uses stage_all_models (e.g. Image Generation).

    Separates the two distinct concerns:
    - Launch Server   → ServerSelectScreen (no per-launch model selection)
    - Manage Models   → ModelManageScreen  (cache/uncache/favorite/status; no launch)
    """

    BINDINGS: ClassVar = [
        Binding("q",             "quit_app",       "Quit"),
        Binding("escape,b,left", "app.pop_screen", "Back",   key_display="esc/←"),
        Binding("enter,right",   "press_focused",  "Select", key_display="↵/→/Space"),
    ]

    def __init__(self, ai_type: AIType) -> None:
        super().__init__()
        self._ai_type = ai_type

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label(
            f"[bold]{AI_TYPE_LABELS[self._ai_type]}[/]",
            id="menu-title",
        )
        yield Vertical(
            Button("Launch Server",  id="btn-launch",  variant="primary"),
            Button("Manage Models",  id="btn-manage",  variant="primary"),
            id="main-menu-buttons",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.query("Button").first().focus()

    @on(Button.Pressed, "#btn-launch")
    def on_launch(self, event: Button.Pressed) -> None:
        self.app.push_screen(ServerSelectScreen(model=None, ai_type=self._ai_type))

    @on(Button.Pressed, "#btn-manage")
    def on_manage(self, event: Button.Pressed) -> None:
        self.app.push_screen(ModelManageScreen(self._ai_type))


# ---------------------------------------------------------------------------
# Cache management — per-model cache/uncache actions
# ---------------------------------------------------------------------------

class CacheActionScreen(ButtonMenuScreen):
    """Cache or uncache a specific model."""

    BINDINGS: ClassVar = [
        Binding("q",             "quit_app",       "Quit"),
        Binding("escape,b,left", "app.pop_screen", "Back",   key_display="esc/←"),
        Binding("enter,right",   "press_focused",  "Select", key_display="↵/→/Space"),
    ]

    def __init__(self, model: AIModel) -> None:
        super().__init__()
        self._model = model
        self._source_gb: float | None = None  # actual on-disk size; computed async on mount

    def compose(self) -> ComposeResult:
        color, _sym = self._model.composite_status(SERVERS)
        yield Header()
        yield Label(
            f"[bold][{color}]{self._model.name}  ·  {self._model.format_tag}[/][/bold]"
            f"  ·  {self._model.size_gb:.0f} GB  ·  {fmt_ctx(self._model.effective_ctx)} ctx"
            + (
                f"[dim]  (native: {fmt_ctx(self._model.ctx)})[/]"
                if self._model.effective_ctx != self._model.ctx
                else ""
            ),
            id="screen-title",
        )
        cached = self._model.cached_in(CACHE_MOUNTS)
        if cached:
            status_txt = f"[dim]Cached in:[/]  [bright_green]{cached.label}[/]"
        else:
            status_txt = "[dim]Not cached locally \u2014 would load from source.[/]"
        yield Label(status_txt, id="cache-status-label")
        yield Vertical(id="cache-ops")
        yield Footer()

    def on_mount(self) -> None:
        self._build_buttons()
        self._fetch_source_size()

    async def on_screen_resume(self) -> None:
        """Rebuild status and buttons after returning from CacheProgressScreen."""
        cached = self._model.cached_in(CACHE_MOUNTS)
        status_txt = (
            f"[dim]Cached in:[/]  [bright_green]{cached.label}[/]"
            if cached else
            "[dim]Not cached locally \u2014 would load from source.[/]"
        )
        self.query_one("#cache-status-label", Label).update(status_txt)
        ops = self.query_one("#cache-ops", Vertical)
        await ops.remove_children()
        self._build_buttons()

    @work(thread=True)
    def _fetch_source_size(self) -> None:
        """Compute the actual on-disk size of the model source path (may be slow on remote storage)."""
        size_gb = launcher.cache_source_size_gb(self._model)
        self.app.call_from_thread(self._on_source_size_ready, size_gb)

    def _on_source_size_ready(self, source_gb: float) -> None:
        """Called on main thread when the source size computation finishes."""
        self._source_gb = source_gb
        self.call_later(self._rebuild_buttons_with_size)

    async def _rebuild_buttons_with_size(self) -> None:
        ops = self.query_one("#cache-ops", Vertical)
        await ops.remove_children()
        self._build_buttons()

    def _build_buttons(self) -> None:
        ops = self.query_one("#cache-ops", Vertical)
        cached = self._model.cached_in(CACHE_MOUNTS)
        # Use actual on-disk source size if computed; fall back to catalog size_gb.
        source_gb = self._source_gb if self._source_gb is not None else self._model.size_gb
        for mount in sorted(CACHE_MOUNTS, key=lambda m: m.priority):
            total, free = launcher.cache_disk_info(mount)
            fits = source_gb <= free
            fit_color = "green" if fits else "red"
            is_here = cached is not None and cached.root == mount.root
            if is_here:
                ops.mount(Button(
                    f"Remove from {mount.label}",
                    id=f"btn-uncache-{mount.priority}",
                    variant="warning",
                ))
            else:
                size_label = f"{source_gb:.0f} GB" if self._source_gb is not None else f"~{source_gb:.0f} GB"
                ops.mount(Button(
                    f"Cache to {mount.label}  "
                    f"[dim](need {size_label}  ·  "
                    f"[{fit_color}]{free:.0f} GB free[/]  ·  {total:.0f} GB total)[/]",
                    id=f"btn-cache-{mount.priority}",
                    variant="primary" if fits else "default",
                ))
        ops.mount(Button("Back", id="btn-back", variant="default"))
        self.call_after_refresh(lambda: self.query("#cache-ops Button").first().focus())

    @on(Button.Pressed)
    def on_btn(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "btn-back":
            self.app.pop_screen()
        elif bid.startswith("btn-cache-"):
            priority = int(bid.rsplit("-", 1)[-1])
            mount = next(m for m in CACHE_MOUNTS if m.priority == priority)
            self.app.push_screen(CacheProgressScreen(self._model, mount, source_gb=self._source_gb))
        elif bid.startswith("btn-uncache-"):
            priority = int(bid.rsplit("-", 1)[-1])
            mount = next(m for m in CACHE_MOUNTS if m.priority == priority)
            try:
                launcher.uncache_model(self._model, mount)
                launcher.save_cached_path(self._model, "")
                self.notify(f"Removed from {mount.label}", timeout=3)
                self.app.pop_screen()
            except Exception as e:
                self.notify(str(e), severity="error", timeout=5)


# ---------------------------------------------------------------------------
# Cache management — Level 3: rsync progress
# ---------------------------------------------------------------------------

class CacheProgressScreen(Screen):
    """Shows elapsed time while rsync copies a model to a local cache mount."""

    BINDINGS: ClassVar = [
        Binding("escape,q,b,left", "cancel", "Cancel", key_display="esc/←"),
    ]

    def __init__(self, model: AIModel, mount: CacheMount, source_gb: float | None = None) -> None:
        super().__init__()
        self._model = model
        self._mount = mount
        self._source_gb = source_gb
        self._proc: "subprocess.Popen | None" = None  # type: ignore[type-arg]
        self._dest_path: str = ""

    def compose(self) -> ComposeResult:
        size_str = f"{self._source_gb:.0f} GB" if self._source_gb is not None else f"{self._model.size_gb:.0f} GB"
        yield Header()
        yield Label(
            f"[bold]Caching:[/]  {self._model.name}  ·  {self._model.format_tag}"
            f"  ({size_str})  \u2192  {self._mount.label}",
            id="cache-op-title",
        )
        yield ProgressBar(total=100.0, show_eta=False, id="cache-progress-bar")
        yield Label("", id="cache-progress-detail")
        yield Label("[dim]Starting rsync\u2026[/]", id="cache-ops-status")
        yield Horizontal(
            Button("Cancel", id="btn-cancel", variant="error"),
            id="cache-op-buttons",
        )
        yield Footer()

    def on_mount(self) -> None:
        self._run_cache()

    @work(thread=True)
    def _run_cache(self) -> None:
        import os
        import select
        import time
        proc, dest_path = launcher.cache_model(self._model, self._mount)
        self._proc = proc
        self._dest_path = dest_path
        buf = b""
        start = time.monotonic()
        while True:
            rlist, _, _ = select.select([proc.stdout], [], [], 0.5)
            elapsed = int(time.monotonic() - start)
            mins, secs = divmod(elapsed, 60)
            self.app.call_from_thread(
                self._update_status,
                f"[dim]{mins}:{secs:02d} elapsed[/]",
            )
            if rlist:
                chunk = proc.stdout.read1(4096)  # type: ignore[union-attr]
                if not chunk:  # EOF
                    break
                buf += chunk
                # rsync --info=progress2 uses \r to overwrite each progress line;
                # newlines appear in the final summary — treat both as line breaks.
                parts = buf.replace(b"\n", b"\r").split(b"\r")
                buf = parts[-1]  # keep last incomplete fragment
                for part in reversed(parts[:-1]):
                    line = part.decode(errors="replace").strip()
                    if line:
                        self.app.call_from_thread(self._update_progress_line, line)
                        break
            elif proc.poll() is not None:
                break  # exited, no more data
        # Ensure returncode is populated (returns immediately — process already exited).
        proc.wait()
        # Read stderr output (safe: process is no longer writing).
        stderr_text = ""
        if proc.stderr:
            stderr_text = proc.stderr.read().decode(errors="replace").strip()
        self.app.call_from_thread(self._on_done, proc.returncode == 0, stderr_text)

    def _update_status(self, text: str) -> None:
        self.query_one("#cache-ops-status", Label).update(text)

    def _update_progress_line(self, line: str) -> None:
        import re
        m = re.search(r'(\d+)%\s+([\d.]+\S+/s)\s+([\d:]+)', line)
        if m:
            pct, speed, eta = int(m.group(1)), m.group(2), m.group(3)
            self.query_one("#cache-progress-bar", ProgressBar).progress = float(pct)
            detail = f"[dim]{speed}  ETA {eta}[/]" if pct < 100 else f"[dim]{speed}[/]"
            self.query_one("#cache-progress-detail", Label).update(detail)

    def _on_done(self, success: bool, stderr_text: str = "") -> None:
        if success:
            launcher.save_cached_path(self._model, self._dest_path)
            self.query_one("#cache-ops-status", Label).update("[bright_green]\u2713 Complete[/]")
            btn = self.query_one("#btn-cancel", Button)
            btn.label = "Done"   # type: ignore[assignment]
            btn.variant = "success"
        else:
            self.query_one("#cache-ops-status", Label).update("[bright_red]\u2717 rsync failed[/]")
            if stderr_text:
                self.query_one("#cache-progress-detail", Label).update(
                    f"[bright_red]{stderr_text}[/]"
                )
            btn = self.query_one("#btn-cancel", Button)
            btn.label = "Close"  # type: ignore[assignment]

    def action_cancel(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self._proc.wait()
        self.app.pop_screen()

    @on(Button.Pressed, "#btn-cancel")
    def on_cancel_btn(self) -> None:
        self.action_cancel()
