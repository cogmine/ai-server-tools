"""
params.py — Typed launch parameters: the schema behind each server's settings.

Each ServerSoftware declares its launch parameters as ParamSpec instances (data.py).
A spec is the single source of truth for a parameter: its default, a description,
how a value is shown and edited, which models it applies to and — for "auto"
parameters — how the value is resolved at launch.  Defaults, the model properties
page and launch-time resolution all derive from the same declaration.

Editing is kind-specific but UI-agnostic: a kind either offers a fixed set of
options() (picked from a list, or cycled with Space) or parses free text (parse()).

Adding a kind: subclass ParamSpec, implement format(), and override options() or
parse()/to_text().
"""
from __future__ import annotations

import shlex
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from launcher.models import AIModel

# Resolves an "auto" value for a launch: (model, merged params) → (value, reason).
AutoResolver = Callable[["AIModel | None", dict[str, Any]], tuple[bool, str]]


@dataclass(frozen=True, kw_only=True)
class ParamSpec(ABC):
    """One launch parameter of a server."""

    key: str
    label: str
    default: Any
    help: str = ""
    # Internal plumbing (e.g. staging directories from config.toml) — not shown per model.
    hidden: bool = False
    # Set by another part of the launch flow; shown read-only with this explanation.
    managed_by: str = ""
    # Shown for a model only when this returns True (None = every model).
    applies_to: Callable[["AIModel"], bool] | None = None

    def applies(self, model: "AIModel") -> bool:
        return self.applies_to is None or self.applies_to(model)

    def default_value(self) -> Any:
        """A fresh copy of the default (safe to mutate, compares equal to stored values)."""
        return self.default

    @property
    def editable(self) -> bool:
        return not self.managed_by

    @abstractmethod
    def format(self, value: Any) -> str:
        """Display form of a value."""

    def options(self) -> tuple[tuple[Any, str], ...] | None:
        """(value, label) choices for picker-edited kinds; None for text-edited kinds."""
        return None

    def parse(self, text: str) -> Any:
        """Value for user-entered text (raises ValueError when invalid)."""
        raise NotImplementedError(f"{type(self).__name__} is edited by picking an option")

    def to_text(self, value: Any) -> str:
        """Initial text for editing *value*."""
        return self.format(value)

    def cycle(self, value: Any) -> Any:
        """The option after *value* (wrapping); picker kinds only."""
        options = [v for v, _ in self.options() or ()]
        if not options:
            raise NotImplementedError(f"{type(self).__name__} cannot be cycled")
        index = options.index(value) if value in options else -1
        return options[(index + 1) % len(options)]


@dataclass(frozen=True, kw_only=True)
class BoolParam(ParamSpec):
    def format(self, value: Any) -> str:
        return "on" if value else "off"

    def options(self) -> tuple[tuple[Any, str], ...]:
        return ((False, "off"), (True, "on"))


@dataclass(frozen=True, kw_only=True)
class AutoBoolParam(ParamSpec):
    """auto / on / off.  None ("auto") is resolved per launch by *resolve*."""

    default: Any = None
    resolve: AutoResolver

    def format(self, value: Any) -> str:
        return "auto" if value is None else ("on" if value else "off")

    def options(self) -> tuple[tuple[Any, str], ...]:
        return ((None, "auto"), (True, "on"), (False, "off"))


@dataclass(frozen=True, kw_only=True)
class IntParam(ParamSpec):
    """Integer within bounds; optionally None (server default) or keyword values such as "auto"."""

    minimum: int | None = None
    maximum: int | None = None
    # When set, blank input means None, shown with this label (e.g. "llama.cpp default").
    none_label: str = ""
    keywords: tuple[str, ...] = ()

    def format(self, value: Any) -> str:
        if value is None:
            return self.none_label or "—"
        return str(value)

    def to_text(self, value: Any) -> str:
        return "" if value is None else str(value)

    def parse(self, text: str) -> Any:
        text = text.strip()
        if not text and self.none_label:
            return None
        if text in self.keywords:
            return text
        try:
            value = int(text)
        except ValueError:
            allowed = " or ".join(["a whole number", *(repr(k) for k in self.keywords)])
            raise ValueError(f"{self.label}: enter {allowed}") from None
        if self.minimum is not None and value < self.minimum:
            raise ValueError(f"{self.label} must be at least {self.minimum}")
        if self.maximum is not None and value > self.maximum:
            raise ValueError(f"{self.label} must be at most {self.maximum}")
        return value


@dataclass(frozen=True, kw_only=True)
class FloatParam(ParamSpec):
    minimum: float | None = None
    maximum: float | None = None

    def format(self, value: Any) -> str:
        return f"{value:g}"

    def parse(self, text: str) -> Any:
        try:
            value = float(text)
        except ValueError:
            raise ValueError(f"{self.label}: enter a number") from None
        if self.minimum is not None and value < self.minimum:
            raise ValueError(f"{self.label} must be at least {self.minimum:g}")
        if self.maximum is not None and value > self.maximum:
            raise ValueError(f"{self.label} must be at most {self.maximum:g}")
        return value


@dataclass(frozen=True, kw_only=True)
class ChoiceParam(ParamSpec):
    """One of a fixed set of string values; *labels* names special values (e.g. "" → "auto")."""

    choices: tuple[str, ...]
    labels: tuple[tuple[str, str], ...] = ()

    def format(self, value: Any) -> str:
        return dict(self.labels).get(value, str(value))

    def options(self) -> tuple[tuple[Any, str], ...]:
        return tuple((c, self.format(c)) for c in self.choices)


@dataclass(frozen=True, kw_only=True)
class TextParam(ParamSpec):
    """Free text; *empty_label* names the blank value (e.g. "none")."""

    empty_label: str = ""

    def format(self, value: Any) -> str:
        return str(value) if value else (self.empty_label or "—")

    def to_text(self, value: Any) -> str:
        return str(value or "")

    def parse(self, text: str) -> Any:
        return text.strip()


@dataclass(frozen=True, kw_only=True)
class FlagsParam(ParamSpec):
    """Extra command-line arguments, edited as shell text."""

    default: Any = ()

    def default_value(self) -> Any:
        return list(self.default)

    def format(self, value: Any) -> str:
        return shlex.join(value) if value else "none"

    def to_text(self, value: Any) -> str:
        return shlex.join(value) if value else ""

    def parse(self, text: str) -> Any:
        try:
            return shlex.split(text)
        except ValueError as e:
            raise ValueError(f"{self.label}: {e}") from None
