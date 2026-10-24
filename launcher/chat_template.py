"""chat_template.py — Read a model's chat template and apply launch-time patches.

Some chat templates reject conversations that are valid for the model itself
(e.g. Qwen 3.5+ raises "System message must be at the beginning." for any later
system message, although the model handles in-place ChatML system turns, which
Qwen2.5/Qwen3 templates rendered).  Patches are detected against the template
text and applied automatically at launch; a model can opt out of individual
patches (AIModel.disabled_template_patches).

Model files are never modified: the patched template is written to
user_data/chat_templates/<model.id>.jinja and passed to the server with its
template-override flag (ServerSoftware.chat_template_flag).

Adding a patch: append a ChatTemplatePatch to TEMPLATE_PATCHES.
"""
from __future__ import annotations

import functools
import json
import re
from dataclasses import dataclass
from pathlib import Path

from launcher.gguf_header import read_gguf_metadata
from launcher.models import AIModel

TEMPLATE_DIR = Path(__file__).parent / "user_data" / "chat_templates"

# HF template locations, in the precedence transformers uses.
_HF_TEMPLATE_FILES = ("chat_template.jinja", "chat_template.json", "tokenizer_config.json")


@dataclass(frozen=True)
class ChatTemplatePatch:
    """A textual fix for a chat template, applied wherever `pattern` matches.

    `requires` are further patterns that must all occur in the template, guarding
    against templates where `replacement` would not be valid (e.g. other syntax).
    """
    id: str              # stable key, persisted in AIModel.disabled_template_patches
    label: str           # short UI description
    description: str     # why the patch exists / what it changes
    pattern: re.Pattern[str]
    replacement: str     # inserted verbatim (no regex backreferences)
    requires: tuple[re.Pattern[str], ...] = ()

    def applies_to(self, template: str) -> bool:
        return bool(self.pattern.search(template)) and all(r.search(template) for r in self.requires)

    def apply(self, template: str) -> str:
        return self.pattern.sub(lambda _: self.replacement, template)


TEMPLATE_PATCHES: tuple[ChatTemplatePatch, ...] = (
    ChatTemplatePatch(
        id="system_anywhere",
        label="Allow system messages anywhere",
        description=(
            "The template rejects system messages after the start of the conversation. "
            "The patch renders them in place as ChatML system turns, as Qwen2.5/Qwen3 "
            "templates did; conversations with only a leading system message are unchanged."
        ),
        # Qwen 3.5+ (official and Unsloth GGUF variants): raised for every system
        # message not already rendered as the leading system block.
        pattern=re.compile(
            r"""\{\{-?\s*raise_exception\(\s*(['"])System message must be at the beginning\.?\1\s*\)\s*-?\}\}"""
        ),
        replacement="{{- '<|im_start|>system\\n' + content + '<|im_end|>\\n' }}",
        requires=(
            re.compile(re.escape("<|im_start|>")),
            # the message loop's rendered content, used by the replacement
            re.compile(r"\{%-?\s*set content\s*="),
        ),
    ),
)


def _gguf_template(path: Path) -> str | None:
    value = read_gguf_metadata(path).get("tokenizer.chat_template")
    return value if isinstance(value, str) else None


def _json_template(path: Path) -> str | None:
    value = json.loads(path.read_text(encoding="utf-8")).get("chat_template")
    if isinstance(value, list):
        # Named templates: the server's default is the one named "default".
        value = next((t.get("template") for t in value if t.get("name") == "default"), None)
    return value if isinstance(value, str) else None


@functools.lru_cache(maxsize=64)
def _read_template(path: str, mtime_ns: int) -> str | None:
    p = Path(path)
    if p.suffix == ".gguf":
        return _gguf_template(p)
    if p.suffix == ".jinja":
        return p.read_text(encoding="utf-8")
    return _json_template(p)


def read_template_at(root: Path) -> str | None:
    """The chat template of the model file/directory at *root*, or None if absent/unreadable."""
    if root.is_file():
        candidates = [root] if root.suffix == ".gguf" else []
    else:
        candidates = [root / name for name in _HF_TEMPLATE_FILES]
    for path in candidates:
        try:
            template = _read_template(str(path), path.stat().st_mtime_ns)
        except (OSError, ValueError):
            continue
        if template:
            return template
    return None


def read_chat_template(model: AIModel) -> str | None:
    """The chat template the server would use for *model*, or None if absent/unreadable."""
    return read_template_at(Path(model.path))


_TOOLS_REFERENCE = re.compile(r"\{[{%][^}]*\btools\b")


def declares_tools(template: str | None) -> bool:
    """True when the template renders a `tools` list, i.e. supports tool calling."""
    return bool(template and _TOOLS_REFERENCE.search(template))


def available_patches(model: AIModel) -> list[ChatTemplatePatch]:
    """Patches that apply to *model*'s template, including ones the model has disabled."""
    template = read_chat_template(model)
    if template is None:
        return []
    return [patch for patch in TEMPLATE_PATCHES if patch.applies_to(template)]


@dataclass(frozen=True)
class PatchedChatTemplate:
    """A model's chat template with its enabled patches applied."""
    model_id: str
    text: str
    patches: tuple[ChatTemplatePatch, ...]

    @property
    def path(self) -> Path:
        return TEMPLATE_DIR / f"{self.model_id}.jinja"

    def write(self) -> Path:
        """Write the template file (only when its content changed) and return its path."""
        path = self.path
        if not path.is_file() or path.read_text(encoding="utf-8") != self.text:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self.text, encoding="utf-8")
        return path


def patched_template(model: AIModel) -> PatchedChatTemplate | None:
    """*model*'s template with every applicable, enabled patch; None when none applies."""
    template = read_chat_template(model)
    if template is None:
        return None
    applied: list[ChatTemplatePatch] = []
    for patch in TEMPLATE_PATCHES:
        if patch.id not in model.disabled_template_patches and patch.applies_to(template):
            template = patch.apply(template)
            applied.append(patch)
    if not applied:
        return None
    return PatchedChatTemplate(model.id, template, tuple(applied))
