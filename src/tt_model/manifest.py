# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""The tt-model manifest: one YAML file, the whole authoring interface.

``tt-model package`` takes exactly one argument — a path to this file. There are no
per-field CLI flags. Authors may commit the file next to their model in the tt-metal
fork (recommended: the serving recipe is then reviewed in the same PR as the model
code) or keep it anywhere.

Schema version 1. ``package`` rewrites the manifest before staging: every git ref is
resolved to a commit sha and a ``built:`` block is added (image tag, resolved versions,
timestamps, per-path digests of the shipped code). The published manifest is therefore
fully pinned provenance even when the author wrote a branch name or a local path.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = 1

# Architectures a fork build can target. The arch is baked into the image by the build,
# which is why it lives at the manifest's top level while everything hardware-shaped
# lives on the serve profiles: every profile of one manifest shares the arch.
ARCHES = ("blackhole", "wormhole_b0")

# The plugin's closed MESH_DEVICE table (vllm_tt_plugin/utils/dp_discovery.py). A value
# outside this table (other than a literal "(rows, cols)" tuple) makes the plugin raise
# at boot — so we refuse it at manifest load instead, ~10 minutes earlier.
MESH_DEVICE_PRESETS: Dict[str, tuple] = {
    "N150": (1, 1),
    "P100": (1, 1),
    "P150": (1, 1),
    "P150x2": (1, 2),
    "N300": (1, 2),
    "P300": (1, 2),
    "N150x4": (1, 4),
    "P150x4": (1, 4),
    "T3K": (1, 8),
    "P150x8": (1, 8),
    "P300x2": (1, 4),
    "TG": (4, 8),
    "BH-Galaxy": (4, 8),
}

# Chips per *board*, keyed by the base of a `hardware` label. A label like "p150x4"
# means <base>×<multiplier>: 4 p150 boards → 4 chips; "p300x2" → 2 dual-chip boards
# → 4 chips. This is the only relationship tt-model asserts between `hardware` (the
# author-owned target label) and `mesh_device` (the string the plugin consumes) —
# deriving one from the other is not possible in general (P150x4 and P300x2 are both
# a (1, 4) mesh), which is exactly why the author states both.
_BOARD_CHIPS = {"p100": 1, "p150": 1, "n150": 1, "e150": 1, "p300": 2, "n300": 2}

_HARDWARE_RE = re.compile(r"^(?P<base>[a-z]\d+[a-z]?)(?:x(?P<mult>\d+))?$")


class ManifestError(ValueError):
    """A manifest that must not proceed. The message is the user-facing diagnosis."""


def hardware_chip_count(hardware: str) -> Optional[int]:
    """Chip count implied by a `hardware` label, or None for an unrecognised base.

    Unrecognised bases are allowed (the label is author-owned); they simply skip the
    mesh cross-check rather than failing it.
    """
    m = _HARDWARE_RE.match(hardware.strip().lower())
    if not m:
        return None
    base = m.group("base")
    # a trailing board-revision letter (p150a, p300c ...) does not change the chip count
    per_board = _BOARD_CHIPS.get(base) or _BOARD_CHIPS.get(re.sub(r"[a-z]$", "", base))
    if per_board is None:
        return None
    return per_board * int(m.group("mult") or 1)


def parse_mesh_device(value: str) -> tuple:
    """Resolve a MESH_DEVICE string to (rows, cols), mirroring the plugin's parser."""
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (tuple, list)) and len(parsed) == 2:
            return (int(parsed[0]), int(parsed[1]))
    except (ValueError, SyntaxError):
        pass
    if value in MESH_DEVICE_PRESETS:
        return MESH_DEVICE_PRESETS[value]
    raise ManifestError(
        f"invalid mesh_device: {value!r}. Expected one of "
        f"{sorted(MESH_DEVICE_PRESETS)} or a literal \"(rows, cols)\" tuple — this is "
        f"the plugin's own table (vllm_tt_plugin/utils/dp_discovery.py) and anything "
        f"else raises at boot."
    )


class GitSource(BaseModel):
    """A git repo + ref. `package` resolves `ref` to a commit sha before staging."""

    model_config = ConfigDict(extra="forbid")
    repo: str
    ref: str
    sha: Optional[str] = None  # filled in by `package`


class Source(BaseModel):
    """Build-time inputs. Nothing under `source:` is consulted at runtime."""

    model_config = ConfigDict(extra="forbid")

    # A local checkout path (default, hermetic: packages exactly the validated tree)
    # or a {repo, ref} to clone (reproducible from a sha; CI-friendly).
    tt_metal: Union[str, GitSource]

    # EXACTLY the model files that ship — an allowlist, never a denylist. These paths
    # (relative to the tt-metal tree) are staged to code/, uploaded to HF as browsable
    # files, and COPY'd into the image as the ONLY `models` package.
    code: List[str] = Field(min_length=1)

    ubuntu: str  # base image, e.g. "22.04"
    python: str  # interpreter, e.g. "3.12" — independent of ubuntu; uv provides it

    @field_validator("code")
    @classmethod
    def _no_absolute_code(cls, v: List[str]) -> List[str]:
        for p in v:
            if p.startswith("/") or ".." in Path(p).parts:
                raise ValueError(f"source.code paths must be relative to the tt-metal tree: {p!r}")
        return v


class ServeSettings(BaseModel):
    """Launch settings. `serve:` holds the defaults every profile inherits;
    each entry of `serve_profiles:` deep-merges over them (dicts merge, everything
    else overrides wholesale)."""

    model_config = ConfigDict(extra="forbid")

    hardware: Optional[str] = None      # device target label: p150, p150x2, p150x4 ...
    mesh_device: Optional[str] = None   # verbatim plugin string: P150x4, "(1, 4)" ...
    port: Optional[int] = None
    max_model_len: Optional[int] = None
    max_num_seqs: Optional[int] = None  # REQUIRED after merge: the TT backend's default fails
    block_size: Optional[int] = None    # REQUIRED after merge: the TT backend's default fails
    server_timeout: Optional[int] = None  # vllm-legacy launcher only
    additional_config: Dict[str, Any] = Field(default_factory=dict)
    args: List[Union[str, List[str]]] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)

    def flat_args(self) -> List[str]:
        """Flatten [--flag, [--opt, value]] into a single argv fragment."""
        out: List[str] = []
        for a in self.args:
            if isinstance(a, list):
                out.extend(str(x) for x in a)
            else:
                out.append(str(a))
        return out


class ServeProfile(ServeSettings):
    """A named, launchable configuration. One image serves all of a model's profiles —
    kernels are JIT-compiled against whatever mesh is opened at launch."""

    name: str
    description: Optional[str] = None


def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        elif v is not None and v != [] and v != {}:
            out[k] = v
    return out


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: int = Field(alias="schema", default=SCHEMA_VERSION)
    repo: str                 # HF repo this publishes to, e.g. hous/laguna-xs-2.1
    name: str
    weights: str              # HF weights id — downloaded to the HOST HF cache, never baked
    type: str                 # model type: see tt_model.types (vllm | vllm-legacy)
    arch: str                 # blackhole | wormhole_b0 — fixed by the build

    source: Source
    runtime: Dict[str, Any] = Field(default_factory=dict)  # shape is type-specific
    serve: ServeSettings = Field(default_factory=ServeSettings)
    serve_profiles: List[ServeProfile] = Field(min_length=1)
    default_profile: Optional[str] = None

    # Model-authored build-time assertions: each entry is a Python statement executed
    # inside the finished image (on top of the type's own import checks). This is where
    # a model catches its silent failure modes — e.g. laguna asserts its precision
    # config file shipped, because tt/model.py falls back to in-code defaults WITHOUT
    # ERROR when it is missing. An under-shipped image then fails on the author's
    # machine, not on a consumer's first boot.
    verify: List[str] = Field(default_factory=list)

    built: Optional[Dict[str, Any]] = None  # provenance; written by `package`

    # ---- profile resolution ------------------------------------------------------

    def profile_names(self) -> List[str]:
        return [p.name for p in self.serve_profiles]

    def resolved_default(self) -> str:
        if self.default_profile:
            return self.default_profile
        return self.serve_profiles[0].name

    def resolve_profile(self, name: Optional[str] = None) -> ServeProfile:
        """The fully merged profile: serve: defaults with the named profile on top."""
        wanted = name or self.resolved_default()
        for p in self.serve_profiles:
            if p.name == wanted:
                merged = _deep_merge(
                    self.serve.model_dump(exclude_none=True),
                    p.model_dump(exclude_none=True),
                )
                return ServeProfile.model_validate(merged)
        raise ManifestError(
            f"no serve profile named {wanted!r}; available: {', '.join(self.profile_names())}"
        )

    # ---- validation --------------------------------------------------------------

    def validate_semantics(self) -> None:
        """Everything pydantic's shape check cannot see. Raises ManifestError."""
        if self.schema_version != SCHEMA_VERSION:
            raise ManifestError(
                f"unsupported schema version {self.schema_version}; this tt-model "
                f"understands schema {SCHEMA_VERSION}"
            )
        if self.arch not in ARCHES:
            raise ManifestError(f"arch must be one of {ARCHES}, got {self.arch!r}")
        if "/" not in self.repo:
            raise ManifestError(f"repo must be a namespaced HF id (org/name), got {self.repo!r}")

        names = self.profile_names()
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ManifestError(f"duplicate serve profile names: {sorted(dupes)}")
        if len(names) > 1 and not self.default_profile:
            raise ManifestError(
                "a manifest with multiple serve profiles must name a default_profile — "
                "the author decides the default, not the consumer's luck. Profiles: "
                + ", ".join(names)
            )
        if self.default_profile and self.default_profile not in names:
            raise ManifestError(
                f"default_profile {self.default_profile!r} names no profile; "
                f"available: {', '.join(names)}"
            )

        for p in self.serve_profiles:
            merged = self.resolve_profile(p.name)
            where = f"serve profile {p.name!r}"
            for field in ("hardware", "mesh_device", "max_num_seqs", "block_size"):
                if getattr(merged, field) is None:
                    raise ManifestError(
                        f"{where}: {field} is required (set it on the profile or under serve:). "
                        + ("The TT backend rejects vLLM's own default."
                           if field in ("max_num_seqs", "block_size") else "")
                    )
            rows, cols = parse_mesh_device(merged.mesh_device)
            chips = hardware_chip_count(merged.hardware)
            if chips is not None and rows * cols != chips:
                raise ManifestError(
                    f"{where}: mesh_device {merged.mesh_device!r} opens a {rows}x{cols} mesh "
                    f"({rows * cols} chips) but hardware {merged.hardware!r} implies {chips}"
                )

        # Type-specific validation (runtime: shape, launcher requirements) lives with
        # the type. Imported here, lazily, to keep manifest.py free of that knowledge.
        from . import types  # noqa: PLC0415

        if self.type not in types.TYPES:
            raise ManifestError(
                f"unsupported model type {self.type!r}; supported: "
                + ", ".join(sorted(types.TYPES))
            )
        types.TYPES[self.type].validate(self)

    # ---- I/O -----------------------------------------------------------------------

    def to_yaml(self) -> str:
        data = self.model_dump(by_alias=True, exclude_none=True)
        return yaml.safe_dump(data, sort_keys=False, width=100)


def load_manifest(path: Union[str, Path]) -> Manifest:
    """Parse + fully validate a manifest file. The single trusted entry point."""
    p = Path(path)
    try:
        raw = yaml.safe_load(p.read_text())
    except FileNotFoundError:
        raise ManifestError(f"manifest not found: {p}") from None
    except yaml.YAMLError as e:
        raise ManifestError(f"{p} is not valid YAML: {e}") from None
    if not isinstance(raw, dict):
        raise ManifestError(f"{p} does not contain a YAML mapping")
    try:
        m = Manifest.model_validate(raw)
    except Exception as e:  # pydantic ValidationError → readable
        raise ManifestError(f"{p} failed validation:\n{e}") from None
    m.validate_semantics()
    return m
