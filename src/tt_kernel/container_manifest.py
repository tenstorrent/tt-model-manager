# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""The container (v5.1) package's AUTHORING interface: one YAML file, no per-field flags.

``tt-model package --container <file.yaml>`` takes exactly one input. Authors commit the
file next to their model in the tt-metal fork (recommended — the serving recipe is then
reviewed in the same PR as the model code) or keep it anywhere.

This module owns the *authored* document. It is deliberately NOT the published one:
``to_wire()`` renders a v5.1 :class:`~tt_kernel.manifest.Manifest`, and that JSON
(``tt_kernel_manifest.json``) is what lands on the Hub. Two reasons for the split:

* ``Manifest.from_json`` gates on ``SUPPORTED_SCHEMAS``, which is what makes an older
  tt-model refuse a newer package loudly instead of half-reading it. Every command
  already resolves a bundle by reading that one filename.
* YAML is a much better authoring surface than JSON (comments, block scalars, no commas)
  but a worse wire format. Authors get the former; the Hub gets the latter.

Validation is front-loaded: everything that can be known without hardware is checked at
load time, because the alternative is finding out ten minutes into a boot.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .manifest import (
    CONTAINER_SCHEMA,
    ContainerSpec,
    ImageRef,
    Manifest,
    ServeProfile,
    ServeSettings,
    WeightsRef,
)

# The authored schema — deliberately the SAME number as the published wire schema
# (``manifest.CONTAINER_SCHEMA``). An author writing ``schema: "5.1"`` and a consumer
# reading ``"schema_version": "5.1"`` are looking at one version of one format; having
# the authoring doc carry its own independent counter only raised the question "what is
# this 1?" without buying anything.
SCHEMA_VERSION = CONTAINER_SCHEMA

# Architectures a fork build can target. The arch is baked into the image by the build,
# which is why it lives at the top level while everything hardware-shaped lives on the
# serve profiles: every profile of one manifest shares the arch.
ARCHES = ("blackhole", "wormhole_b0")


# The profile synthesized for a manifest that declares none. A model with one
# configuration — the v5 idiom, where the repo name carries the target
# ("you/mymodel-blackholex1") — should not have to learn what a profile is: it writes a
# flat ``serve:`` block and that IS its single profile. The name only ever surfaces in
# `list` output and as the argument `--profile` would take.
DEFAULT_PROFILE_NAME = "default"

# The plugin's closed MESH_DEVICE table (vllm_tt_plugin/utils/dp_discovery.py). A value
# outside this table (other than a literal "(rows, cols)" tuple) makes the plugin raise
# at boot — so it is refused at manifest load instead, ~10 minutes earlier.
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
    # The same four chips as P300x2, opened as a square rather than a line. Some models
    # cannot use a line at all: FLUX.2 needs its sequence AND tensor parallel factors
    # both above 1, so on four chips 2x2 is its only legal geometry.
    "QB2": (2, 2),
    "TG": (4, 8),
    "BH-Galaxy": (4, 8),
}

# Chips per *board*, keyed by the base of a ``hardware`` label. A label like "p150x4"
# means <base>x<multiplier>: 4 p150 boards -> 4 chips; "p300x2" -> 2 dual-chip boards
# -> 4 chips. This is the ONLY relationship tt-model asserts between ``hardware`` (the
# board target label) and ``mesh_device`` (the string the plugin consumes) — deriving one
# from the other is not possible in general (P150x4 and P300x2 are both a (1, 4) mesh),
# which is exactly why the author states both.
#
# The label names BOARDS, never the box they sit in: a T3000 is "n300x4", a QB2 is
# "p300x2". Box names live in MESH_DEVICE_PRESETS above, and putting one here is refused
# by validate_semantics rather than silently publishing device_count: 1.
_BOARD_CHIPS = {"p100": 1, "p150": 1, "n150": 1, "e150": 1, "p300": 2, "n300": 2}

_HARDWARE_RE = re.compile(r"^(?P<base>[a-z]\d+[a-z]?)(?:x(?P<mult>\d+))?$")


class ContainerManifestError(ValueError):
    """A manifest that must not proceed. The message is the user-facing diagnosis."""


def hardware_chip_count(hardware: str) -> Optional[int]:
    """Chip count implied by a ``hardware`` label, or None for an unrecognised base.

    Total on purpose — it never raises — so ``to_wire`` can call it on a manifest that
    skipped validation. The *refusal* of an unrecognised label lives in
    ``validate_semantics`` instead: a label this cannot read is a label whose
    ``device_count`` would be invented and whose mesh cross-check would be skipped, and
    both failures are silent. See the comment on ``_BOARD_CHIPS`` for the grammar.
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
    """Resolve a MESH_DEVICE string to ``(rows, cols)``, mirroring the plugin's parser."""
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (tuple, list)) and len(parsed) == 2:
            return (int(parsed[0]), int(parsed[1]))
    except (ValueError, SyntaxError):
        pass
    if value in MESH_DEVICE_PRESETS:
        return MESH_DEVICE_PRESETS[value]
    raise ContainerManifestError(
        f"invalid mesh_device: {value!r}. Expected one of {sorted(MESH_DEVICE_PRESETS)} "
        f"or a literal \"(rows, cols)\" tuple — this is the plugin's own table "
        f"(vllm_tt_plugin/utils/dp_discovery.py) and anything else raises at boot."
    )


class GitSource(BaseModel):
    """A git repo + ref. ``package`` resolves ``ref`` to a commit sha before staging."""

    model_config = ConfigDict(extra="forbid")

    repo: str
    ref: str
    sha: Optional[str] = None  # filled in by `package`


class Source(BaseModel):
    """Build-time inputs. Nothing under ``source:`` is consulted at runtime."""

    model_config = ConfigDict(extra="forbid")

    # A local checkout path (the default, and hermetic: packages exactly the tree the
    # author validated) or a {repo, ref} to clone (reproducible from a sha; CI-friendly).
    tt_metal: Union[str, GitSource]

    # EXACTLY the model files that ship — an allowlist, never a denylist. These paths
    # (relative to the tt-metal tree) are staged to code/, uploaded to HF as browsable
    # files, and COPY'd into the image as the ONLY `models` package. Under-listing fails
    # the image's own build-time import check, on the author's machine.
    code: List[str] = Field(min_length=1)

    ubuntu: str  # base image, e.g. "22.04"
    python: str  # interpreter, e.g. "3.12" — independent of ubuntu; uv provides it

    @field_validator("code")
    @classmethod
    def _no_escaping_paths(cls, v: List[str]) -> List[str]:
        for p in v:
            if p.startswith("/") or ".." in Path(p).parts:
                raise ValueError(
                    f"source.code paths must be relative to the tt-metal tree: {p!r}"
                )
        return v


class WeightsSpec(BaseModel):
    """The weights, as a POINTER. Never baked into the image.

    ``revision`` matters more than it looks: without it a consumer downloads whatever the
    repo's default branch points at TODAY, which may not be what the author validated —
    the one input to a "fully pinned" package that was left floating. ``allow_patterns`` /
    ``ignore_patterns`` are passed straight to ``snapshot_download`` for repos carrying
    several formats where only one is wanted.
    """

    model_config = ConfigDict(extra="forbid")

    repo: str
    revision: Optional[str] = None
    allow_patterns: Optional[List[str]] = None
    ignore_patterns: Optional[List[str]] = None


class ImageSettings(BaseModel):
    """Where the built image is published. Defaults to riding inside the HF repo."""

    model_config = ConfigDict(extra="forbid")

    # "hf" => exploded OCI blobs under image/ in the model repo (self-contained, one
    # auth, xet layer dedupe). Anything else is an OCI registry namespace, e.g.
    # "ghcr.io/tenstorrent" — the repo then carries a pointer and `docker pull` works
    # for consumers who never install tt-model.
    registry: str = "hf"
    repository: Optional[str] = None  # name under a real registry; defaults to `name`


class CardSettings(BaseModel):
    """Optional model-authored Markdown for the generated model card."""

    model_config = ConfigDict(extra="forbid")

    # One or two sentences on what the model IS and what it is for ("intended for
    # agentic coding"). Leads the card, right under the title — the tool cannot know
    # this, so the author states it.
    description: Optional[str] = None
    quickstart: Optional[str] = None


class ContainerManifest(BaseModel):
    """The authored ``tt-model.yaml``."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    # The format version of this file. Written as a string ("5.1") so YAML cannot turn
    # it into a float and lose the distinction between 5.1 and 5.10.
    schema_version: str = Field(alias="schema", default=SCHEMA_VERSION)
    repo: str  # the HF repo this publishes to, e.g. you/my-model
    name: str
    # An HF id, or a WeightsSpec to pin a revision / select files. Downloaded to the
    # HOST HF cache at pull time; never baked into the image.
    weights: Union[str, WeightsSpec]
    kind: str = "vllm-plugin"  # launcher flavour; see tt_kernel.launchers.KINDS
    arch: str  # blackhole | wormhole_b0 — fixed by the build

    source: Source
    image: ImageSettings = Field(default_factory=ImageSettings)
    runtime: Dict[str, Any] = Field(default_factory=dict)  # shape is kind-specific
    serve: ServeSettings = Field(default_factory=ServeSettings)
    # Optional. Omit it entirely for a single-configuration model: `serve:` alone is then
    # the whole launch config, and one profile named "default" is synthesized from it.
    # Declare profiles when ONE image should serve several launch configs — different
    # device targets (p150x2 vs p150x4), or different deployment shapes on the same
    # hardware (one interactive user at full speed vs 32 concurrent).
    serve_profiles: List[ServeProfile] = Field(default_factory=list)
    default_profile: Optional[str] = None
    card: Optional[CardSettings] = None

    # Model-authored build-time assertions: each entry is a Python statement executed
    # inside the finished image, on top of the launcher's own import checks. This is
    # where a model catches its silent failure modes — e.g. a model whose precision
    # config falls back to in-code defaults WITHOUT ERROR when the file is missing
    # asserts that file is present. An under-shipped image then fails on the author's
    # machine, not on a consumer's first boot.
    verify: List[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _name_is_a_safe_slug(cls, v: str) -> str:
        # `name` becomes a host filesystem path (``~/.cache/tt-model/<name>/``, holding both
        # the JIT kernel cache and the converted-weight cache, which ``tt-model rm`` deletes
        # with rmtree) and a docker container name. A traversal or absolute name would let a
        # *pulled* package's ``rm`` escape the cache dir, so constrain it to a safe slug at
        # authoring time.
        import re
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", v):
            raise ContainerManifestError(
                f"name {v!r} is not a safe slug — use lowercase letters, digits and "
                "'.'/'_'/'-', starting alphanumeric (it becomes a cache path + container name)."
            )
        return v

    # ---- profile resolution -------------------------------------------------------

    def effective_profiles(self) -> List[ServeProfile]:
        """The declared profiles, or the single synthesized one for a flat manifest.

        Everything downstream — validation, resolution, and the published document —
        goes through this, so a flat manifest and a one-profile manifest are the same
        thing by construction rather than by two code paths agreeing.
        """
        return self.serve_profiles or [ServeProfile(name=DEFAULT_PROFILE_NAME)]

    @property
    def weights_repo(self) -> str:
        return self.weights if isinstance(self.weights, str) else self.weights.repo

    @property
    def weights_ref(self) -> WeightsRef:
        if isinstance(self.weights, str):
            return WeightsRef(repo=self.weights)
        return WeightsRef(
            repo=self.weights.repo,
            revision=self.weights.revision,
            allow_patterns=self.weights.allow_patterns,
            ignore_patterns=self.weights.ignore_patterns,
        )

    def profile_names(self) -> List[str]:
        return [p.name for p in self.effective_profiles()]

    def resolved_default(self) -> str:
        return self.default_profile or self.effective_profiles()[0].name

    def resolve_profile(self, name: Optional[str] = None) -> ServeProfile:
        """The fully merged profile: ``serve:`` defaults with the named profile on top."""
        return self._as_spec_for_resolution().resolve_profile(name)

    def _as_spec_for_resolution(self) -> ContainerSpec:
        # Profile merging is wire-side logic; reuse it rather than reimplementing it here
        # so the authored view and the published view can never disagree about a default.
        return ContainerSpec(
            image=ImageRef(tag="unresolved"),
            kind=self.kind,
            serve=self.serve,
            serve_profiles=self.effective_profiles(),
            default_profile=self.default_profile,
        )

    # ---- validation ---------------------------------------------------------------

    def validate_semantics(self) -> None:
        """Everything pydantic's shape check cannot see. Raises ContainerManifestError."""
        if str(self.schema_version) != SCHEMA_VERSION:
            raise ContainerManifestError(
                f"unsupported manifest schema {self.schema_version!r}; this tt-model "
                f"authors schema {SCHEMA_VERSION!r}"
            )
        if self.arch not in ARCHES:
            raise ContainerManifestError(f"arch must be one of {ARCHES}, got {self.arch!r}")
        from .launchers import KINDS, LauncherError, launcher_for

        try:
            launcher = launcher_for(self.kind)
        except LauncherError:
            raise ContainerManifestError(
                f"kind must be one of {tuple(sorted(KINDS))}, got {self.kind!r}"
            ) from None
        if "/" not in self.repo:
            raise ContainerManifestError(
                f"repo must be a namespaced HF id (org/name), got {self.repo!r}"
            )
        if "/" not in self.weights_repo:
            raise ContainerManifestError(
                f"weights must be a namespaced HF id (org/name), got {self.weights_repo!r}"
            )

        names = self.profile_names()
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ContainerManifestError(f"duplicate serve profile names: {dupes}")
        if len(names) > 1 and not self.default_profile:
            raise ContainerManifestError(
                "a manifest with multiple serve profiles must name a default_profile — "
                "the author decides the default, not the consumer's luck. Profiles: "
                + ", ".join(names)
            )
        if self.default_profile and self.default_profile not in names:
            raise ContainerManifestError(
                f"default_profile {self.default_profile!r} names no profile; "
                f"available: {', '.join(names)}"
            )

        # Which launch fields a profile must carry is the kind's business: max_num_seqs
        # and block_size describe a continuous-batching engine and mean nothing to, say,
        # a diffusion server. Ask the launcher rather than assuming every kind is vLLM.
        from .launchers import required_serve_fields

        required = required_serve_fields(self.kind)

        for p in self.effective_profiles():
            merged = self.resolve_profile(p.name)
            where = f"serve profile {p.name!r}"
            for field in required:
                if getattr(merged, field) is None:
                    hint = (
                        " The TT backend rejects vLLM's own default."
                        if field in ("max_num_seqs", "block_size")
                        else ""
                    )
                    raise ContainerManifestError(
                        f"{where}: {field} is required (set it on the profile or under "
                        f"serve:).{hint}"
                    )
            rows, cols = parse_mesh_device(merged.mesh_device)
            chips = hardware_chip_count(merged.hardware)
            # An unreadable label is refused rather than defaulted: to_wire would publish
            # device_count: 1 for it, and the cross-check below would be skipped — so a
            # 4-chip model would ship claiming one chip, with nothing said. `x0` lands here
            # too, because a zero count is falsy and would be rewritten to 1 the same way.
            if chips is None or chips < 1:
                hint = (
                    f" {merged.hardware!r} is a mesh_device SKU, not a board label."
                    if merged.hardware in MESH_DEVICE_PRESETS
                    else ""
                )
                raise ContainerManifestError(
                    f"{where}: hardware {merged.hardware!r} is not a recognised board "
                    f"label, so device_count and the mesh cross-check cannot be derived."
                    f"{hint} Expected <board>[xN] with board one of "
                    f"{sorted(_BOARD_CHIPS)} — a T3000 is 'n300x4', a QB2 is 'p300x2'."
                )
            if rows * cols != chips:
                raise ContainerManifestError(
                    f"{where}: mesh_device {merged.mesh_device!r} opens a {rows}x{cols} "
                    f"mesh ({rows * cols} chips) but hardware {merged.hardware!r} "
                    f"implies {chips}"
                )

        launcher.validate(self)

    def validate_sources_exist(self, root: Optional[Path] = None) -> None:
        """Every ``source.code`` entry must exist — a missing one is an ERROR, never a
        silent skip, because the failure it causes otherwise is an import error deep
        inside a multi-hour build (or worse, a silent fallback at serve time).

        Split out from ``validate_semantics`` so the manifest stays loadable, and fully
        checkable, on a machine that does not have the author's tt-metal tree — a
        consumer reading a published package, or the test suite.
        """
        if isinstance(self.source.tt_metal, GitSource):
            return  # a remote tree is not on disk yet; `package` checks after cloning
        base = Path(root) if root is not None else Path(self.source.tt_metal)
        missing = [c for c in self.source.code if not (base / c).exists()]
        if missing:
            raise ContainerManifestError(
                f"source.code lists {len(missing)} path(s) that do not exist under {base}: "
                + ", ".join(sorted(missing))
            )

    # ---- rendering to the published document ---------------------------------------

    def to_wire(
        self,
        *,
        image_tag: str,
        tt_metal_version: str,
        tt_kernel_version: str,
        built: Optional[Dict[str, Any]] = None,
        code_dir: Optional[str] = "code",
        digest: Optional[str] = None,
        hostname: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> Manifest:
        """Render the v5.1 :class:`Manifest` that gets published as JSON.

        ``package`` supplies the values the author cannot know: the built image tag, the
        resolved tt-metal version, and the ``built`` provenance block.
        """
        import datetime
        import socket

        from .manifest import Producer

        default = self.resolve_profile()
        return Manifest(
            schema_version=CONTAINER_SCHEMA,
            name=self.name,
            tt_metal_version=tt_metal_version,
            arch=self.arch,
            device_count=hardware_chip_count(default.hardware or "") or 1,
            build_key=None,  # kernels JIT inside the container into a mounted cache dir
            kernel_count=0,
            fast_path_kernels=None,
            producer=Producer(
                tt_kernel_version=tt_kernel_version,
                created_at=created_at
                or datetime.datetime.now(datetime.timezone.utc).isoformat(),
                hostname=hostname if hostname is not None else socket.gethostname(),
            ),
            weights=self.weights_ref,
            container=ContainerSpec(
                image=ImageRef(
                    registry=self.image.registry,
                    repository=self.image.repository or self.name,
                    tag=image_tag,
                    digest=digest,
                ),
                kind=self.kind,
                runtime=dict(self.runtime),
                serve=self.serve,
                # Always at least one profile on the wire, synthesized if the author
                # declared none: consumers, `list`, and `--profile` see ONE shape
                # regardless of which authoring style produced the package.
                serve_profiles=self.effective_profiles(),
                default_profile=self.default_profile,
                code_dir=code_dir,
                verify=list(self.verify),
                built=dict(built or {}),
            ),
        )


def load_container_manifest(
    path: Union[str, Path], *, check_sources: bool = False
) -> ContainerManifest:
    """Parse + fully validate an authored manifest. The single trusted entry point.

    ``check_sources`` additionally requires every ``source.code`` path to exist on this
    machine; ``package`` passes it, everything else does not.
    """
    try:
        import yaml
    except ModuleNotFoundError:  # pragma: no cover - dependency is declared
        raise ContainerManifestError(
            "PyYAML is required to read a container manifest; reinstall tt-model."
        ) from None

    p = Path(path)
    try:
        raw = yaml.safe_load(p.read_text())
    except FileNotFoundError:
        raise ContainerManifestError(f"manifest not found: {p}") from None
    except yaml.YAMLError as e:
        raise ContainerManifestError(f"{p} is not valid YAML: {e}") from None
    if not isinstance(raw, dict):
        raise ContainerManifestError(f"{p} does not contain a YAML mapping")
    try:
        m = ContainerManifest.model_validate(raw)
    except ContainerManifestError:
        raise
    except Exception as e:  # pydantic ValidationError -> readable
        raise ContainerManifestError(f"{p} failed validation:\n{e}") from None
    m.validate_semantics()
    if check_sources:
        m.validate_sources_exist()
    return m
