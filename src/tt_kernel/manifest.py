# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""The bundle manifest — the correctness core of tt-model.

Every shippable bundle carries its own venv (the "wall" between models) and needs only a TT
card + firmware on the box. Two schemas are supported:

- **v5 "fat"** (``bundled``): the platform artifacts the author built — *their* ttnn wheel
  (custom kernels compiled in), an empty-target vLLM wheel, the plugin wheel, and the modified
  metal tree — are embedded in the bundle and installed into a fresh venv.
- **v6 "thin"** (``deps``): the venv is built from pip dependency pins (ttnn / tt-metal-models)
  plus bundled wheels (the vLLM plugin + generic_op), with an empty-target vLLM build step.

Both render the same plugin-owned ``vllm_metadata.json`` (``EXTRA_MODELS_DIR`` contract) from
the shared ``entrypoint`` / ``mesh`` / ``resources`` / ``capabilities`` / ``weights`` blocks, and
both gate compatibility only on ``arch`` (fatal) + ``device_count`` (forceable): the engine that
runs is the venv's, not the host's, so no host tt-metal/version check applies.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

# Current authored schema. ``stage_package`` writes "5" and ``stage_thin_package`` writes "6";
# this is only the default for a bare ``Manifest(...)``.
SCHEMA_VERSION = "5"

# Every schema version this tt-model can read. v5 is the self-contained ("fat") schema (embedded
# platform wheels); v6 is the "thin" schema (pip-dep venv). Legacy v3 (kernel-cache/dispatch) and
# v4 (host-provisioned vLLM) are no longer supported — a bundle on any other version is refused
# outright; re-publish it with a current tt-model.
#
# v5.1 is the CONTAINER schema: the platform ships as an OCI image rather than as a venv (see
# ``container``), so the consumer needs only Docker + a TT card. It is numbered as a POINT
# release of v5 because it is the same promise — a package needing no host tt-metal — by a
# stronger mechanism, which is what left the whole number free for v6 "thin".
CONTAINER_SCHEMA = "5.1"
SUPPORTED_SCHEMAS = frozenset({"5", "5.1", "6"})


class Producer(BaseModel):
    tt_kernel_version: str
    created_at: str
    hostname: Optional[str] = None


class WeightsRef(BaseModel):
    """Where to fetch model weights from the Hub.

    The single, actionable record of which model a bundle targets: a normal HF model
    repo, downloaded separately from the bundle (skippable with ``--no-weights``).
    """

    # Aliased ``repo`` in JSON so a manifest can write the natural ``"weights": {"repo": ...}``
    # while the field stays ``repo_id`` everywhere in code.
    repo_id: str = Field(alias="repo")
    revision: Optional[str] = None
    allow_patterns: Optional[List[str]] = None
    ignore_patterns: Optional[List[str]] = None
    repo_type: str = "model"

    model_config = {"populate_by_name": True}


class WheelArtifact(BaseModel):
    """One platform wheel shipped inside a self-contained (v5) bundle.

    ``path`` is relative to the bundle repo root (e.g. ``wheels/ttnn-0.75.0-cp312-cp312-linux_x86_64.whl``)
    and is git-LFS tracked. The wheel-compatibility tags are parsed from the filename so ``pull``
    can refuse an install on an interpreter/platform the wheel was not built for — the wheels are
    the author's build (cp312/linux_x86_64), not universal.
    """

    path: str
    sha256: str
    size: int = 0
    python_tag: Optional[str] = None  # e.g. "cp312"
    abi_tag: Optional[str] = None  # e.g. "cp312"
    platform_tag: Optional[str] = None  # e.g. "linux_x86_64"


class BundledPlatform(BaseModel):
    """The self-contained ("fat") platform shipped INSIDE a v5 bundle.

    These are the actual artifacts the author built on their box — *their* ttnn wheel (custom
    C++/LLK kernels already compiled in), the empty-target base vLLM wheel, and the vLLM plugin
    wheel — embedded in the bundle via git-LFS and installed into a fresh venv by ``install_script``.
    This is what makes a package "package what's on your box": a consumer needs only a TT card +
    firmware, not a pre-provisioned tt-metal/vLLM stack. ``metal_dir`` is the author's modified
    tt-metal-community tree (the ttnn *Python* building blocks + model code), embedded alongside.
    """

    ttnn_wheel: Optional[WheelArtifact] = None
    vllm_wheel: Optional[WheelArtifact] = None
    plugin_wheel: Optional[WheelArtifact] = None
    extra_wheels: List[WheelArtifact] = Field(default_factory=list)
    metal_dir: Optional[str] = None  # embedded metal-community tree path, e.g. "metal"
    python: Optional[str] = None  # pinned interpreter (major.minor, e.g. "3.12") uv provisions
    deps_vendored: bool = False  # True => the full dependency closure is vendored in wheels/ (offline install)
    requirements: Optional[str] = None  # requirements.txt path within the bundle
    install_script: Optional[str] = None  # e.g. "install.sh"
    run_script: Optional[str] = None  # e.g. "run.sh"
    firmware_min: Optional[str] = None  # minimum card firmware/driver version, informational

    @property
    def wheels(self) -> List[WheelArtifact]:
        """All shipped wheels in install order (platform runtime first, then plugin, then extras)."""
        ordered = [self.ttnn_wheel, self.vllm_wheel, self.plugin_wheel, *self.extra_wheels]
        return [w for w in ordered if w is not None]


class Vllm(BaseModel):
    """How a v6 thin bundle installs vLLM core for the ``vllm-tt-plugin``.

    vLLM is **not** a plain pip pin. It must be **stock upstream vLLM built with
    ``VLLM_TARGET_DEVICE=empty``** — NOT the CUDA ``vllm`` wheel on PyPI — because the ``tt``
    platform is supplied by the ``vllm-tt-plugin`` at runtime (out-of-tree platform plugin). This
    mirrors the plugin's own ``docs/install-vllm-tt.sh``: install vLLM's *common* deps under a TT
    **override set** first (so ttnn's ``numpy<2`` is not bumped to numpy 2 by opencv), then install
    vLLM itself with ``--no-deps`` — either built from source (``--no-binary vllm``) or from a
    prebuilt empty-target wheel. See tenstorrent/vllm-tt-plugin.

    The plugin wheel itself and any ``generic_op`` wheels ride in ``Deps.wheels`` (installed by path
    AFTER vLLM). ``VLLM_TARGET_DEVICE`` is a *build-time* variable only — it is never set at serve.
    """

    version: str = "0.25.1"          # upstream vllm-project/vllm tag (empty-target build)
    target_device: str = "empty"     # VLLM_TARGET_DEVICE at build; the plugin provides `tt` at runtime
    overrides: Optional[str] = None  # bundle-relative override file (opencv/numpy pins) applied to common.txt
    # bundle-relative pinned copy of vLLM's requirements/common.txt; None => fetch it from the pinned tag
    common_requirements: Optional[str] = None
    # bundle-relative PREBUILT empty-target vLLM wheel (stock vLLM built empty, NOT the fork). When set,
    # install it with --no-deps instead of building from source — a hermetic, faster install.
    wheel: Optional[str] = None


class Deps(BaseModel):
    """v6 "thin" bundle: the per-model venv is built from pip dependency pins + bundled wheels,
    not from embedded platform wheels (see issue #29).

    ``requirements`` (a file shipped in the bundle) lists the pins — ``ttnn`` (team-provided /
    PyPI), the ``tt-metal-models`` wheel, etc. ``wheels_dir`` is a bundle folder of shipped wheels
    (the ``vllm-tt-plugin`` and the model's ``generic_op`` custom-op wheel) added to the install via
    ``--find-links`` and installed BY PATH. ``vllm`` describes the separate empty-target vLLM install
    step (see ``Vllm``) — vLLM is NOT in ``requirements`` or ``wheels`` because it needs its own
    ordered build. ``model_dir`` is where ``model.py`` lives (added to PYTHONPATH at serve). SFPI and
    firmware are external, box-managed deps — never in here.
    """

    python: Optional[str] = None            # pinned interpreter (major.minor), uv provisions
    requirements: str = "requirements.txt"  # pip pins from an index: ttnn, tt-metal-models, ...
    # Bundle-relative wheels installed BY PATH (things not on a pinnable index): the ``vllm-tt-plugin``
    # and any ``generic_op`` custom-op wheel. Installed AFTER vLLM (see ``vllm``).
    wheels: List[str] = Field(default_factory=list)
    wheels_dir: Optional[str] = None         # bundle dir holding those wheels -> also on --find-links
    # vLLM core install (empty-target, for the plugin). None => bundle serves no vLLM (non-vLLM model).
    vllm: Optional["Vllm"] = None
    model_dir: str = "."                     # where model.py lives (bundle root), added to PYTHONPATH


class Mesh(BaseModel):
    """Device topology the model was authored for.

    Structured (rather than buried in an opaque launch command's env) so the launch renderer
    can compose ``MESH_DEVICE``/fabric env and so search can reason about topology.
    """

    devices: int = 1
    topology: Optional[str] = None  # e.g. "1x4"
    fabric: Optional[str] = None  # e.g. "FABRIC_1D_RING"


class Entrypoint(BaseModel):
    """How the vLLM plugin loads the model.

    Maps directly onto the plugin-owned ``vllm_metadata.json``: ``cls`` -> ``main_class``
    (``"module:Class"``) and ``arch_name`` -> ``arch`` (the HF ``architectures`` name the
    plugin registers under its ``TT`` prefix). Aliased as ``class`` in JSON (a Python keyword).
    """

    cls: str = Field(alias="class")
    arch_name: str

    model_config = {"populate_by_name": True}


class Resources(BaseModel):
    """Structured launch knobs the renderer turns into vLLM args.

    Declarative-with-escape-hatch: the common knobs are structured so tt-model can validate
    and search on them, while ``command_override`` / ``extra_args`` let an author bypass or
    extend composition without waiting for a new field per vLLM flag.
    """

    max_model_len: Optional[int] = None
    max_num_seqs: Optional[int] = None
    block_size: Optional[int] = None
    trace_region_bytes: Optional[int] = None
    # Escape hatches (see docstring): raw args appended after the composed ones, or a full
    # argv that replaces composition entirely (per machine key, or "default").
    extra_args: List[str] = Field(default_factory=list)
    command_override: Dict[str, List[str]] = Field(default_factory=dict)


class Capabilities(BaseModel):
    """Serving capabilities the model exposes (rendered into vLLM args + repo tags)."""

    tool_parser: Optional[str] = None
    reasoning_parser: Optional[str] = None


class ContainerCapabilities(Capabilities):
    """``Capabilities`` with typos made fatal, for the container path only.

    The base model stays permissive because v3/v4/v5 bundles are already published and an
    unknown key must not retroactively make one unreadable. A v5.1 manifest is authored by
    hand in YAML, where a silently-ignored ``tool_praser:`` means the model ships without
    tool calling and nobody finds out until a client's tool call comes back as prose — so
    here the typo is refused at load.
    """

    model_config = ConfigDict(extra="forbid")


class ImageRef(BaseModel):
    """Where the container image lives — the registry is pluggable.

    ``registry="hf"`` (the default) means the image travels *inside the model repo* as an
    exploded OCI layout under ``image/``: one identity, one auth, one visibility gate for
    code + image + weights, and xet dedupes layers across models built on the same tt-metal
    commit. tt-model-manager can fetch the artifact and run it with docker.

    Any other value is an OCI registry namespace (e.g. ``ghcr.io/tenstorrent``), in which case
    the repo carries only a pointer and the image is fetched with ``docker pull``. That variant
    is what makes the image consumable by k8s/CI and by those not using tt-model-manager.
    """

    registry: str = "hf"  # "hf" => blobs under image/ in this repo; else an OCI namespace
    repository: Optional[str] = None  # for a real registry: the name under it, e.g. "laguna"
    tag: str  # the image tag, e.g. "tt-model/laguna:a1b2c3d4e"
    digest: Optional[str] = None  # sha256:… of the image manifest, when known

    @property
    def is_hub_hosted(self) -> bool:
        return self.registry == "hf"

    @property
    def pull_ref(self) -> Optional[str]:
        """The ``docker pull`` reference, or None when the image rides in the repo."""
        if self.is_hub_hosted:
            return None
        name = self.repository or self.tag.split("/")[-1].split(":")[0]
        ref = f"{self.registry.rstrip('/')}/{name}"
        return f"{ref}@{self.digest}" if self.digest else f"{ref}:{self.tag.split(':')[-1]}"


class ServeSettings(BaseModel):
    """Launch settings for a container package.

    ``serve`` holds the defaults every profile inherits; each entry of ``serve_profiles``
    deep-merges over them (dicts merge, everything else overrides wholesale).
    """

    hardware: Optional[str] = None  # device target label: p150, p150x2, p150x4 ...
    mesh_device: Optional[str] = None  # verbatim plugin string: "P150x4", "(1, 4)" ...
    port: Optional[int] = None
    max_model_len: Optional[int] = None
    max_num_seqs: Optional[int] = None  # REQUIRED after merge: the TT backend's default fails
    block_size: Optional[int] = None  # REQUIRED after merge: the TT backend's default fails
    server_timeout: Optional[int] = None

    # Tool-calling / reasoning parsers. Same block, same field names, same rendering rules
    # as the v4 path (see ``bundles._compose_launch_*``): ``tool_parser`` emits
    # ``--enable-auto-tool-choice --tool-call-parser X`` because vLLM hard-errors on the
    # latter without the former, and ``reasoning_parser`` keeps its underscore on purpose.
    # Mergeable like everything else: declare it once under ``serve:``, override per profile.
    capabilities: Optional[ContainerCapabilities] = None

    # Escape hatch for launcher settings that have no first-class field. Prefer a named
    # field when one exists — this dict is passed through opaquely and cannot be validated.
    additional_config: Dict[str, object] = Field(default_factory=dict)
    args: List[object] = Field(default_factory=list)  # str | [str, str] pairs
    env: Dict[str, str] = Field(default_factory=dict)

    def flat_args(self) -> List[str]:
        """Flatten ``[--flag, [--opt, value]]`` into a single argv fragment."""
        out: List[str] = []
        for a in self.args:
            if isinstance(a, list):
                out.extend(str(x) for x in a)
            else:
                out.append(str(a))
        return out


class ServeProfile(ServeSettings):
    """A named, launchable configuration.

    ONE image serves ALL of a model's profiles — kernels are JIT-compiled against whatever
    mesh is opened at launch, so a device target (p150x2 vs p150x4) and a deployment shape
    (latency vs capacity) are both just launch arguments, not separate builds.
    """

    name: str
    description: Optional[str] = None


class ContainerSpec(BaseModel):
    """The v5.1 block: the model's platform as an OCI image plus how to launch it.

    Present <=> this is a container package. Absent for every v3/v4/v5 bundle, which is
    what keeps those paths byte-for-byte unaffected.
    """

    image: ImageRef
    # Matches the authoring default. NOT plain "vllm": that is neither of the supported
    # kinds, so a wire manifest omitting the field would fail with "unsupported kind".
    kind: str = "vllm-plugin"
    runtime: Dict[str, object] = Field(default_factory=dict)  # shape is kind-specific
    serve: ServeSettings = Field(default_factory=ServeSettings)
    serve_profiles: List[ServeProfile] = Field(default_factory=list)
    default_profile: Optional[str] = None
    code_dir: Optional[str] = None  # browsable copy of what is inside the image, e.g. "code"
    verify: List[str] = Field(default_factory=list)  # build-time assertions run in the image
    built: Dict[str, object] = Field(default_factory=dict)  # pinned provenance from `package`

    def profile_names(self) -> List[str]:
        return [p.name for p in self.serve_profiles]

    def resolved_default(self) -> str:
        if self.default_profile:
            return self.default_profile
        if not self.serve_profiles:
            raise ValueError(
                "this container package declares no serve profiles — it was published by "
                "a tt-model that could not render them, or the manifest was hand-edited"
            )
        return self.serve_profiles[0].name

    def resolve_profile(self, name: Optional[str] = None) -> ServeProfile:
        """The fully merged profile: ``serve`` defaults with the named profile on top."""
        wanted = name or self.resolved_default()
        for p in self.serve_profiles:
            if p.name == wanted:
                merged = _deep_merge(
                    self.serve.model_dump(exclude_none=True),
                    p.model_dump(exclude_none=True),
                )
                return ServeProfile.model_validate(merged)
        raise ValueError(
            f"no serve profile named {wanted!r}; available: {', '.join(self.profile_names())}"
        )


def _deep_merge(base: Dict[str, object], over: Dict[str, object]) -> Dict[str, object]:
    """Merge ``over`` onto ``base``: dicts recurse, everything else overrides wholesale.

    Empty values in ``over`` do not erase a set default — a profile that omits ``args``
    inherits the ``serve:`` args rather than blanking them.
    """
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        elif v is not None and v != [] and v != {}:
            out[k] = v
    return out


class Manifest(BaseModel):
    """Root document of a bundle (``tt_kernel_manifest.json``).

    Always self-contained: exactly one of ``bundled`` (v5 fat) or ``deps`` (v6 thin) is set, and
    the bundle carries/builds its own venv. The ``entrypoint`` / ``mesh`` / ``resources`` /
    ``capabilities`` / ``weights`` blocks feed the plugin-owned ``vllm_metadata.json`` at serve.
    """

    schema_version: str = SCHEMA_VERSION
    name: str
    # The producer's tt-metal/ttnn version at package time, stamped for provenance. Informational
    # only — a self-contained bundle runs the engine from its own venv, so this is never gated.
    tt_metal_version: str
    arch: str  # blackhole | wormhole_b0 | ...
    device_count: int = 1
    producer: Producer
    weights: Optional[WeightsRef] = None

    # Shared serving blocks — rendered into the plugin-owned vllm_metadata.json at pull/serve.
    mesh: Optional[Mesh] = None
    entrypoint: Optional[Entrypoint] = None
    resources: Optional[Resources] = None
    capabilities: Optional[Capabilities] = None
    # Extra process env for serving, overlaid on the rendered launch env.
    env: Dict[str, str] = Field(default_factory=dict)

    # --- v5 self-contained ("fat") block --------------------------------------------------------
    # The platform artifacts shipped inside the bundle; pull installs these wheels into a fresh venv.
    bundled: Optional[BundledPlatform] = None

    # --- v6 "thin" block (issue #29) ------------------------------------------------------------
    # The per-model venv is built from pip dependency pins + bundled wheels (no embedded platform
    # wheels, no metal tree). SFPI is an external box dep.
    deps: Optional[Deps] = None

    @property
    def is_self_contained(self) -> bool:
        """True for a v5 bundle that ships its own platform wheels (needs no host tt-metal/vLLM)."""
        return self.bundled is not None and bool(self.bundled.wheels)

    # --- v5.1 container block (optional; absent => v5 fat or v6 thin) ---------------------------
    # The platform as an OCI image rather than a venv. When present, pull loads the image into
    # docker and serve runs it: the host needs Docker + a TT card and nothing else.
    container: Optional[ContainerSpec] = None

    @property
    def is_container(self) -> bool:
        """True for a v5.1 container package (the platform ships as an OCI image)."""
        return self.container is not None

    @property
    def is_thin(self) -> bool:
        """True for a v6 "thin" bundle: its venv is built from pip dep pins (no embedded wheels)."""
        return self.deps is not None

    @property
    def has_own_venv(self) -> bool:
        """True when the bundle carries/builds its own venv (v5 fat OR v6 thin) — the install &
        serve paths and the compat rules are the same for both: gate only on arch + device_count.

        A v5.1 container package is NOT one of these: it builds no venv on the host at all, so the
        v5/v6 install path must not claim it. ``compare()`` already treats every schema the same
        way (arch + device_count), which is exactly what a container needs too."""
        return self.is_self_contained or self.is_thin

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, text: str) -> "Manifest":
        """Parse and validate a manifest, rejecting any unsupported schema version.

        This tt-model reads schema v5 (self-contained/fat) and v6 (thin). Legacy v3/v4 bundles are
        refused outright rather than silently half-read — re-publish them with a current tt-model.
        """
        m = cls.model_validate_json(text)
        if m.schema_version not in SUPPORTED_SCHEMAS:
            supported = ", ".join(sorted(SUPPORTED_SCHEMAS))
            raise ValueError(
                f"Unsupported bundle schema_version {m.schema_version!r}; this tt-model "
                f"reads schema(s) {supported}. Re-publish the bundle with a current tt-model."
            )
        return m


class Incompatibility(BaseModel):
    """A single reason a bundle may not be usable on the local environment."""

    field: str
    expected: str  # from the manifest
    detected: str  # from the local environment
    fatal: bool  # True => never installable; False => guarded behind --force


class CompatibilityReport(BaseModel):
    """Verdict of comparing a manifest against the detected local environment."""

    compatible: bool  # no issues at all
    issues: List[Incompatibility] = Field(default_factory=list)

    @property
    def has_fatal(self) -> bool:
        return any(i.fatal for i in self.issues)

    @property
    def forceable(self) -> bool:
        """True when the only blockers are non-fatal (installable with --force)."""
        return bool(self.issues) and not self.has_fatal


def compare(manifest: Manifest, local: "LocalEnv") -> CompatibilityReport:  # noqa: F821
    """Compare a manifest against the detected local environment.

    ``local`` is a ``metal.LocalEnv`` (imported lazily to avoid a cycle). Every bundle is
    self-contained (v5 fat or v6 thin): the engine that runs is the venv's, not the host's, so
    none of the host's tt-metal facts are relevant. Only two things must match:

    - ``arch`` mismatch is **fatal** — the binaries/kernels target a specific ISA.
    - ``device_count`` mismatch is forceable — the model was authored for a given mesh.

    v5 wheels' interpreter/platform tags are checked separately at install
    (``host_incompatible_wheels``); v6 resolves its deps via pip at install.
    """
    issues: List[Incompatibility] = []

    if local.arch and manifest.arch != local.arch:
        issues.append(
            Incompatibility(field="arch", expected=manifest.arch, detected=local.arch, fatal=True)
        )

    if local.device_count and manifest.device_count != local.device_count:
        issues.append(
            Incompatibility(
                field="device_count",
                expected=str(manifest.device_count),
                detected=str(local.device_count),
                fatal=False,
            )
        )

    return CompatibilityReport(compatible=not issues, issues=issues)
