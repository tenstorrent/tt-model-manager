# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""The compatibility manifest — the correctness core of tt-model.

A manifest pins everything the cached binaries depend on so that ``pull`` can refuse
an install that would silently miss (or, worse, load wrong binaries). It records the
``build_key`` that names the cache subtree on disk *and* the inputs that determine it,
because a pure-Python consumer cannot compute its local ``build_key`` without opening a
device. See ``compute_build_key`` in tt-metal ``build_env_manager.cpp:164-184`` and the
per-kernel hash in ``program_descriptors.cpp:126-141``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

# Current authored schema. ``from_json`` also accepts prior versions so a bundle already
# published to the Hub keeps installing unchanged (see SUPPORTED_SCHEMAS).
SCHEMA_VERSION = "5"

# Every schema version this tt-model can read. v5 is the self-contained ("fat") schema: it adds
# a ``bundled`` block recording the platform wheels (ttnn/vllm/plugin) the author's box shipped
# INSIDE the bundle, so a consumer needs only a TT card + firmware. v4 is the unified "model +
# manifest" schema (structured target/mesh/ranges/resources — vLLM only, host-provisioned
# platform); v3 is the legacy kernel-cache/dispatch schema, read-only supported. A bundle on any
# other version is refused outright rather than silently half-read.
SUPPORTED_SCHEMAS = frozenset({"3", "4", "5"})


class FileEntry(BaseModel):
    """One file in the cache subtree, indexed for integrity verification.

    ``path`` is relative to the ``<build_key>/`` root of the subtree.
    """

    path: str
    sha256: str
    size: int


class BuildKeyInputs(BaseModel):
    """The inputs to tt-metal's ``compute_build_key`` (build_env_manager.cpp:164-184).

    ``harvesting_mask`` only participates in the build_key when coordinate
    virtualization is disabled, so it is excluded from comparison when
    ``coordinate_virtualization_enabled`` is true (mirrors the C++ logic).
    """

    dispatch_core_type: str = "WORKER"
    dispatch_core_axis: str = "ROW"
    num_hw_cqs: int = 1
    coordinate_virtualization_enabled: bool = True
    harvesting_mask: int = 0
    compile_hash_string: str = ""


class Producer(BaseModel):
    tt_kernel_version: str
    created_at: str
    hostname: Optional[str] = None
    # Absolute path of the producer's tt-metal source root, as embedded in the kernel
    # cache's .dephash dependency paths. Lets a consumer on a different host (different
    # checkout path / HOME) rewrite those tree-dep prefixes to its own tt-metal so the
    # pulled cache hits instead of recompiling. None => producer couldn't detect it
    # (consumer falls back to in-cache relocation only, correct on the same host).
    tt_metal_home: Optional[str] = None


class RunnerPayload(BaseModel):
    """The runner a bundle serves through. ``backend`` selects which serving layer and,
    with it, which runner contract the payload satisfies:

    - ``backend == "dispatch"`` (default, legacy): a Python runner following the legacy
      contract (``generate()``/``generate_stream()``/``benchmark()``), served by tt-model's
      own legacy-runner server (``tt_kernel.legacy_serve``). Two modes:

      - **packaged**: ``wheels`` is non-empty — the wheel(s) are stored under ``python/``
        in the bundle and indexed in ``Manifest.files`` (path prefix ``python/``) so the
        existing integrity check covers them. ``pull`` pip-installs them.
      - **reference**: ``wheels`` is empty — the runner is *not* shipped; the consumer is
        expected to already have it or to install it from ``source``.

      ``spec`` is the opaque ``"module:Class"`` string the legacy-runner server loads;
      tt-model records it but never imports it.

    - ``backend == "vllm"``: the model is served through the Tenstorrent vLLM plugin. The
      payload is a self-contained bundle *folder* (``bundle_dir``) holding a plugin-owned
      ``vllm_metadata.json`` (arch name, main-class path, per-machine launch command, HF
      weights ref) plus the ``VllmGeneratorAdapter`` class and its dependencies. tt-model
      lays the folder into ``EXTRA_MODELS_DIR`` at serve time; the plugin scans it and
      registers the model. ``vllm_metadata.json`` — not this payload — is the source of
      truth for the serving contract, so ``spec``/``wheels`` are unused for vLLM.
    """

    spec: str = ""  # "module:Class" for dispatch --runner (dispatch backend only)
    wheels: List[str] = Field(default_factory=list)  # filenames under python/; empty => reference
    entry_point: Optional[str] = None  # name registered under tt_models.runners
    source: Optional[str] = None  # where to get a reference (not-shipped) runner: pip name / git URL
    requires_python: Optional[str] = None  # informational
    backend: str = "dispatch"  # "dispatch" | "vllm"
    # For backend=="vllm": path (within the bundle repo) of the folder holding
    # vllm_metadata.json + the adapter class + its deps. None for dispatch backend.
    bundle_dir: Optional[str] = None

    @property
    def is_packaged(self) -> bool:
        """True when the bundle ships the runner wheel(s); False => reference-only.

        For the vLLM backend the folder itself (``bundle_dir``) is the shipped payload."""
        return bool(self.wheels) or (self.backend == "vllm" and self.bundle_dir is not None)

    @property
    def is_vllm(self) -> bool:
        return self.backend == "vllm"


class WeightsRef(BaseModel):
    """Where to fetch model weights from the Hub.

    The single, actionable record of which model a bundle targets: a normal HF model
    repo, downloaded separately from the kernel bundle (skippable with ``--no-weights``).
    """

    # Aliased ``repo`` in JSON so a v4 manifest can write the natural ``"weights": {"repo": ...}``
    # while the field stays ``repo_id`` everywhere in code.
    repo_id: str = Field(alias="repo")
    revision: Optional[str] = None
    allow_patterns: Optional[List[str]] = None
    ignore_patterns: Optional[List[str]] = None
    repo_type: str = "model"

    model_config = {"populate_by_name": True}


class Platform(BaseModel):
    """The platform (tt-metal/ttnn) envelope a v4 bundle runs on.

    ``ttnn`` is a PEP 440 version specifier (e.g. ``">=0.72,<0.76"``) — a *range*, not the
    exact pin the legacy kernel-cache path uses. When set, ``compare()`` gates on it (the
    installed ttnn falling outside the range is forceable, not fatal) instead of the v3
    exact-string ``tt_metal_version`` check. Kernels-less bundles JIT for the running
    platform, so a range is the honest contract: the model advertises the envelope it
    supports and the consumer resolves against what's installed.
    """

    ttnn: Optional[str] = None  # PEP 440 specifier, e.g. ">=0.72,<0.76"


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

    Unlike ``Platform.ttnn`` (a version *range* gated against a host-installed ttnn), these are the
    actual artifacts the author built on their box — *their* ttnn wheel (custom C++/LLK kernels
    already compiled in), the empty-target base vLLM wheel, and the vLLM plugin wheel — embedded in
    the bundle via git-LFS and installed into a fresh venv by ``install_script``. This is what makes
    a package "package what's on your box": a consumer needs only a TT card + firmware, not a
    pre-provisioned tt-metal/vLLM stack. ``metal_dir`` is the author's modified tt-metal-community
    tree (the ttnn *Python* building blocks + model code), embedded alongside the wheel.
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


class Runtime(BaseModel):
    """The serving runtime a v4 bundle needs.

    ``kind`` selects the serving layer (only ``"vllm"`` today). ``version`` is a PEP 440
    specifier for the runtime core (vLLM); ``plugin_version`` is a separate specifier for the
    Tenstorrent vLLM *plugin* (``vllm_tt_plugin``) — a distinct package from vLLM core that
    the fork ships alongside it. Either range being unsatisfied by what's installed is
    forceable (non-fatal). Omitting ``plugin_version`` keeps the legacy presence-only plugin
    check (the fork tracks ``dev`` with no version floor).
    """

    kind: str = "vllm"
    version: Optional[str] = None  # PEP 440 specifier for vLLM core, e.g. ">=0.24"
    plugin_version: Optional[str] = None  # PEP 440 specifier for vllm_tt_plugin, e.g. ">=0.3,<0.4"


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


class Manifest(BaseModel):
    """Root document of a bundle (``tt_kernel_manifest.json``)."""

    schema_version: str = SCHEMA_VERSION
    name: str
    tt_metal_version: str  # MUST match local (per-kernel hash dependency)
    arch: str  # blackhole | wormhole_b0 | ...
    device_count: int = 1
    # uint64 naming the cache subtree on disk. ``None`` => a kernels-less bundle (no
    # precompiled cache shipped): the vLLM path JITs at first-run warmup into tt-metal's
    # own local cache, and a dispatch bundle without kernels falls back to dynamic JIT.
    build_key: Optional[int] = None
    build_key_inputs: BuildKeyInputs = Field(default_factory=BuildKeyInputs)
    kernel_count: int = 0
    # Whether the cache carries the traced-decode / on-device-lm_head kernels a fast-path
    # consumer (DISPATCH_TRACE/DISPATCH_ONDEVICE_LMHEAD) needs (#6). None => not recorded
    # (older bundle); False => baseline-only (fast-path serving will re-JIT those kernels).
    fast_path_kernels: Optional[bool] = None
    files: List[FileEntry] = Field(default_factory=list)
    producer: Producer
    # Runtime payload (both optional): a runner to dispatch to (packaged or reference)
    # and the model weights to fetch. Absent => a pure kernel (warm compile-cache) bundle.
    runner: Optional[RunnerPayload] = None
    weights: Optional[WeightsRef] = None

    # --- v4 unified-manifest blocks (all optional; absent => a v3 bundle) -------------------
    # These describe a kernels-less vLLM model in one authoritative document. tt-model renders
    # the plugin-owned ``vllm_metadata.json`` from them at pull/serve; ``compare()`` gates on
    # the ranges in ``platform``/``runtime``. A v3 bundle leaves them all None and behaves
    # exactly as before.
    platform: Optional[Platform] = None
    runtime: Optional[Runtime] = None
    target: Optional[str] = None  # searchable SKU name, e.g. "p150x4"
    mesh: Optional[Mesh] = None
    entrypoint: Optional[Entrypoint] = None
    resources: Optional[Resources] = None
    capabilities: Optional[Capabilities] = None
    # Extra process env for serving, overlaid on the rendered launch env.
    env: Dict[str, str] = Field(default_factory=dict)

    # --- v5 self-contained ("fat") block (optional; absent => v3/v4 host-provisioned) ----------
    # The platform artifacts shipped inside the bundle. When present, pull installs these wheels
    # into a fresh venv instead of gating on a host-installed tt-metal/vLLM: the package is
    # self-contained and needs only a TT card + firmware.
    bundled: Optional[BundledPlatform] = None

    @property
    def is_v4(self) -> bool:
        """True for a unified vLLM manifest (has an entrypoint / platform block)."""
        return self.entrypoint is not None or self.platform is not None

    @property
    def is_self_contained(self) -> bool:
        """True for a v5 bundle that ships its own platform wheels (needs no host tt-metal/vLLM)."""
        return self.bundled is not None and bool(self.bundled.wheels)

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, text: str) -> "Manifest":
        """Parse and validate a manifest, rejecting any unsupported schema version.

        This tt-model reads every schema in ``SUPPORTED_SCHEMAS`` (currently v3 legacy
        kernel-cache and v4 unified vLLM). A bundle on any other version is refused outright
        rather than silently half-read — re-publish it with a matching tt-model.
        """
        m = cls.model_validate_json(text)
        if m.schema_version not in SUPPORTED_SCHEMAS:
            supported = ", ".join(sorted(SUPPORTED_SCHEMAS))
            raise ValueError(
                f"Unsupported bundle schema_version {m.schema_version!r}; this tt-model "
                f"reads schema(s) {supported}. Re-publish the bundle with a current tt-model."
            )
        return m

    @property
    def total_size(self) -> int:
        return sum(f.size for f in self.files)


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


def _range_issues(manifest: Manifest, local: "LocalEnv") -> List[Incompatibility]:  # noqa: F821
    """Forceable version-range issues for a v4 manifest's ``platform``/``runtime``.

    Only emits when a range is declared AND the installed version is a real, parseable
    version that falls outside it. An unresolvable installed version (None / bare git sha)
    is treated as "assume OK" by ``version_satisfies`` (returns None) and produces no issue —
    a dev checkout is never falsely blocked. Both issues are non-fatal (``--force``-able).
    """
    from .toolchain import version_satisfies

    out: List[Incompatibility] = []
    if manifest.platform and manifest.platform.ttnn:
        # tt_metal_version doubles as the installed ttnn version (same source).
        if version_satisfies(local.tt_metal_version, manifest.platform.ttnn) is False:
            out.append(
                Incompatibility(
                    field="platform.ttnn",
                    expected=manifest.platform.ttnn,
                    detected=local.tt_metal_version or "unknown",
                    fatal=False,
                )
            )
    if manifest.runtime and manifest.runtime.version:
        if version_satisfies(local.vllm_version, manifest.runtime.version) is False:
            out.append(
                Incompatibility(
                    field=f"runtime.{manifest.runtime.kind}",
                    expected=manifest.runtime.version,
                    detected=local.vllm_version or "unknown",
                    fatal=False,
                )
            )
    if manifest.runtime and manifest.runtime.plugin_version:
        if version_satisfies(local.vllm_plugin_version, manifest.runtime.plugin_version) is False:
            out.append(
                Incompatibility(
                    field=f"runtime.{manifest.runtime.kind}-plugin",
                    expected=manifest.runtime.plugin_version,
                    detected=local.vllm_plugin_version or "unknown",
                    fatal=False,
                )
            )
    return out


def compare(manifest: Manifest, local: "LocalEnv") -> CompatibilityReport:  # noqa: F821
    """Compare a manifest against the detected local environment.

    ``local`` is a ``metal.LocalEnv`` (imported lazily to avoid a cycle). Comparison
    rules, from the verified tt-metal source:

    - ``arch`` mismatch is **fatal** — the binaries are for a different ISA.
    - ``tt_metal_version`` mismatch is a hard block (non-fatal: forceable) — per-kernel
      hashes won't match, so the cache would silently miss.
    - build_key inputs that differ change the build_key integer, so the consumer's
      tt-metal would look under a different directory => silent miss (forceable).
    - ``device_count`` mismatch is a warning (forceable).

    A **self-contained** (v5) bundle ships its own ttnn/vLLM engine wheels, so the host's
    tt-metal is irrelevant entirely: it is gated ONLY on ``arch`` (fatal) and ``device_count``
    (forceable) — no tt_metal_version, build_key, harvesting, or version-range check applies,
    and the host need not have tt-metal installed at all.

    A **kernels-less** bundle (``build_key is None``) ships no precompiled cache, so none
    of the cache-dependent gates apply: only ``arch`` (still fatal — the adapter/kernels
    JIT for a specific ISA) and ``device_count`` are checked. For a **v4** bundle the
    ``platform.ttnn`` / ``runtime.version`` *ranges* are also checked here: an installed
    version outside the declared range is forceable (non-fatal), mirroring the legacy
    ``tt_metal_version`` block — the model advertises an envelope, the consumer resolves
    against it, and ``--force`` overrides. A v3 kernels-less bundle carries no ranges, so its
    version check stays with ``runner_version_advisory`` (a warning).
    """
    issues: List[Incompatibility] = []
    inp = manifest.build_key_inputs

    if local.arch and manifest.arch != local.arch:
        issues.append(
            Incompatibility(field="arch", expected=manifest.arch, detected=local.arch, fatal=True)
        )

    # A self-contained (v5) bundle ships its own ttnn/vLLM engine wheels and installs them into its
    # OWN venv, so NONE of the host's tt-metal facts are relevant: not the tt_metal_version (the
    # engine that runs is the bundle's, not the host's — the host need not have tt-metal at all),
    # not the kernel-cache build_key / harvesting (the bundle JITs into its own cache), not the
    # platform/runtime version ranges. Gate ONLY on arch (the ISA the shipped binaries target —
    # still fatal above) and device_count (the mesh the model needs). The shipped wheels' own
    # interpreter/platform tags are verified separately at install time (host_incompatible_wheels).
    if manifest.is_self_contained:
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

    if manifest.build_key is None:
        # Kernels-less (non-self-contained v3/v4): skip every cache-dependent gate. A v4 bundle
        # references a host-provisioned tt-metal, so its declared platform/runtime version ranges
        # DO gate here (an installed version outside the range is forceable). device_count too.
        issues.extend(_range_issues(manifest, local))
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

    if local.tt_metal_version and manifest.tt_metal_version != local.tt_metal_version:
        issues.append(
            Incompatibility(
                field="tt_metal_version",
                expected=manifest.tt_metal_version,
                detected=local.tt_metal_version,
                fatal=False,
            )
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

    # harvesting_mask only affects the build_key when virtualization is disabled.
    if not inp.coordinate_virtualization_enabled and local.harvesting_mask is not None:
        if inp.harvesting_mask != local.harvesting_mask:
            issues.append(
                Incompatibility(
                    field="harvesting_mask",
                    expected=str(inp.harvesting_mask),
                    detected=str(local.harvesting_mask),
                    fatal=False,
                )
            )

    # If --probe gave us a real local build_key, an integer mismatch is decisive.
    if local.build_key is not None and manifest.build_key != local.build_key:
        issues.append(
            Incompatibility(
                field="build_key",
                expected=str(manifest.build_key),
                detected=str(local.build_key),
                fatal=False,
            )
        )

    return CompatibilityReport(compatible=not issues, issues=issues)


def runner_version_advisory(manifest: Manifest, local: "LocalEnv") -> Optional[Incompatibility]:  # noqa: F821
    """Non-fatal version check for the runner wheel + weights.

    Unlike the kernel ``compare()`` gate (which hard-blocks a tt_metal_version
    mismatch because mismatched kernels are useless), the runner wheel and weights
    are reusable and not as hard-locked, so a mismatch installs anyway with a loud
    warning — the user resolves the version blocker themselves. Returns an
    informational ``Incompatibility`` (always ``fatal=False``) or None.
    """
    if manifest.runner is None and manifest.weights is None:
        return None
    # A self-contained bundle ships its own engine; the host tt-metal version is irrelevant to it
    # (and the host may not have tt-metal installed at all), so never advise on it.
    if manifest.is_self_contained:
        return None
    if local.tt_metal_version and manifest.tt_metal_version != local.tt_metal_version:
        return Incompatibility(
            field="runner_tt_metal_version",
            expected=manifest.tt_metal_version,
            detected=local.tt_metal_version,
            fatal=False,
        )
    return None
