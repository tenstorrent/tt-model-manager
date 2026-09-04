# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Assemble self-contained bundles — v5 "fat" ("package what's on your box") and v6 "thin".

A v5 bundle stages ONE running folder that carries the author's actual built artifacts: their
ttnn wheel (custom C++/LLK kernels compiled in), the empty-target base vLLM wheel, the vLLM
plugin wheel, and their modified tt-metal-community tree — plus a generated ``install.sh``/
``run.sh`` and a v5 manifest. A v6 bundle instead ships ``model.py`` + pip dependency pins and
builds the venv at install (see ``stage_thin_package``). A consumer needs only a TT card + firmware.

Weights are NEVER embedded: the manifest carries the HF repo id and ``pull`` downloads them.

This module does the filesystem staging only (no network); ``cli.package`` calls it then uploads
via ``hub`` (git-LFS handles the large wheels automatically). Kept import-light and hardware-free
so it is unit-testable offline.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import re
import shutil
import socket
from pathlib import Path
from typing import Dict, List, Optional

from .manifest import (
    BundledPlatform,
    Deps,
    Entrypoint,
    Manifest,
    Mesh,
    Producer,
    Resources,
    Vllm,
    WeightsRef,
    WheelArtifact,
)

# Where the shipped wheels and the embedded metal tree live inside the bundle.
WHEELS_DIR = "wheels"
METAL_DIR = "metal"
INSTALL_SCRIPT = "install.sh"
RUN_SCRIPT = "run.sh"
REQUIREMENTS = "requirements.txt"
# Override file for the empty-target vLLM install (pins that keep ttnn's numpy<2 from being bumped).
VLLM_OVERRIDES = "vllm-overrides.txt"
# Default upstream vLLM tag the vllm-tt-plugin builds against (empty target). Keep in step with the
# plugin's docs/install-vllm-tt.sh (tenstorrent/vllm-tt-plugin).
VLLM_VERSION = "0.25.1"
# Per-model vLLM bundle folders live under here; this dir (not the bundle root) is EXTRA_MODELS_DIR
# so the plugin's child-scan finds exactly the model metadata and not metal/, wheels/, venv/.
METADATA_DIR = "vllm_models"
# The plugin-owned metadata file at the root of each per-model folder. tt-model writes it from the
# author's entrypoint (arch + main_class); the vLLM plugin reads it at serve via EXTRA_MODELS_DIR.
VLLM_METADATA_NAME = "vllm_metadata.json"

# torch is the CPU build for Tenstorrent (never CUDA); requirements install uses this index.
_PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"


class StagingError(RuntimeError):
    """Staging the embedded ``metal/`` tree failed part-way through.

    ``shutil.copytree`` walks the whole source before raising, so a failure (EACCES on a
    root-owned build artifact, ENOSPC mid-copy, a socket/fifo) surfaces only at the end with a
    raw traceback. This carries the offending source paths so the CLI can render the repo's
    styled error instead. ``paths`` is a list of human-readable ``"<path>: <why>"`` strings.
    """

    def __init__(self, message: str, paths: Optional[List[str]] = None) -> None:
        super().__init__(message)
        self.paths = paths or []


# Basenames excluded at EVERY depth of the metal tree (VCS, byte-caches, venvs, logs, loose
# per-type build dirs). Mirrors the pre-existing ignore_patterns list unchanged.
_METAL_IGNORE_ANYWHERE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", ".git", "venv", ".venv", "model_cache",
    "generated", "*.log", ".pytest_cache", "dist", "build_*",
)
# Regenerable multi-GB caches (~3.7/~4/~2 GB) and build output, excluded ONLY at the tree ROOT —
# mirroring tt-metal's own .gitignore anchoring (e.g. ``/python_env/``). Anchoring at the root
# keeps a tracked NESTED dir of the same basename (e.g. tt_metal/python_env/requirements-dev.txt)
# from silently vanishing. ``build`` is the ``build -> build_Release`` symlink build_metal.sh
# creates; ``built``/``built_kernels`` are tt-metal's generated kernel caches (its .gitignore).
_METAL_IGNORE_ROOT_ONLY = frozenset(
    {".cpmcache", "python_env", "tt_cache", "build", "built", "built_kernels"}
)


def _normalize_staged_symlinks(root: Path) -> None:
    """Make the staged ``metal/`` tree self-contained after a ``symlinks=True`` copytree.

    Copying links as links (rather than following them) means the staged tree can hold links that
    dangle (their target was excluded, e.g. ``build -> build_Release``) or point OUTSIDE the tree
    (an absolute path that leaks the author's host, or a ``..`` escape). Neither belongs in a
    shipped artifact — a dereferencing copy (``cp -rL``, ``rsync -aL``, ``tar -czhf``) breaks on a
    dangling link, and ``hub.push_folder`` silently drops symlinked directories, so a ``--out``
    bundle and the same bundle after push/pull would diverge. Walk the tree and, per symlink:

    * dangling (target missing, or a symlink loop) -> drop it;
    * resolves INSIDE ``root`` -> keep it (a self-contained relative link);
    * resolves OUTSIDE ``root`` -> replace it with a real copy of the target (file or directory)
      so nothing leaks and the bundle carries the content the pre-change copytree used to embed.
    """
    root = root.resolve()

    def _normalize_link(link: Path) -> None:
        try:
            target = link.resolve(strict=True)  # follows the chain; strict flags a missing target
        except (OSError, RuntimeError):
            link.unlink()  # dangling target or a symlink loop -> no bundle should ship it
            return
        try:
            target.relative_to(root)
            return  # inside the staged tree already: a self-contained link, keep as-is
        except ValueError:
            pass  # points outside -> materialize below so the host path never ships
        link.unlink()
        if target.is_dir():
            shutil.copytree(target, link, symlinks=True)
            _walk(link)  # the fresh copy may itself carry links that escape the tree
        else:
            shutil.copy2(target, link)

    def _walk(directory: Path) -> None:
        for entry in list(directory.iterdir()):  # snapshot: we mutate as we go
            if entry.is_symlink():
                _normalize_link(entry)
            elif entry.is_dir():  # a real dir (symlinks handled above, never followed)
                _walk(entry)

    _walk(root)

# A wheel filename is: {distribution}-{version}(-{build})?-{python}-{abi}-{platform}.whl
# (PEP 427). We only need the trailing three compatibility tags.
_WHEEL_RE = re.compile(r"^(?P<dist>.+?)-(?P<ver>[^-]+)(-\d[^-]*)?-(?P<py>[^-]+)-(?P<abi>[^-]+)-(?P<plat>[^-]+)\.whl$")


def host_python_tag() -> str:
    """This interpreter's CPython wheel tag, e.g. ``cp312``."""
    import sys

    return f"cp{sys.version_info.major}{sys.version_info.minor}"


# manylinux platform tags → the minimum glibc (major, minor) they require. The perennial
# ``manylinux_<major>_<minor>`` form encodes it directly; the legacy aliases are fixed points.
_LEGACY_MANYLINUX_GLIBC = {
    "manylinux1": (2, 5),
    "manylinux2010": (2, 12),
    "manylinux2014": (2, 17),
}


def glibc_floor_of_tag(platform_tag: Optional[str]) -> Optional[tuple]:
    """The minimum glibc (major, minor) a manylinux platform tag requires, or None.

    ``manylinux_2_39_x86_64`` -> (2, 39); ``manylinux2014_x86_64`` -> (2, 17). A bare
    ``linux_x86_64`` (unrepaired) or ``any`` carries no declared floor -> None (we can't
    reason about it, so it's never flagged here).
    """
    if not platform_tag:
        return None
    for legacy, floor in _LEGACY_MANYLINUX_GLIBC.items():
        if platform_tag.startswith(legacy):
            return floor
    m = re.match(r"manylinux_(\d+)_(\d+)_", platform_tag)
    return (int(m.group(1)), int(m.group(2))) if m else None


def host_glibc() -> Optional[tuple]:
    """This host's glibc (major, minor), or None if it can't be determined (e.g. musl)."""
    import platform as _platform

    try:
        _name, ver = _platform.libc_ver()
        if ver:
            parts = ver.split(".")
            return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
    except Exception:  # noqa: BLE001
        pass
    return None


def host_incompatible_wheels(bundled: "BundledPlatform") -> List[str]:  # noqa: F821
    """Shipped wheels whose interpreter/platform tags don't match this host.

    The shipped wheels are the author's build (e.g. cp312/manylinux_2_39_x86_64), NOT universal —
    a consumer on a different Python minor, OS, or an older glibc can't install them. Universal
    wheels (``py3-none-any`` like the plugin) are skipped. Returns human-readable reasons; empty ==
    all installable here.

    The glibc check turns the otherwise-cryptic runtime failure (``version 'GLIBC_2.39' not
    found`` at dlopen) into a clear, actionable message at pull time: the author must repackage
    on an older Ubuntu (e.g. 22.04, glibc 2.35) to serve older hosts.
    """
    import sys

    problems: List[str] = []
    host_py = host_python_tag()
    host_is_linux = sys.platform.startswith("linux")
    hg = host_glibc()
    for w in bundled.wheels:
        if not w.python_tag or w.python_tag.startswith("py") or w.abi_tag in (None, "none"):
            continue  # universal / non-CPython-pinned wheel
        name = Path(w.path).name
        if w.python_tag != host_py:
            problems.append(f"{name}: built for {w.python_tag}, host is {host_py}")
        if w.platform_tag and w.platform_tag != "any" and "linux" in w.platform_tag and not host_is_linux:
            problems.append(f"{name}: built for {w.platform_tag}, host is {sys.platform}")
        floor = glibc_floor_of_tag(w.platform_tag)
        if floor and hg and hg < floor:
            problems.append(
                f"{name}: needs glibc >= {floor[0]}.{floor[1]}, host has {hg[0]}.{hg[1]} — "
                f"the bundle must be repackaged on an older Ubuntu (e.g. 22.04, glibc 2.35) "
                f"to run on this host"
            )
    return problems


def sha256_file(path: Path, _chunk: int = 1 << 20) -> str:
    """Streaming sha256 of a file (wheels are large — don't slurp)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_chunk), b""):
            h.update(block)
    return h.hexdigest()


def parse_wheel_tags(filename: str) -> Dict[str, Optional[str]]:
    """Extract (python_tag, abi_tag, platform_tag) from a wheel filename.

    Returns None tags for a non-conforming name rather than raising — the artifact still ships;
    it just carries no install-time compatibility gate.
    """
    m = _WHEEL_RE.match(Path(filename).name)
    if not m:
        return {"python_tag": None, "abi_tag": None, "platform_tag": None}
    return {"python_tag": m["py"], "abi_tag": m["abi"], "platform_tag": m["plat"]}


def make_wheel_artifact(src: Path, rel_path: str) -> WheelArtifact:
    """Build a WheelArtifact (path within bundle + sha + size + tags) for one wheel file."""
    tags = parse_wheel_tags(src.name)
    return WheelArtifact(
        path=rel_path,
        sha256=sha256_file(src),
        size=src.stat().st_size,
        python_tag=tags["python_tag"],
        abi_tag=tags["abi_tag"],
        platform_tag=tags["platform_tag"],
    )


def render_install_sh(manifest: Manifest) -> str:
    """A reproducible, isolated installer built on **uv**.

    The old installer used the host's ``python3`` and a fresh unpinned ``pip`` resolve, so the
    same bundle installed differently (or not at all) on another machine. This version is
    deterministic across machines:

    - **uv provisions the exact interpreter** (``uv venv --python <pinned>``) — the target host
      needs no matching Python; uv downloads it if absent.
    - **install is offline from the vendored wheels** when ``deps_vendored`` is set
      (``--no-index --find-links wheels/``): the full dependency closure ships in the bundle, so
      there is no PyPI/resolver drift and no network at install time. If deps aren't vendored, it
      falls back to installing the platform wheels + ``requirements.txt`` from the CPU index.
    - uv is bootstrapped into the bundle (``.uv/``) if not already on PATH — a single static binary.

    Idempotent and path-relative; takes an optional venv path as ``$1`` (default ``./venv``).
    """
    # --link-mode=copy: copy wheel contents INTO the venv instead of hardlinking them from uv's
    # global cache — the installed folder must not depend on anything outside its own wall.
    if manifest.deps is not None:
        # v6 "thin": build the venv from pip dependency pins (ttnn / tt-metal-models, listed in
        # requirements) + a separate empty-target vLLM build + bundled wheels (the vllm-tt-plugin and
        # the model's generic_op wheel) installed by path. No embedded platform wheels, no metal tree.
        # (SFPI is an external box dep, not installed here.) The order is load-bearing — see below.
        d = manifest.deps
        pyver = d.python or "3.12"
        pip = 'uv pip install --python "$VENV/bin/python" --link-mode=copy'
        steps: List[str] = []
        # (1) Engine + models FIRST: ttnn (bundles the tt-metal runtime) and, once published,
        # tt-metal-models. This establishes torch + numpy<2 in the venv before vLLM's deps resolve.
        # --find-links checks wheels_dir first, so a locally-built wheel there (e.g. a hand-built
        # tt-metal-models wheel staged ahead of its index publish) satisfies its requirements.txt
        # pin without a network resolve; anything not present there still falls through to the index.
        req_find_links = f'--find-links "$HERE/{d.wheels_dir}" ' if d.wheels_dir else ""
        steps.append(
            f'{pip} {req_find_links}--extra-index-url {_PYTORCH_CPU_INDEX} -r "$HERE/{d.requirements}"'
        )
        # (2) vLLM core for the plugin: STOCK upstream vLLM built with VLLM_TARGET_DEVICE=empty (NOT
        # the CUDA `vllm` on PyPI). Mirrors tenstorrent/vllm-tt-plugin docs/install-vllm-tt.sh: install
        # vLLM's common deps under the TT override set (so ttnn's numpy<2 is not bumped by opencv),
        # then vLLM itself with --no-deps. VLLM_TARGET_DEVICE is build-time only.
        if d.vllm is not None:
            v = d.vllm
            override = f'--override "$HERE/{v.overrides}" ' if v.overrides else ""
            if v.common_requirements:
                common_src = f'"$HERE/{v.common_requirements}"'
                fetch = ""
                cleanup = ""
            else:
                common_src = '"$VLLM_COMMON"'
                fetch = (
                    'VLLM_COMMON="$(mktemp)"\n'
                    f'curl -fsSL "https://raw.githubusercontent.com/vllm-project/vllm/'
                    f'v{v.version}/requirements/common.txt" -o "$VLLM_COMMON"\n'
                )
                cleanup = '\nrm -f "$VLLM_COMMON"'
            common_install = f'{fetch}{pip} {override}-r {common_src}{cleanup}'
            if v.wheel:
                # Prebuilt empty-target wheel (stock vLLM built empty, NOT the fork): install by path.
                core = f'{pip} --no-deps "$HERE/{v.wheel}"'
            else:
                # Build stock upstream vLLM from source with the empty target.
                core = (
                    f'VLLM_TARGET_DEVICE={v.target_device} {pip} '
                    f'--no-deps --no-binary vllm vllm=={v.version}'
                )
            steps.append(common_install)
            steps.append(core)
        # (3) The vllm-tt-plugin + any generic_op wheels, installed BY PATH, AFTER vLLM. The plugin's
        # pyproject omits vllm on purpose, so this never re-resolves the empty-target build.
        if d.wheels:
            bundled = " ".join(f'"$HERE/{w}"' for w in d.wheels)
            find_links = f'--find-links "$HERE/{d.wheels_dir}" ' if d.wheels_dir else ""
            steps.append(f'{pip} {find_links}{bundled}')
        install = "\n".join(steps)
        deps_note = "v6 thin: ttnn/tt-metal-models (index) + empty-target vLLM + plugin/ops wheels (by path)"
    else:
        b = manifest.bundled
        pyver = (b.python if b and b.python else "3.12")
        plat_wheels = " ".join(f'"$HERE/{w.path}"' for w in (b.wheels if b else []))
        vendored = bool(b and b.deps_vendored)
        if vendored:
            install = (
                f'uv pip install --python "$VENV/bin/python" --link-mode=copy --no-index '
                f'--find-links "$HERE/{WHEELS_DIR}" {plat_wheels} -r "$HERE/{REQUIREMENTS}"'
            )
            deps_note = "offline, from the vendored wheels (reproducible, no network)"
        else:
            install = (
                f'uv pip install --python "$VENV/bin/python" --link-mode=copy {plat_wheels} && \\\n'
                f'  uv pip install --python "$VENV/bin/python" --link-mode=copy '
                f'--extra-index-url {_PYTORCH_CPU_INDEX} -r "$HERE/{REQUIREMENTS}"'
            )
            deps_note = "from the CPU index (deps not vendored — pass --vendor-deps for offline)"
    return f"""#!/usr/bin/env bash
# Install this self-contained TT model package into an isolated, reproducible venv (via uv).
# Usage: ./{INSTALL_SCRIPT} [venv-path]   (default: ./venv)
# Deps: {deps_note}
#
# HERMETIC INSTALL: everything the model needs to SERVE ends up UNDER this folder — the pinned
# interpreter (in .python/), the venv (with package contents copied in), and at serve time the
# caches/weights (run.sh points HF_HOME/TT_CACHE_PATH/... here). After this runs, serving depends
# on nothing outside the folder except the TT device + system libc. Only THIS install step reaches
# the network (to fetch the interpreter and, unless --vendor-deps, the pip deps).
set -euo pipefail
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
VENV="${{1:-$HERE/venv}}"
PYVER="{pyver}"

# uv gives us a pinned interpreter + deterministic installs, independent of the host Python.
if ! command -v uv >/dev/null 2>&1; then
  export UV_INSTALL_DIR="$HERE/.uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
  export PATH="$HERE/.uv:$PATH"
fi

# Keep the pinned interpreter INSIDE the bundle (not in uv's global ~/.local store), so the venv's
# python resolves within the folder wall. python-build-standalone (what uv provisions) is
# relocatable, so a --relocatable venv built against it stays self-contained.
export UV_PYTHON_INSTALL_DIR="$HERE/.python"
uv python install "$PYVER"
uv venv --relocatable --python "$PYVER" "$VENV"
{install}
echo "installed into $VENV (python $PYVER, interpreter under $HERE/.python)"
"""


def render_run_sh(manifest: Manifest) -> str:
    """A standalone launcher that wires the engine env and serves the OpenAI endpoint.

    Sets the non-obvious env this stack needs (LD_PRELOAD of _ttnncpp.so; TT_METAL_HOME at the
    installed ttnn; EXTRA_MODELS_DIR at this folder so the plugin finds vllm_metadata.json;
    single-chip fabric-off defaults) plus any model-specific ``manifest.env``, then launches vLLM.
    Works with only tt-model absent — ``tt-model serve`` is the managed path, this is the raw one.
    """
    weights = manifest.weights.repo_id if manifest.weights else ""
    mesh_device = (manifest.mesh.topology if manifest.mesh and manifest.mesh.topology else "") or ""
    extra_env = "".join(
        f'export {k}="{v}"\n' for k, v in (manifest.env or {}).items()
    )
    # The tt_transformers adapter reads HF_MODEL from the env (not vLLM's --model), so export it.
    hf_export = f'export HF_MODEL="${{HF_MODEL:-{weights}}}"\n' if weights else ""
    # The TT vLLM backend REQUIRES a supported batch size and a concrete block_size (its default
    # of 256 / None both fail), so always emit them — from the manifest's resources, with the
    # known-good tt_transformers defaults when unset.
    res = manifest.resources
    max_num_seqs = (res.max_num_seqs if res and res.max_num_seqs else 32)
    block_size = (res.block_size if res and res.block_size else 64)
    serving = f"--max_num_seqs {max_num_seqs} --block_size {block_size}"
    if res and res.max_model_len:
        serving += f" --max_model_len {res.max_model_len}"
    # Tool/reasoning parsers, if the manifest declares them. Same vLLM flag spelling the
    # compose path uses (see bundles._compose_launch_command): vLLM's FlexibleArgumentParser
    # normalizes '_'->'-', so '--tool_parser' would become the nonexistent '--tool-parser';
    # the real flag is '--tool-call-parser', and vLLM hard-errors on it without
    # '--enable-auto-tool-choice'. '--reasoning_parser' normalizes to the valid '--reasoning-parser'.
    cap = manifest.capabilities
    if cap is not None:
        if cap.tool_parser:
            serving += f" --enable-auto-tool-choice --tool-call-parser {cap.tool_parser}"
        if cap.reasoning_parser:
            serving += f" --reasoning_parser {cap.reasoning_parser}"
    if res and res.extra_args:
        serving += " " + " ".join(str(a) for a in res.extra_args)
    # PYTHONPATH: a v5 fat bundle embeds the modified metal tree at metal/; a v6 thin bundle gets
    # tt_transformers/TTTv2 from the installed wheels and only needs its own model.py on the path
    # (bundle root, or deps.model_dir). This is the one serve-time difference between the regimes.
    if manifest.deps is not None:
        md = (manifest.deps.model_dir or ".").strip("/")
        pythonpath_entry = "$HERE" if md in ("", ".") else f"$HERE/{md}"
    else:
        pythonpath_entry = f"$HERE/{METAL_DIR}"
    return f"""#!/usr/bin/env bash
# Serve this model on TT hardware. Assumes ./{INSTALL_SCRIPT} has been run.
set -euo pipefail
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
VENV="${{VENV:-$HERE/venv}}"
PYBIN="$VENV/bin/python"

# Locate ttnn WITHOUT importing it — importing loads _ttnn.so, which is exactly what needs the
# LD_PRELOAD below (chicken-and-egg). find_spec resolves the path without executing the module.
TTNN_DIR="$("$PYBIN" -c 'import importlib.util,os;print(os.path.dirname(importlib.util.find_spec("ttnn").origin))')"
# _ttnncpp.so lives in ttnn.libs/ for an auditwheel-repaired (portable) wheel, or build/lib/ for a
# raw one; preload it to avoid the glibc "static TLS block" error on late dlopen.
# Prefer the auditwheel-vendored copy in *.libs/ (that's the one _ttnn.so actually loads via
# RPATH); fall back to build/lib for a raw (unrepaired) wheel.
LD_PRELOAD="$(ls "$TTNN_DIR"/../*.libs/_ttnncpp*.so 2>/dev/null | head -1)"
[ -n "$LD_PRELOAD" ] || LD_PRELOAD="$(ls "$TTNN_DIR"/build/lib/_ttnncpp*.so 2>/dev/null | head -1)"
export LD_PRELOAD="${{LD_PRELOAD:?could not locate _ttnncpp.so in the ttnn install}}"
export TT_METAL_HOME="$TTNN_DIR"
# EXTRA_MODELS_DIR is a PARENT of per-model bundle folders; the plugin scans its children for
# each vllm_metadata.json (so the metadata lives in {METADATA_DIR}/<model>/, not the bundle root).
export EXTRA_MODELS_DIR="$HERE/{METADATA_DIR}"
export TT_VLLM_BUILTIN_MODELS=0
# Do NOT set VLLM_PLUGINS: it is an ALLOW-LIST — setting it suppresses the vllm.general_plugins
# group, so the model's tool/reasoning-parser overrides would silently not load. The TT platform
# + model registry load via entry points without it.
export PYTHONPATH="{pythonpath_entry}:${{PYTHONPATH:-}}"   # resolves the adapter/model imports
export MESH_DEVICE="${{MESH_DEVICE:-{mesh_device}}}"
export TT_METAL_VISIBLE_DEVICES="${{TT_METAL_VISIBLE_DEVICES:-0}}"
# HERMETIC RUNTIME: keep every cache/home INSIDE the folder wall, so serving writes and reads
# nothing outside it (the ttnn tensor cache even DEFAULTS to a hard-coded /mnt/... path upstream —
# a classic other-machine leak we must override). Each is overridable if the operator sets it.
export HF_HOME="${{HF_HOME:-$HERE/.hf}}"                  # HF weights + hub cache
export TT_CACHE_PATH="${{TT_CACHE_PATH:-$HERE/.tt_cache}}"    # ttnn weight/tensor cache
export TT_CACHE_HOME="${{TT_CACHE_HOME:-$HERE/.tt_cache}}"    # override upstream's /mnt/... default
export XDG_CACHE_HOME="${{XDG_CACHE_HOME:-$HERE/.cache}}"     # generic catch-all (triton, etc.)
export TRITON_CACHE_DIR="${{TRITON_CACHE_DIR:-$HERE/.cache/triton}}"
export TORCHINDUCTOR_CACHE_DIR="${{TORCHINDUCTOR_CACHE_DIR:-$HERE/.cache/inductor}}"
{hf_export}{extra_env}CMD=("$PYBIN" -m vllm.entrypoints.openai.api_server --model "{weights}" {serving} "$@")
# TT_MODEL_PRINT=1 (set by `tt-model serve --print`) echoes the fully-resolved command+env
if [ "${{TT_MODEL_PRINT:-0}}" = "1" ]; then
  printf 'LD_PRELOAD=%s TT_METAL_HOME=%s EXTRA_MODELS_DIR=%s MESH_DEVICE=%s HF_MODEL=%s\n  %s\n' \\
    "$LD_PRELOAD" "$TT_METAL_HOME" "$EXTRA_MODELS_DIR" "$MESH_DEVICE" "${{HF_MODEL:-}}" "${{CMD[*]}}"
  exit 0
fi
exec "${{CMD[@]}}"
"""


def stage_package(
    staged: Path,
    *,
    name: str,
    arch: str,
    ttnn_wheel: Path,
    metal_dir: Path,
    vllm_metadata: dict,
    tt_kernel_version: str,
    vllm_wheel: Optional[Path] = None,
    plugin_wheel: Optional[Path] = None,
    extra_wheels: Optional[List[Path]] = None,
    weights: Optional[WeightsRef] = None,
    device_count: int = 1,
    mesh: Optional[Mesh] = None,
    env: Optional[Dict[str, str]] = None,
    resources: Optional[Resources] = None,
    tt_metal_version: str = "unknown",
    firmware_min: Optional[str] = None,
    python_version: Optional[str] = None,
    deps_vendored: bool = False,
) -> Manifest:
    """Materialize the running-folder layout under ``staged`` and return the v5 manifest.

    Copies the author's wheels into ``wheels/``, the modified metal tree into ``metal/``, writes
    ``vllm_metadata.json`` (the EXTRA_MODELS_DIR contract), the generated ``install.sh``/``run.sh``,
    a ``requirements.txt`` (from the metal tree if present), and ``tt_kernel_manifest.json``.
    No network.
    """
    staged.mkdir(parents=True, exist_ok=True)
    wheels_root = staged / WHEELS_DIR
    wheels_root.mkdir(exist_ok=True)

    def _copy_wheel(src: Path) -> WheelArtifact:
        dest = wheels_root / src.name
        shutil.copy2(src, dest)
        return make_wheel_artifact(dest, f"{WHEELS_DIR}/{src.name}")

    ttnn_art = _copy_wheel(ttnn_wheel)
    if python_version is None and ttnn_art.python_tag and ttnn_art.python_tag.startswith("cp"):
        digits = ttnn_art.python_tag[2:]
        python_version = f"{digits[0]}.{digits[1:]}" if len(digits) >= 2 else None
    vllm_art = _copy_wheel(vllm_wheel) if vllm_wheel else None
    plugin_art = _copy_wheel(plugin_wheel) if plugin_wheel else None
    extra_arts = [_copy_wheel(w) for w in (extra_wheels or [])]

    # Embed the author's modified metal-community tree (skip caches/venvs/artifacts).
    # symlinks=True: copy links as links instead of following them. A built tt-metal
    # checkout normally has dangling symlinks; following them (the default) makes
    # copytree raise. The root-anchored excludes (.cpmcache/python_env/tt_cache/build/...)
    # are regenerable multi-GB caches + build output — embedding them defeats the point
    # of shipping wheels. _normalize_staged_symlinks then makes the tree self-contained.
    metal_root = metal_dir.resolve()

    def _ignore(src, names):
        ignored = set(_METAL_IGNORE_ANYWHERE(src, names))
        if Path(src).resolve() == metal_root:  # root-anchored only, mirroring tt-metal's .gitignore
            ignored |= _METAL_IGNORE_ROOT_ONLY.intersection(names)
        return ignored

    try:
        shutil.copytree(metal_dir, staged / METAL_DIR, symlinks=True, ignore=_ignore)
        # Make the staged tree self-contained: drop dangling links, materialize any that escape it.
        # Inside the try (not after it) so its own unlink/copy2/copytree failures — EACCES/ENOSPC,
        # or a shutil.Error from the recursive copy — surface as a StagingError with context too,
        # rather than the raw traceback this function exists to prevent.
        _normalize_staged_symlinks(staged / METAL_DIR)
    except shutil.Error as exc:  # per-entry failures accumulated across the whole walk
        details = []
        for item in (exc.args[0] if exc.args else []):
            try:
                src_p, _dst, why = item
                details.append(f"{src_p}: {why}")
            except (TypeError, ValueError):
                details.append(str(item))
        raise StagingError(
            f"Failed to stage the embedded metal tree from {metal_dir}.", details
        ) from exc
    except OSError as exc:  # EACCES/ENOSPC, and SpecialFileError (socket/fifo) — both OSError
        where = getattr(exc, "filename", None) or metal_dir
        raise StagingError(
            f"Failed to stage the embedded metal tree from {metal_dir}: {exc}", [str(where)]
        ) from exc

    # requirements.txt: prefer the metal tree's, else a minimal note.
    req_src = metal_dir / "requirements.txt"
    if req_src.is_file():
        shutil.copy2(req_src, staged / REQUIREMENTS)
    else:
        (staged / REQUIREMENTS).write_text("# add pip deps here (torch is installed CPU-only)\n")

    # The plugin's EXTRA_MODELS_DIR contract: it scans the *children* of EXTRA_MODELS_DIR for a
    # vllm_metadata.json in each. So the metadata goes in a per-model SUBFOLDER under METADATA_DIR
    # (which run.sh points EXTRA_MODELS_DIR at) — NOT the bundle root, where it would be missed.
    safe_key = name.replace("/", "__")
    model_bundle = staged / METADATA_DIR / safe_key
    model_bundle.mkdir(parents=True, exist_ok=True)
    (model_bundle / VLLM_METADATA_NAME).write_text(json.dumps(vllm_metadata, indent=2))

    bundled = BundledPlatform(
        ttnn_wheel=ttnn_art,
        vllm_wheel=vllm_art,
        plugin_wheel=plugin_art,
        extra_wheels=extra_arts,
        metal_dir=METAL_DIR,
        python=python_version,
        deps_vendored=deps_vendored,
        requirements=REQUIREMENTS,
        install_script=INSTALL_SCRIPT,
        run_script=RUN_SCRIPT,
        firmware_min=firmware_min,
    )
    entrypoint = Entrypoint(
        **{"class": vllm_metadata["main_class"], "arch_name": vllm_metadata["arch"]}
    )
    manifest = Manifest(
        schema_version="5",
        name=name,
        tt_metal_version=tt_metal_version,
        arch=arch,
        device_count=device_count,
        producer=Producer(
            tt_kernel_version=tt_kernel_version,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            hostname=socket.gethostname(),
        ),
        weights=weights,
        entrypoint=entrypoint,
        mesh=mesh,
        env=env or {},
        resources=resources,
        bundled=bundled,
    )

    # Generated scripts (run.sh reads the manifest's weights/mesh/env).
    (staged / INSTALL_SCRIPT).write_text(render_install_sh(manifest))
    (staged / RUN_SCRIPT).write_text(render_run_sh(manifest))
    for s in (INSTALL_SCRIPT, RUN_SCRIPT):
        # Owner read/write only (0o600), NO execute bit: these scripts are always invoked as
        # `bash <script>` (runtime.install_self_contained / _serve_self_contained / the docs), so
        # they never need to be executable — least privilege (Cycode SAST: permissive file assignment).
        (staged / s).chmod(0o600)

    (staged / "tt_kernel_manifest.json").write_text(manifest.to_json())
    return manifest


CUSTOM_OPS_DIR = "custom_ops"

# Placeholder requirements for a v6 thin bundle when the author doesn't supply one. Reflects the
# issue #29 plan exactly; the not-yet-published deps are commented TODOs the lab uncomments/pins
# once TTTv2 and the models wheel land (M0).
_THIN_REQUIREMENTS_TEMPLATE = """\
# v6 thin bundle — per-model venv dependency pins (see issue #29).
# SFPI + firmware are EXTERNAL box deps (installer-managed) and are NOT listed here.
#
# The models tree (incl. tt_transformers) is packaged as `tt-metal-models`, which pins ttnn
# exactly (tt-metal-models==X => ttnn==X). In progress upstream: tenstorrent/tt-metal#54478
# (pip/apt/dnf). Once published, this ONE pin pulls the matching ttnn transitively:
# tt-metal-models==<X>     # TODO: pin once published (#29 M0 / tt-metal#54478)
#
ttnn>=0.77                 # engine (PyPI today; bundles the tt-metal runtime). Until tt-metal-models
                           # lands you pin ttnn directly; after, tt-metal-models pulls the exact ttnn.
#
# vLLM is NOT pinned here. It is stock upstream vLLM built with VLLM_TARGET_DEVICE=empty and is
# installed by install.sh in its own ordered step (mirroring tenstorrent/vllm-tt-plugin's
# docs/install-vllm-tt.sh) — NOT the CUDA `vllm` on PyPI. The `vllm-tt-plugin` (the integration)
# ships as a bundled wheel (--plugin-wheel). Don't add `vllm` to this file: a resolvable pin would
# silently pull the CUDA wheel and clobber the empty-target build.
#
# <your-model>-ops==<Z>    # optional: your generic_op custom-op wheel (ship it via --ops-wheel)
"""

# The empty-target vLLM install applies these overrides to vLLM's requirements/common.txt so the
# tt-metal env's numpy<2 is not upgraded (opencv is vLLM's only numpy-2 puller and no TT-registered
# model uses its video path). Kept verbatim in step with tenstorrent/vllm-tt-plugin docs/vllm-overrides.txt.
_VLLM_OVERRIDES_TEMPLATE = """\
# vLLM dependency overrides for the empty-target (TT) build — see tenstorrent/vllm-tt-plugin.
# ttnn needs numpy<2; vLLM's common.txt wants opencv-python-headless>=4.13 which needs numpy>=2.
# Pin opencv to the last numpy-1 release (its video path is unused by TT models) and hold numpy<2
# (fixed by the tt-metal/ttnn env this runs inside). Revisit on a vLLM bump or if a model gains video.
opencv-python-headless==4.11.0.86
numpy>=1.24.4,<2
"""


def stage_thin_package(
    staged: Path,
    *,
    name: str,
    arch: str,
    model_py: Path,
    vllm_metadata: dict,
    tt_kernel_version: str,
    requirements: Optional[Path] = None,
    plugin_wheel: Optional[Path] = None,
    extra_wheels: Optional[List[Path]] = None,
    models_wheels: Optional[List[Path]] = None,
    vllm_wheel: Optional[Path] = None,
    vllm_version: str = VLLM_VERSION,
    with_vllm: bool = True,
    weights: Optional[WeightsRef] = None,
    device_count: int = 1,
    mesh: Optional[Mesh] = None,
    env: Optional[Dict[str, str]] = None,
    resources: Optional[Resources] = None,
    python_version: str = "3.12",
    tt_metal_version: str = "unknown",
) -> Manifest:
    """Materialize a v6 "thin" bundle (issue #29) and return its manifest.

    Ships: ``model.py`` (the runner), a ``requirements.txt`` of index pins (ttnn / tt-metal-models),
    the ``vllm_metadata.json`` (EXTRA_MODELS_DIR contract), generated ``install.sh``/``run.sh``, and
    — in ``wheels/`` — the **bundled wheels installed by path**: the ``vllm-tt-plugin``
    (``--plugin-wheel``, the vLLM integration) and any ``generic_op`` custom-op wheels
    (``extra_wheels``). ``models_wheels`` are also staged into ``wheels/`` but are NOT installed by
    path — they only ride along on ``--find-links`` so a ``requirements.txt`` pin that isn't on an
    index yet (e.g. a hand-built ``tt-metal-models`` wheel, ahead of its publish) still resolves.

    vLLM core is installed by ``install.sh`` as **stock upstream vLLM built with
    ``VLLM_TARGET_DEVICE=empty``** (the plugin's ``docs/install-vllm-tt.sh`` path — NOT the CUDA
    ``vllm`` on PyPI, NOT a fork). We ship a ``vllm-overrides.txt`` (numpy<2 / opencv pins) so that
    step doesn't clobber ttnn's numpy; the upstream ``common.txt`` is fetched at install unless the
    author bundles a pinned copy. Pass ``vllm_wheel`` to ship a prebuilt empty-target wheel (stock
    vLLM built empty) for a hermetic install instead of building from source. ``with_vllm=False``
    packages a non-vLLM model (no vLLM step). NO embedded ttnn wheel, NO metal tree —
    ttnn/tt-metal-models resolve from the index at install. Weights stay a pointer.

    NOTE (draft): reflects the #29 plan; fully functional once tt-metal-models publishes so the
    ttnn/tt-metal-models pins are real.
    """
    staged.mkdir(parents=True, exist_ok=True)

    # The runner, copied to the bundle root under its own name so `--main-class <module>:<Class>`
    # resolves it via PYTHONPATH=$HERE at serve time.
    model_dest = staged / Path(model_py).name
    shutil.copy2(model_py, model_dest)

    # requirements.txt: the author's index pins, or the #29 template with TODO lines to fill.
    if requirements is not None:
        shutil.copy2(requirements, staged / REQUIREMENTS)
    else:
        (staged / REQUIREMENTS).write_text(_THIN_REQUIREMENTS_TEMPLATE)

    # Bundled wheels -> wheels/, installed BY PATH: the vllm-tt-plugin (the vLLM integration — we
    # ship no custom vLLM fork), then any generic_op custom-op wheels. These are the things not on a
    # pinnable index; ttnn/tt-metal-models still come from requirements.txt.
    deps_wheels: List[str] = []
    ordered = [w for w in (plugin_wheel, *(extra_wheels or [])) if w is not None]
    if ordered:
        wheels_root = staged / WHEELS_DIR
        wheels_root.mkdir(exist_ok=True)
        for w in ordered:
            shutil.copy2(w, wheels_root / Path(w).name)
            deps_wheels.append(f"{WHEELS_DIR}/{Path(w).name}")

    # Wheels that only need to satisfy a requirements.txt pin locally (not installed by path) — a
    # locally-built tt-metal-models wheel ahead of its index publish is the motivating case.
    models_deps_wheels: List[str] = []
    for w in models_wheels or []:
        wheels_root = staged / WHEELS_DIR
        wheels_root.mkdir(exist_ok=True)
        shutil.copy2(w, wheels_root / Path(w).name)
        models_deps_wheels.append(f"{WHEELS_DIR}/{Path(w).name}")

    # vLLM core: stock upstream vLLM built empty-target (see Vllm). Ship the override file (numpy<2 /
    # opencv pin) so the common-deps install doesn't bump ttnn's numpy; the upstream common.txt is
    # fetched at install. An optional prebuilt empty-target wheel avoids building from source.
    vllm_spec: Optional[Vllm] = None
    if with_vllm:
        (staged / VLLM_OVERRIDES).write_text(_VLLM_OVERRIDES_TEMPLATE)
        vllm_rel: Optional[str] = None
        if vllm_wheel is not None:
            wheels_root = staged / WHEELS_DIR
            wheels_root.mkdir(exist_ok=True)
            shutil.copy2(vllm_wheel, wheels_root / Path(vllm_wheel).name)
            vllm_rel = f"{WHEELS_DIR}/{Path(vllm_wheel).name}"
        vllm_spec = Vllm(version=vllm_version, overrides=VLLM_OVERRIDES, wheel=vllm_rel)

    # vllm_metadata.json in the per-model subfolder under vllm_models/ (EXTRA_MODELS_DIR contract).
    safe_key = name.replace("/", "__")
    model_bundle = staged / METADATA_DIR / safe_key
    model_bundle.mkdir(parents=True, exist_ok=True)
    (model_bundle / VLLM_METADATA_NAME).write_text(json.dumps(vllm_metadata, indent=2))

    deps = Deps(
        python=python_version,
        requirements=REQUIREMENTS,
        wheels=deps_wheels,
        models_wheels=models_deps_wheels,
        wheels_dir=(WHEELS_DIR if (deps_wheels or models_deps_wheels
                                    or (vllm_spec and vllm_spec.wheel)) else None),
        vllm=vllm_spec,
        model_dir=".",
    )
    entrypoint = Entrypoint(
        **{"class": vllm_metadata["main_class"], "arch_name": vllm_metadata["arch"]}
    )
    manifest = Manifest(
        schema_version="6",
        name=name,
        tt_metal_version=tt_metal_version,
        arch=arch,
        device_count=device_count,
        producer=Producer(
            tt_kernel_version=tt_kernel_version,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            hostname=socket.gethostname(),
        ),
        weights=weights,
        entrypoint=entrypoint,
        mesh=mesh,
        env=env or {},
        resources=resources,
        deps=deps,
    )

    (staged / INSTALL_SCRIPT).write_text(render_install_sh(manifest))
    (staged / RUN_SCRIPT).write_text(render_run_sh(manifest))
    for s in (INSTALL_SCRIPT, RUN_SCRIPT):
        # Owner read/write only (0o600), NO execute bit: these scripts are always invoked as
        # `bash <script>` (runtime.install_self_contained / _serve_self_contained / the docs), so
        # they never need to be executable — least privilege (Cycode SAST: permissive file assignment).
        (staged / s).chmod(0o600)

    (staged / "tt_kernel_manifest.json").write_text(manifest.to_json())
    return manifest


__all__ = [
    "WHEELS_DIR",
    "METAL_DIR",
    "CUSTOM_OPS_DIR",
    "sha256_file",
    "parse_wheel_tags",
    "make_wheel_artifact",
    "host_python_tag",
    "host_incompatible_wheels",
    "render_install_sh",
    "render_run_sh",
    "stage_package",
    "stage_thin_package",
    "StagingError",
]
