# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""``tt-model package --container``: manifest -> Docker image -> staged HF repo dir.

Three jobs live here:

1. **Provenance.** Resolve ``source.tt_metal`` (a local checkout or a git ref) to a
   commit sha + version and pin every git ref under ``runtime:``, so the PUBLISHED
   manifest is fully pinned even when the author wrote a path or a branch name. This is
   the point of the whole path: a plugin that moved under a validated model is the bug
   class this prevents.
2. **Staging.** Assemble the build context (Dockerfile, the ``code/`` allowlist, the
   generated install/verify scripts, the lock) and, after the build, the HF repo
   directory (manifest JSON, README, ``code/``, ``image/`` as an exploded OCI layout).
3. **The long build, survivably.** A cold build is 2.5-4 hours. Output streams live and
   tees to a log whose ``tail -f`` command is printed BEFORE the wait starts. The child
   runs in its own process group behind an interrupt guard: on a TTY the first Ctrl-C
   only *warns* and the build continues; a second within the window cancels, keeps the
   caches, and reports the resume command.

This module holds no presentation. Progress arrives through an ``echo`` callable so the
CLI owns rendering, and everything else is pure enough to test without a docker daemon.
"""

from __future__ import annotations

import hashlib
import os
import shlex
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

from . import MANIFEST_NAME, __version__
from .container_manifest import (
    ContainerManifest,
    ContainerManifestError,
    GitSource,
    load_container_manifest,
)

BASE_IMAGE_DEFAULT = "ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-{ubuntu}-dev-amd64:latest"

# tt-metal's own top-level CMakeLists.txt refuses to configure without this file, and it
# is the first thing build_metal.sh does. Checking it here turns a failure that costs a
# base-image pull plus a CMake configure into an instant, actionable one.
SUBMODULE_SENTINEL = "tt_metal/third_party/umd/CMakeLists.txt"


class BuildError(RuntimeError):
    """A packaging step that must not proceed. The message is user-facing."""


def build_log_path(name: str) -> Path:
    d = Path.home() / ".cache" / "tt-model" / "build"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}.log"


# ------------------------------------------------------------------- provenance


def _git(repo: Path, *args: str) -> Optional[str]:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def _remote_url_containing_head(metal: Path, remote_refs: Optional[str]) -> Optional[str]:
    """Return the most relevant remote that actually contains ``HEAD``.

    A fork checkout commonly keeps upstream as ``origin`` and pushes the working
    branch to another remote.  Recording origin merely because it has that conventional
    name creates a plausible-looking but broken model-card link.  Prefer the current
    branch's tracking remote when it contains HEAD, then any containing remote, and use
    origin only as orientation when the commit has not been pushed anywhere.
    """
    remotes = (_git(metal, "remote") or "").splitlines()
    refs = {
        line.strip().lstrip("* ").split(" -> ", 1)[0]
        for line in (remote_refs or "").splitlines()
        if line.strip()
    }

    branch = _git(metal, "rev-parse", "--abbrev-ref", "HEAD")
    tracking_remote = (
        _git(metal, "config", "--get", f"branch.{branch}.remote")
        if branch and branch != "HEAD"
        else None
    )

    def contains_head(remote: str) -> bool:
        return any(ref.startswith(f"{remote}/") for ref in refs)

    candidates = []
    if tracking_remote and tracking_remote in remotes and contains_head(tracking_remote):
        candidates.append(tracking_remote)
    candidates.extend(remote for remote in remotes if contains_head(remote) and remote not in candidates)
    if "origin" in remotes and "origin" not in candidates:
        candidates.append("origin")
    candidates.extend(remote for remote in remotes if remote not in candidates)
    return _git(metal, "remote", "get-url", candidates[0]) if candidates else None


def scm_version(metal: Path) -> str:
    """The version the ttnn editable install should carry.

    tt-metal derives it with setuptools-scm against release tags only
    (``v[0-9]*.[0-9]*.[0-9]*``, dev/rc excluded). The same scheme is approximated from
    git directly so packaging does not require setuptools-scm on the host; the value is
    a label, not an API — SETUPTOOLS_SCM_PRETEND_VERSION makes the build take it as-is.
    """
    desc = _git(metal, "describe", "--tags", "--long", "--dirty",
                "--match", "v[0-9]*.[0-9]*.[0-9]*", "--exclude", "*-dev*",
                "--exclude", "*-rc*")
    if not desc:
        return "0.0.0.dev0"
    m = re.match(r"^v(\d+)\.(\d+)\.(\d+)-(\d+)-g([0-9a-f]+)(-dirty)?$", desc)
    if not m:
        return "0.0.0.dev0"
    major, minor, patch, distance, node, dirty = m.groups()
    if distance == "0":
        version = f"{major}.{minor}.{patch}"
    else:  # setuptools-scm's guess_next_dev: next patch, .devN
        version = f"{major}.{minor}.{int(patch) + 1}.dev{distance}"
    if dirty:
        version += f"+g{node}"
    return version


# NOT excluded: tests/ — tools/scaleout/fabric_manager #includes headers out of
# tests/tt_metal/test_utils, so the default build needs the tree present. It still never
# reaches the runtime image: that stage COPYs named directories only.
PLUGIN_CONTEXT_EXCLUDES = (
    ".git", "__pycache__", ".venv", "venv", "build", "dist", ".pytest_cache",
    ".mypy_cache", ".ruff_cache",
)


def _copy_plugin_tree(src: Path, dest: Path) -> None:
    """Stage a local plugin checkout, shipping what git considers part of the project.

    A hardcoded exclude list is not enough: a working plugin checkout accumulates runtime
    artifacts that dwarf the source — this was found against a real one carrying a 30 GB
    ``model_cache/`` beside 928 KB of ``src/``. The project already declares what is not
    part of it, in .gitignore, so ask git rather than guessing:

        git ls-files --cached --others --exclude-standard

    That is tracked files plus untracked-but-not-ignored ones, so uncommitted work still
    ships (the hermetic default) while ignored artifacts never do. A non-git directory
    falls back to the exclude list.
    """
    listed = subprocess.run(
        ["git", "-C", str(src), "ls-files", "--cached", "--others", "--exclude-standard"],
        capture_output=True, text=True,
    )
    if listed.returncode != 0:
        def _ignore(dirpath, names):
            return {n for n in names
                    if n in PLUGIN_CONTEXT_EXCLUDES or n.endswith((".pyc", ".egg-info"))}

        shutil.copytree(src, dest, symlinks=True, ignore=_ignore)
        return

    dest.mkdir(parents=True, exist_ok=True)
    for rel in listed.stdout.splitlines():
        if not rel:
            continue
        parts = Path(rel).parts
        # git handles project-specific ignores; this list handles what is never wanted,
        # including in a repo that simply forgot to gitignore its __pycache__.
        if any(p in PLUGIN_CONTEXT_EXCLUDES for p in parts) or rel.endswith(".pyc"):
            continue
        f = src / rel
        if not f.is_file():  # a deleted-but-tracked path
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)


METAL_CONTEXT_EXCLUDES = (
    ".git", ".cpmcache", "python_env", "generated", "built", "models", "docs",
    "tech_reports", "tt-train", "model_tracer", ".github", "infra", "jobs",
    "releases", "dockerfile", "contributing", ".claude",
)


def _copy_metal_tree(metal: Path, dest: Path) -> None:
    def _ignore(dirpath, names):
        top = Path(dirpath) == metal  # build trees are pruned at the ROOT only:
        drop = set()                  # "build" must not eat tt_metal/.../build_*.cpp
        for n in names:
            if n in METAL_CONTEXT_EXCLUDES or n.startswith(".venv"):
                drop.add(n)
            elif top and (
                n in ("build", "venv")
                or (n.startswith("build_") and not n.endswith(".sh"))
            ):
                drop.add(n)
            elif n.endswith(".log"):
                drop.add(n)
        return drop

    shutil.copytree(metal, dest, symlinks=True, ignore=_ignore)


@dataclass
class MetalSource:
    mode: str                # "local" | "git"
    context: Path            # dir handed to --build-context metalsrc=
    origin: Optional[Path]   # the author's actual checkout (local mode); code/ stages from here
    sha: Optional[str]
    describe: Optional[str]
    dirty: bool
    scm_version: str
    git_repo: Optional[str] = None  # git mode only
    git_ref: Optional[str] = None
    # Local mode: WHERE the sha came from, recorded so a reader can orient themselves
    # — not because anything resolves it. The image ships the tree, which is what lets
    # a local branch or fork be a first-class input rather than a compromise. `pushed`
    # simply says whether the commit happens to exist on a remote too.
    remote: Optional[str] = None
    branch: Optional[str] = None
    pushed: Optional[bool] = None  # is HEAD reachable from any remote branch?


def resolve_metal_source(m: ContainerManifest, scratch: Path) -> MetalSource:
    src = m.source.tt_metal
    if isinstance(src, str):
        metal = Path(src).expanduser()
        if not (metal / "tt_metal").is_dir():
            raise BuildError(
                f"source.tt_metal: {metal} does not look like a tt-metal checkout "
                "(no tt_metal/ directory)"
            )
        if not (metal / SUBMODULE_SENTINEL).is_file():
            raise BuildError(
                f"source.tt_metal: {metal} has uninitialised git submodules "
                f"({SUBMODULE_SENTINEL} is missing), so the image's tt-metal build would "
                "fail at CMake configure. Populate them first:\n"
                f"    git -C {metal} submodule update --init --recursive"
            )

        # A FILTERED COPY becomes the build context. Handing BuildKit the raw checkout
        # would transfer the whole worktree (.git alone is ~5.7 GB, plus .cpmcache,
        # python_env and build trees) before the Dockerfile's excludes ever run. The
        # tracked source is a few hundred MB; this copy takes seconds and IS the
        # hermetic input.
        filtered = scratch / "metalsrc"
        _copy_metal_tree(metal, filtered)
        # `git branch -r --contains HEAD` is empty when the commit exists only here,
        # which is the NORMAL case for this tool: community developers package from
        # local branches and forks, and the image carries the tree, so nothing ever
        # fetches this sha. Recorded as a fact about the build, not as a problem.
        on_remote = _git(metal, "branch", "-r", "--contains", "HEAD")
        return MetalSource(
            mode="local",
            context=filtered,
            origin=metal,
            sha=_git(metal, "rev-parse", "HEAD"),
            remote=_remote_url_containing_head(metal, on_remote),
            branch=_git(metal, "rev-parse", "--abbrev-ref", "HEAD"),
            pushed=bool(on_remote),
            # tt-metal's OWN describe invocation (cmake/version.cmake), so the stub tag
            # the Dockerfile creates reproduces the exact PROJECT_VERSION.
            describe=_git(metal, "describe", "--abbrev=10", "--first-parent",
                          "--dirty=-dirty"),
            dirty=bool(_git(metal, "status", "--porcelain")),
            scm_version=scm_version(metal),
        )

    # git mode: the Dockerfile clones; metalsrc becomes an empty placeholder context.
    placeholder = scratch / "empty-metalsrc"
    placeholder.mkdir(parents=True, exist_ok=True)
    sha = resolve_git_ref(src.repo, src.ref)
    return MetalSource(
        mode="git", context=placeholder, origin=None, sha=sha, describe=None,
        dirty=False, scm_version="0.0.0.dev0", git_repo=src.repo, git_ref=sha,
    )


def resolve_git_ref(repo: str, ref: str) -> str:
    """A ref on a remote -> a commit sha. This is where "ref: main" stops being a
    moving target and becomes provenance."""
    if re.fullmatch(r"[0-9a-f]{40}", ref):
        return ref
    ls = subprocess.run(
        ["git", "ls-remote", repo, ref, f"{ref}^{{}}"], capture_output=True, text=True
    )
    if ls.returncode == 0 and ls.stdout.strip():
        # prefer the peeled tag line when present
        return ls.stdout.strip().splitlines()[-1].split("\t")[0]
    if re.fullmatch(r"[0-9a-f]{7,39}", ref):
        return ref  # short sha: not resolvable via ls-remote, trust it
    raise BuildError(
        f"could not resolve ref {ref!r} on {repo} (git ls-remote found nothing and it "
        "is not a commit sha)"
    )


# ---------------------------------------------------------------------- staging


# Dropped from inside an allowlisted path regardless of what the author wrote. Keep this
# list NARROW and keep every entry justifiable: `source.code` promises "exactly what
# ships", so anything removed here is a promise quietly broken.
#
# Only two justifications qualify:
#   - build detritus that is regenerated (caches, venvs, egg-info, logs), and
#   - model weights, which are a POINTER in this design and would otherwise silently
#     turn a 2 GB image into a 60 GB one.
#
# A blanket `readiness_*` used to live here, meant for bring-up artifacts in autoport
# directories. It also matched `models/common/readiness_check` — a real package models
# import — and produced a ModuleNotFoundError in the finished image with nothing
# anywhere explaining why. Patterns that match a plausible PACKAGE name do not belong
# here; an author who does not want a directory shipped simply does not list it.
CODE_IGNORE = (
    "__pycache__", "*.pyc", ".pytest_cache", "*.egg-info", ".venv", "venv",
    "logs", "*.log",
    "*.safetensors", "*.pt", "*.pth", "*.ckpt", "*.bin", "*.refpt",
)


def stage_code(m: ContainerManifest, metal: Path, dest: Path) -> "StagedCode":
    """Copy the ``source.code`` allowlist into ``dest``, refusing missing paths.

    A missing path RAISES rather than shipping nothing: the silent miss is the failure
    mode where the image ImportErrors on a consumer long after the push looked fine.
    Returns the rendered file tree for the model card AND everything ``CODE_IGNORE``
    dropped, so a skip can be reported rather than discovered as an ImportError inside
    the finished image.
    """
    skipped: List[str] = []

    def _ignore(dirpath: str, names: List[str]) -> set:
        dropped = set(shutil.ignore_patterns(*CODE_IGNORE)(dirpath, names))
        for n in sorted(dropped):
            skipped.append(str(Path(dirpath).relative_to(metal) / n))
        return dropped

    entries: List[str] = []
    for rel in m.source.code:
        src = metal / rel
        if not src.exists():
            raise BuildError(
                f"source.code entry {rel!r} does not exist under {metal} — the "
                "allowlist names exactly what ships, so a missing entry is an error, "
                "not a skip"
            )
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(
                src, target, symlinks=False, dirs_exist_ok=True,
                ignore=_ignore,
            )
        else:
            shutil.copy2(src, target)
        entries.append(rel + ("/" if src.is_dir() else ""))
    return StagedCode(tree=entries, skipped=skipped)


def _sha256_tree(root: Path) -> str:
    """A stable digest of the staged code, so the manifest records WHAT shipped."""
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


@dataclass
class StagedCode:
    tree: List[str]        # rendered file tree, for the model card
    skipped: List[str]     # paths CODE_IGNORE dropped, so the caller can report them


@dataclass
class Staged:
    manifest: ContainerManifest
    ctx: Path                 # docker build context
    out: Path                 # HF repo staging dir
    image: str                # docker tag (provisional until retag_to_digest)
    digest: Optional[str] = None  # the image's own config digest, once built
    built: Dict[str, object] = field(default_factory=dict)
    build_args: Dict[str, str] = field(default_factory=dict)
    metal: Optional[MetalSource] = None
    code_tree: List[str] = field(default_factory=list)
    code_skipped: List[str] = field(default_factory=list)


def stage(manifest_path: Path, out_root: Optional[Path] = None) -> Staged:
    """Everything before ``docker build``: resolve, pin, and lay out the context."""
    from .launchers import launcher_for

    manifest_path = Path(manifest_path)
    m = load_container_manifest(manifest_path)
    launcher = launcher_for(m.kind)

    out = (out_root or Path.cwd() / "build") / m.name
    if out.exists():
        shutil.rmtree(out)
    ctx = out / "ctx"
    ctx.mkdir(parents=True)

    metal = resolve_metal_source(m, scratch=ctx)

    # -- pin every git ref under runtime: ------------------------------------------
    for key in ("vllm", "plugin"):
        entry = m.runtime.get(key)
        if isinstance(entry, dict) and entry.get("repo") and entry.get("ref"):
            entry["sha"] = resolve_git_ref(entry["repo"], entry["ref"])

    # -- a LOCAL plugin checkout, staged like the metal tree ---------------------------
    # v5 shipped the author's own plugin wheel; v5.1 could only clone a pushed ref, so an
    # author iterating on the plugin could not package what they were actually running.
    # The directory is always created (possibly empty) because the Dockerfile COPYs it
    # unconditionally and a missing path would fail the build for every other model.
    plugin_ctx = ctx / "plugin-src"
    plugin_entry = m.runtime.get("plugin") or {}
    plugin_path = plugin_entry.get("path") if isinstance(plugin_entry, dict) else None
    if plugin_path:
        src = Path(plugin_path).expanduser()
        if not (src / "pyproject.toml").is_file() and not (src / "setup.py").is_file():
            raise BuildError(
                f"runtime.plugin.path: {src} does not look like a python package "
                "(no pyproject.toml or setup.py)"
            )
        _copy_plugin_tree(src, plugin_ctx)
        plugin_entry["sha"] = _git(src, "rev-parse", "HEAD") or "unknown"
        plugin_entry["dirty"] = bool(_git(src, "status", "--porcelain"))
    else:
        plugin_ctx.mkdir(parents=True, exist_ok=True)

    # -- a LOCAL vLLM source tree or wheel, and any extra wheels -----------------------
    # v5 shipped the author's own vLLM wheel (--vllm-wheel) and extra wheels
    # (--extra-wheel); both directories are always created because the Dockerfile COPYs
    # them unconditionally.
    vllm_ctx = ctx / "vllm-src"
    wheels_ctx = ctx / "wheels"
    wheels_ctx.mkdir(parents=True, exist_ok=True)
    vllm_entry = m.runtime.get("vllm") or {}

    if isinstance(vllm_entry, dict) and vllm_entry.get("path"):
        src = Path(vllm_entry["path"]).expanduser()
        if not (src / "setup.py").is_file() and not (src / "pyproject.toml").is_file():
            raise BuildError(
                f"runtime.vllm.path: {src} does not look like a python package "
                "(no pyproject.toml or setup.py)"
            )
        _copy_plugin_tree(src, vllm_ctx)
        vllm_entry["sha"] = _git(src, "rev-parse", "HEAD") or "unknown"
        vllm_entry["dirty"] = bool(_git(src, "status", "--porcelain"))
    else:
        vllm_ctx.mkdir(parents=True, exist_ok=True)

    if isinstance(vllm_entry, dict) and vllm_entry.get("wheel"):
        # A wheel filename carries the version and platform tags, so authors reach for a
        # glob (`dist/vllm-*.whl`). Path().glob() rejects an absolute pattern, so split it.
        pat = Path(vllm_entry["wheel"]).expanduser()
        if any(ch in pat.name for ch in "*?["):
            found = sorted(pat.parent.glob(pat.name))
        else:
            found = [pat] if pat.is_file() else []
        if not found:
            raise BuildError(f"runtime.vllm.wheel: no wheel matches {vllm_entry['wheel']}")
        # The Dockerfile installs /ctx/wheels/vllm-*.whl, so the name must survive.
        for w in found:
            if not w.name.startswith("vllm-"):
                raise BuildError(
                    f"runtime.vllm.wheel: {w.name} must be a vllm wheel (vllm-*.whl)"
                )
            shutil.copy2(w, wheels_ctx / w.name)
        vllm_entry["wheel_name"] = found[0].name

    for extra in (m.runtime.get("wheels") or []):
        w = Path(extra).expanduser()
        if not w.is_file():
            raise BuildError(f"runtime.wheels: {w} does not exist")
        if w.name.startswith("vllm-"):
            raise BuildError(
                f"runtime.wheels: {w.name} would be picked up as the vLLM wheel; use "
                "runtime.vllm.wheel for that"
            )
        shutil.copy2(w, wheels_ctx / w.name)

    # -- code/ ----------------------------------------------------------------------
    code_dir = ctx / "code"
    if metal.mode == "local":
        staged_code = stage_code(m, metal.origin, code_dir)
    else:
        # git mode: shallow-clone just to stage code/ (the image build re-clones)
        tmp = ctx / "metal-for-code"
        subprocess.run(
            ["git", "clone", "--filter=blob:none", metal.git_repo, str(tmp)], check=True
        )
        subprocess.run(["git", "-C", str(tmp), "checkout", metal.sha], check=True)
        staged_code = stage_code(m, tmp, code_dir)
        shutil.rmtree(tmp)

    # -- lock ------------------------------------------------------------------------
    lock_name = m.runtime.get("lock")
    if lock_name:
        lock_src = manifest_path.parent / lock_name
        if not lock_src.exists():
            raise BuildError(f"runtime.lock names {lock_src}, which does not exist")
        shutil.copy2(lock_src, ctx / "requirements.lock")

    # -- generated scripts + the packaged docker assets --------------------------------
    header = "#!/bin/bash\nset -euxo pipefail\n"
    (ctx / "install_engine.sh").write_text(
        header + "\n".join(launcher.install_lines(m)) + "\n"
    )
    (ctx / "verify.sh").write_text(header + "\n".join(launcher.verify_lines(m)) + "\n")
    (ctx / "serve-default.sh").write_text(render_default_serve(m, launcher))
    pkg_docker = Path(__file__).parent / "docker"
    shutil.copy2(pkg_docker / "Dockerfile", ctx / "Dockerfile")
    shutil.copy2(pkg_docker / "entrypoint.sh", ctx / "entrypoint.sh")

    # -- provenance + image tag ---------------------------------------------------------
    # PROVISIONAL tag, used only to name the build output. The published tag is derived
    # from the image's own digest once it exists (see retag_to_digest): a tag taken from
    # tt-metal's HEAD is not the image's identity and is wrong in both directions — a
    # republish that changes the plugin pin, the code allowlist or a serve setting without
    # moving that sha reuses the tag, so a consumer's `pull` sees the tag present, skips
    # the load and serves the OLD image while reporting success; and any unrelated commit
    # in the metal tree mints a new 10 GB tag for byte-identical content.
    image = f"tt-model/{m.name}:build-{(metal.sha or 'dev')[:9]}"
    built: Dict[str, object] = {
        "image": image,
        # the repo the author named in the manifest, so `push <dir>` can default to it
        "repo": m.repo,
        "tt_model_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tt_metal": {
            "sha": metal.sha, "describe": metal.describe, "dirty": metal.dirty,
            "scm_version": metal.scm_version, "mode": metal.mode,
            # where the sha lives, so a consumer can actually find it
            "remote": metal.remote or metal.git_repo,
            "branch": metal.branch,
            "pushed": metal.pushed,
        },
        "code_sha256": _sha256_tree(code_dir),
    }
    for key in ("vllm", "plugin"):
        entry = m.runtime.get(key)
        if isinstance(entry, dict) and entry.get("wheel_name"):
            built[key] = {"wheel": entry["wheel_name"]}
            continue
        if isinstance(entry, dict) and entry.get("sha"):
            built[key] = {"sha": entry["sha"]}
            if entry.get("repo"):
                built[key]["repo"] = entry["repo"]
            if entry.get("path"):
                # Recorded, and flagged when uncommitted — the same honesty the metal
                # tree gets, so "pinned" never overstates what was actually shipped.
                built[key]["path"] = str(Path(entry["path"]).expanduser())
                built[key]["dirty"] = bool(entry.get("dirty"))

    # -- docker build args ---------------------------------------------------------------
    # EXTRA_MODELS_DIR is the directory the plugin SCANS for per-model
    # vllm_metadata.json files. A model may state it outright (extra_models_dir) — the
    # layouts in the wild differ — or leave it to be derived from a pip-installed
    # extension. Registering zero architectures is a silent failure, so verify.sh checks
    # the directory actually contains something.
    ext = m.runtime.get("extension")
    emd = m.runtime.get("extra_models_dir")
    if emd:
        extra_models_dir = f"/opt/tt-metal/{emd}"
    elif ext:
        extra_models_dir = f"/opt/tt-metal/{ext}/extra_models"
    else:
        extra_models_dir = ""
    build_args = {
        "BASE_IMAGE": BASE_IMAGE_DEFAULT.format(ubuntu=m.source.ubuntu),
        "UBUNTU_VERSION": m.source.ubuntu,
        "PYTHON_VERSION": m.source.python,
        "METAL_MODE": metal.mode,
        "SCM_VERSION": metal.scm_version,
        "METAL_DESCRIBE": metal.describe or "v0.0.0",
        "EXTRA_MODELS_DIR": extra_models_dir,
        # suppress builtin registration ONLY when the model brings its own registration —
        # "0" for a builtin-registry model would register zero architectures
        "TT_VLLM_BUILTIN_MODELS": "0" if extra_models_dir else "",
        "TT_MODEL_KIND": m.kind,
        "MODEL_NAME": m.name,
        "MODEL_REPO": m.repo,
        "MODEL_WEIGHTS": m.weights_repo,
        "MODEL_ARCH": m.arch,
        "MODEL_PROFILES": ",".join(m.profile_names()),
        # The tag is the image's digest, which is the right identity but tells a human
        # nothing. `docker inspect` is where someone triaging a mystery image looks, so the
        # source commits belong in the labels. org.opencontainers.image.revision is the
        # standard key for "the commit this was built from".
        "MODEL_TT_METAL_SHA": metal.sha or "",
        "MODEL_TT_METAL_DESCRIBE": metal.describe or "",
        "MODEL_PLUGIN_SHA": str(
            ((m.runtime.get("plugin") or {}).get("sha")
             or (m.runtime.get("plugin") or {}).get("version") or "")
        ),
    }
    if metal.mode == "git":
        build_args["METAL_GIT_REPO"] = metal.git_repo or ""
        build_args["METAL_GIT_REF"] = metal.git_ref or ""

    return Staged(manifest=m, ctx=ctx, out=out, image=image, built=built,
                  build_args=build_args, metal=metal, code_tree=staged_code.tree,
                  code_skipped=staged_code.skipped)


def render_default_serve(m: ContainerManifest, launcher) -> str:
    """The image's default CMD: serve the default profile, correctly configured.

    Without this the image had NO default command — ``ENTRYPOINT`` simply exec'd its
    arguments. A Docker image invites ``docker run``, but the correct invocation lived
    only in the manifest, on the host, inside tt-model: the mesh, the tt additional-config,
    the builtin-registry suppression and the launcher flags are all supplied by
    ``tt-model serve``. Run directly, the server started and emitted nonsense — no error
    anywhere. Baking the default profile's env and argv makes the image self-describing:
    a bare ``docker run`` either does the right thing or says exactly what is missing.

    tt-model still passes explicit argv, which overrides CMD, so its own path is unchanged.
    """
    # The wire manifest is what the launcher consumes; render a throwaway one just to
    # resolve the default profile the same way a consumer would.
    wire = m.to_wire(image_tag="unresolved", tt_metal_version="unresolved",
                     tt_kernel_version=__version__, hostname="", created_at="")
    assert wire.container is not None
    profile = wire.container.resolve_profile()
    argv = launcher.serve_argv(wire, profile)
    env = launcher.serve_env(wire, profile)

    exports = "\n".join(
        f"export {k}={shlex.quote(str(v))}" for k, v in sorted(env.items())
    )
    command = " ".join(shlex.quote(a) for a in argv)
    port = profile.port or 8000
    return f"""#!/bin/bash
# Generated by tt-model. The default profile ({profile.name}) of this image, launched the
# way `tt-model serve` launches it. Overridden by any command passed to `docker run`.
set -euo pipefail

if [ ! -e /dev/tenstorrent ]; then
  cat >&2 <<'USAGE'
This image needs Tenstorrent devices and a specific host setup, and was started without
them. It is normally launched with:

    tt-model serve <org/name>

To run it directly, all of these are required:

    docker run --rm \\
      --device /dev/tenstorrent \\
      --ipc host \\
      --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \\
      --volume "$HOME/.cache/huggingface:/hf" \\
      --volume "<a writable dir>:/cache" \\
      --user "$(id -u):$(id -g)" \\
      --publish {port}:{port} \\
      <image>

The hugepages mount must be exactly /dev/hugepages-1G: umd matches that path in
/proc/mounts. Weights are NOT in this image; they are read from the mounted HF cache.
USAGE
  exit 1
fi

{exports}

exec {command}
"""


def build_argv(staged: Staged) -> List[str]:
    argv = ["docker", "build", "--progress=plain", "--tag", staged.image,
            "--build-context", f"metalsrc={staged.metal.context}"]
    for k, v in staged.build_args.items():
        argv += ["--build-arg", f"{k}={v}"]
    argv += [str(staged.ctx)]
    return argv


# ------------------------------------------------------------- the interrupt guard

_STAGE_RE = re.compile(r"^#\d+ \[(?P<stage>[^]]+)\]")


class InterruptGuard:
    """Run a long child survivably.

    The child gets its own process group — a terminal Ctrl-C otherwise goes straight to
    it and there is nothing left to intercept. On a TTY the first SIGINT prints a warning
    (elapsed, current stage, what a cancel costs) and the work CONTINUES; only a second
    SIGINT within ``window_s`` cancels. SIGTERM, or SIGINT with no TTY (nobody there to
    press twice), cancels immediately.
    """

    def __init__(self, describe: str, window_s: float = 10.0, on_cancel_note: str = "",
                 tty: Optional[bool] = None, warn: Callable[[str], None] = None):
        self.describe = describe
        self.window_s = window_s
        self.on_cancel_note = on_cancel_note
        self.tty = sys.stdin.isatty() if tty is None else tty
        self.warn = warn or (lambda s: sys.stderr.write(s))
        self.started = time.monotonic()
        self.armed_until = 0.0
        self.cancelled = False
        self.current_stage = ""
        self._proc: Optional[subprocess.Popen] = None
        self._old: Dict[int, object] = {}

    def __enter__(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._old[sig] = signal.signal(sig, self._handle)
        return self

    def __exit__(self, *exc):
        for sig, handler in self._old.items():
            signal.signal(sig, handler)
        return False

    def spawn(self, argv: List[str], **popen_kw) -> subprocess.Popen:
        self._proc = subprocess.Popen(argv, start_new_session=True, **popen_kw)
        return self._proc

    def elapsed(self) -> str:
        s = int(time.monotonic() - self.started)
        return (f"{s // 3600}h {s % 3600 // 60:02d}m" if s >= 3600
                else f"{s // 60}m {s % 60:02d}s")

    def note_line(self, line: str) -> None:
        """Feed output through so the warning can name the stage being lost."""
        m = _STAGE_RE.match(line)
        if m:
            self.current_stage = m.group("stage")

    def _handle(self, sig, frame):
        now = time.monotonic()
        if sig == signal.SIGINT and self.tty and now >= self.armed_until and not self.cancelled:
            self.armed_until = now + self.window_s
            stage = f"   ·   current stage: {self.current_stage}" if self.current_stage else ""
            self.warn(
                f"\n⚠  Interrupt received — {self.describe} is STILL RUNNING.\n"
                f"     elapsed   {self.elapsed()}{stage}\n"
                f"     {self.on_cancel_note}\n"
                f"   Press Ctrl-C again within {int(self.window_s)}s to cancel, "
                f"or ignore this to continue.\n"
            )
            return
        self._cancel()

    def _cancel(self):
        self.cancelled = True
        if self._proc and self._proc.poll() is None:
            try:
                os.killpg(self._proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and self._proc.poll() is None:
                time.sleep(0.2)
            if self._proc.poll() is None:
                try:
                    os.killpg(self._proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


# ---------------------------------------------------------------------- the build


def run_build(staged: Staged, echo: Optional[Callable[[str], None]] = None) -> None:
    """``docker build``, streamed live + teed to a log, under the interrupt guard."""
    log_path = build_log_path(staged.manifest.name)
    argv = build_argv(staged)

    with open(log_path, "w") as log, InterruptGuard(
        "the build",
        on_cancel_note=("Cancelling loses this stage's remaining work. The BuildKit/"
                        "ccache/CPM caches stay warm, so a re-run resumes there."),
    ) as guard:
        log.write("+ " + " ".join(argv) + "\n")
        proc = guard.spawn(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, errors="replace")
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                guard.note_line(line)
                log.write(line)
                log.flush()
                if echo:
                    echo(line.rstrip("\n"))
        finally:
            code = proc.wait()

        if guard.cancelled:
            raise KeyboardInterrupt(
                f"build cancelled after {guard.elapsed()}\n"
                f"    kept      BuildKit/ccache/CPM caches — a re-run resumes there\n"
                f"    log       {log_path}"
            )
        if code != 0:
            raise BuildError(f"docker build failed (exit {code}); full log: {log_path}")


def image_digest(image: str) -> str:
    """The image's own config digest — what ``docker inspect`` reports as ``.Id``.

    This is the image's identity: it changes if and only if the image changes. Everything
    that made the old tag unreliable (the plugin pin, the code allowlist, serve settings,
    the base image) is inside it.
    """
    r = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
        capture_output=True, text=True,
    )
    digest = r.stdout.strip()
    if r.returncode != 0 or not digest.startswith("sha256:"):
        raise BuildError(
            f"could not read the digest of {image}: {(r.stderr or r.stdout).strip()[:300]}"
        )
    return digest


def retag_to_digest(staged: "Staged") -> str:
    """Retag the built image by its digest and return the final tag.

    Done BEFORE the OCI export so the exported layout carries the final name — the tag a
    consumer's ``docker load`` will produce has to be the one the manifest records, or
    ``pull`` looks for an image that is present under another name.

    The provisional build tag is dropped afterwards: leaving both would make every build
    look like two images to ``docker images``.
    """
    digest = image_digest(staged.image)
    final = f"tt-model/{staged.manifest.name}:{digest.split(':', 1)[1][:12]}"
    if final != staged.image:
        r = subprocess.run(["docker", "tag", staged.image, final],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise BuildError(f"could not tag {final}: {r.stderr.strip()[:300]}")
        subprocess.run(["docker", "image", "rm", staged.image],
                       capture_output=True, text=True)
    staged.image = final
    staged.digest = digest
    staged.built["image"] = final
    staged.built["image_digest"] = digest
    return final


def freeze_from_image(image: str) -> str:
    """The exact python env inside the built image, as a requirements lock.

    This is what turns the FIRST build of a model (which resolves live) into every later
    build being reproducible: commit the result and name it under ``runtime.lock``.
    """
    code = (
        "import importlib.metadata as md\n"
        "for d in sorted(md.distributions(), key=lambda d: (d.metadata['Name'] or '').lower()):\n"
        "    print(f\"{d.metadata['Name']}=={d.version}\")\n"
    )
    r = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "/opt/tt-venv/bin/python", image, "-c", code],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise BuildError(f"could not freeze the image env: {r.stderr[-800:]}")
    return r.stdout


def _https_repo_url(repo: str) -> Optional[str]:
    """A git remote in any common spelling -> a browsable https URL, or None."""
    if repo.startswith("git@"):  # git@github.com:org/repo(.git)
        host, sep, path = repo[4:].partition(":")
        if not sep:
            return None
        repo = f"https://{host}/{path}"
    if repo.endswith(".git"):
        repo = repo[:-4]
    return repo.rstrip("/") if repo.startswith(("https://", "http://")) else None


def _pinned_sha(sha: str, repo: Optional[str], *, public: bool = True,
                flags: str = "") -> str:
    """A provenance cell. The sha appears ONLY as a working public link: a commit a
    reader cannot fetch (not pushed, or from a local path) is not shown at all — the
    cell says where the build came from instead."""
    url = _https_repo_url(repo) if (repo and public) else None
    if url and sha and sha != "unknown":
        cell = f"[`{sha}`]({url}/commit/{sha})"
    else:
        cell = "a local checkout — commit not published"
    return cell + (f" {flags}" if flags else "")


TT_MODEL_MANAGER_URL = "https://github.com/tenstorrent/tt-model-manager"


def render_model_card(m: ContainerManifest, built: Dict[str, object]) -> str:
    """The generated README.md for the HF repo.

    The order is the reader's order: what the model is and what hardware it needs
    first, then how to run it, then (only when there is a choice) the profile table,
    then provenance. Provenance names each component by its official name and shows a
    commit only as a working public link.
    """
    from . import TT_MODEL_TAG
    from .launchers import launcher_for

    tt_metal = built.get("tt_metal") or {}
    tags = sorted({TT_MODEL_TAG, m.arch, m.kind, "tt-model-container"})
    lines = [
        "---",
        "tags:",
        *[f"- {t}" for t in tags],
        "---",
        "",
        f"# {m.name}",
        "",
    ]
    if m.card and m.card.description:
        lines += [m.card.description.rstrip(), ""]

    # Hardware requirement, up front. One profile: the whole launch config in a
    # sentence. Several: the targets here, the details in the table below.
    profiles = [m.resolve_profile(n) for n in m.profile_names()]
    if len(profiles) == 1:
        p = profiles[0]
        detail = []
        if p.max_model_len:
            detail.append(f"{p.max_model_len:,}-token context")
        if p.max_num_seqs:
            detail.append(f"up to {p.max_num_seqs} concurrent sequences")
        lines += [
            f"Runs on **{p.hardware}** (mesh `{p.mesh_device}`)"
            + (" — " + ", ".join(detail) if detail else "") + ".",
            "",
        ]
    else:
        targets = " or ".join(f"**{p.hardware}**" for p in profiles if p.hardware)
        lines += [f"Runs on {targets} — see the serve profiles below.", ""]

    lines += [
        f"Packaged and published with [tt-model-manager]({TT_MODEL_MANAGER_URL}) "
        f"{built.get('tt_model_version', '')} (manifest schema {m.schema_version}).",
        "",
        "## Quickstart",
        "",
        "```bash",
        f"tt-model pull  {m.repo}",
        f"tt-model serve {m.repo}",
        "```",
        "",
        f"`pull` downloads the Docker image and the "
        f"[`{m.weights_repo}`](https://huggingface.co/{m.weights_repo}) weights"
        + (f" at `{m.weights_ref.revision}`" if m.weights_ref.revision else "")
        + " (into your HF cache; they are not in the image). `serve` starts an "
        f"OpenAI-compatible server on port {m.serve.port or 8000}; the first start "
        "compiles kernels for your device, which takes several minutes, and the "
        f"server is ready when it logs `{launcher_for(m.kind).READY_LINE}`.",
        "",
    ]
    if m.card and m.card.quickstart:
        lines += [m.card.quickstart.rstrip(), ""]
    if len(profiles) > 1:
        lines += [
            "## Serve profiles",
            "",
            "One image serves every profile below; pick one with `--profile`.",
            "",
            "| profile | hardware | mesh | max_num_seqs | max_model_len |",
            "| --- | --- | --- | --- | --- |",
        ]
        default = m.resolved_default()
        for name in m.profile_names():
            p = m.resolve_profile(name)
            label = f"`{name}`" + (" *(default)*" if name == default else "")
            lines.append(
                f"| {label} | {p.hardware or ''} | {p.mesh_device or ''} | "
                f"{p.max_num_seqs or ''} | {p.max_model_len or ''} |"
            )
        lines.append("")
    lines += [
        "## Provenance",
        "",
        "The exact sources the image was built from — `code/` in this repo is "
        "byte-identical to the model code inside the image:",
        "",
        "| component | built from |",
        "| --- | --- |",
    ]

    # tt-metal: the commit, shown only as a working public link.
    metal_sha = str(tt_metal.get("sha") or "unknown")
    dirty_flag = "*(dirty tree — the image includes uncommitted changes)*"
    lines.append(
        "| tt-metal | "
        + _pinned_sha(metal_sha, tt_metal.get("remote"),
                      public=tt_metal.get("pushed") is not False,
                      flags=dirty_flag if tt_metal.get("dirty") else "") + " |"
    )

    # vLLM: a released version, the author's own wheel, or a pinned checkout (the cell's
    # link names the actual source — upstream release or the tenstorrent/vllm fork).
    vllm_built = built.get("vllm")
    vllm_rt = m.runtime.get("vllm") or {}
    if isinstance(vllm_built, dict) and vllm_built.get("wheel"):
        lines.append(f"| vLLM | `{vllm_built['wheel']}` — a wheel the author built |")
    elif isinstance(vllm_built, dict) and vllm_built.get("sha"):
        cell = _pinned_sha(
            str(vllm_built["sha"]), vllm_built.get("repo"),
            flags="*(dirty tree)*" if vllm_built.get("dirty") else "",
        )
        lines.append(f"| vLLM | {cell} |")
    elif vllm_rt.get("version"):
        v = vllm_rt["version"]
        lines.append(
            f"| vLLM | [`v{v}`](https://github.com/vllm-project/vllm/releases/tag/v{v}) |"
        )

    # The Tenstorrent platform plugin, by its official name — never just "plugin".
    plugin_built = built.get("plugin")
    plugin_rt = m.runtime.get("plugin") or {}
    if isinstance(plugin_built, dict) and plugin_built.get("sha"):
        cell = _pinned_sha(
            str(plugin_built["sha"]), plugin_built.get("repo"),
            flags="*(dirty tree — the image includes uncommitted changes)*"
            if plugin_built.get("dirty") else "",
        )
        lines.append(f"| vllm-tt-plugin | {cell} |")
    elif plugin_rt.get("version"):
        v = plugin_rt["version"]
        lines.append(
            f"| vllm-tt-plugin | [`v{v}`](https://pypi.org/project/vllm-tt-plugin/{v}/) |"
        )

    lines += [
        f"| `code/` digest | `{str(built.get('code_sha256', ''))[:16]}` (sha256, "
        "first 16 hex digits) |",
        f"| built | {built.get('created_at', '')} by tt-model {built.get('tt_model_version', '')} |",
        "",
    ]
    return "\n".join(lines) + "\n"


def finalize(staged: Staged, *, echo: Optional[Callable[[str], None]] = None) -> Path:
    """After a successful build: freeze, export the OCI layout, write the repo dir."""
    from . import oci

    out = staged.out
    m = staged.manifest

    # requirements.lock: from the image when this build resolved live; passed through
    # unchanged when the build installed from an existing lock.
    # Identity first: the export and the manifest must both name the final tag.
    retag_to_digest(staged)

    lock = staged.ctx / "requirements.lock"
    lock_text = lock.read_text() if lock.exists() else freeze_from_image(staged.image)
    (out / "requirements.lock").write_text(lock_text)
    m.runtime["lock"] = "requirements.lock"

    # code/ MOVES from the build context into the repo dir — the same directory the
    # image was built from, so what a reviewer reads on the Hub is byte-identical to
    # what runs.
    shutil.move(str(staged.ctx / "code"), str(out / "code"))

    if echo:
        echo("exporting image to an OCI layout (most of the repo's size)")
    oci.save(staged.image, out / "image")

    (out / "README.md").write_text(
        render_model_card(m, staged.built)
    )
    wire = m.to_wire(
        image_tag=staged.image,
        digest=staged.digest,
        tt_metal_version=str((staged.built.get("tt_metal") or {}).get("scm_version")
                             or "unknown"),
        tt_kernel_version=__version__,
        built=staged.built,
    )
    (out / MANIFEST_NAME).write_text(wire.to_json())

    shutil.rmtree(staged.ctx, ignore_errors=True)
    return out
