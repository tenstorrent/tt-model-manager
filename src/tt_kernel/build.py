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
        return MetalSource(
            mode="local",
            context=filtered,
            origin=metal,
            sha=_git(metal, "rev-parse", "HEAD"),
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
    image: str                # docker tag
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
    pkg_docker = Path(__file__).parent / "docker"
    shutil.copy2(pkg_docker / "Dockerfile", ctx / "Dockerfile")
    shutil.copy2(pkg_docker / "entrypoint.sh", ctx / "entrypoint.sh")

    # -- provenance + image tag ---------------------------------------------------------
    image = f"tt-model/{m.name}:{(metal.sha or 'dev')[:9]}"
    built: Dict[str, object] = {
        "image": image,
        # the repo the author named in the manifest, so `push <dir>` can default to it
        "repo": m.repo,
        "tt_model_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tt_metal": {
            "sha": metal.sha, "describe": metal.describe, "dirty": metal.dirty,
            "scm_version": metal.scm_version, "mode": metal.mode,
        },
        "code_sha256": _sha256_tree(code_dir),
    }
    for key in ("vllm", "plugin"):
        entry = m.runtime.get(key)
        if isinstance(entry, dict) and entry.get("sha"):
            built[key] = {"repo": entry["repo"], "sha": entry["sha"]}

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
        "MODEL_WEIGHTS": m.weights,
        "MODEL_ARCH": m.arch,
        "MODEL_PROFILES": ",".join(m.profile_names()),
    }
    if metal.mode == "git":
        build_args["METAL_GIT_REPO"] = metal.git_repo or ""
        build_args["METAL_GIT_REF"] = metal.git_ref or ""

    return Staged(manifest=m, ctx=ctx, out=out, image=image, built=built,
                  build_args=build_args, metal=metal, code_tree=staged_code.tree,
                  code_skipped=staged_code.skipped)


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


def render_model_card(m: ContainerManifest, built: Dict[str, object],
                      code_tree: List[str]) -> str:
    """The generated README.md for the HF repo.

    The serve-profile table travels WITH the model instead of living in a quickstart doc
    someone has to find, and the provenance block is the pinned truth of what was built.
    """
    from . import TT_MODEL_TAG

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
        "A **tt-model container package**: the serving platform ships as a Docker image, "
        "so a consumer needs only Docker and a Tenstorrent card — no tt-metal, no vLLM, "
        "no venv on the host.",
        "",
        "## Serve it",
        "",
        "```bash",
        f"tt-model pull  {m.repo}",
        f"tt-model serve {m.repo}",
        "```",
        "",
    ]
    if m.card and m.card.quickstart:
        lines += ["## Quickstart", "", m.card.quickstart.rstrip(), ""]
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
    lines += [
        "",
        "## What is inside",
        "",
        f"- **weights**: [`{m.weights}`](https://huggingface.co/{m.weights}) — "
        "downloaded to *your* HF cache at pull time, never baked into the image",
        f"- **arch**: {m.arch}",
        f"- **serving stack**: `{m.kind}`",
        "",
        "## Provenance",
        "",
        "Everything below is pinned; the image was built from exactly these.",
        "",
        "| component | pinned to |",
        "| --- | --- |",
        f"| tt-metal | `{tt_metal.get('sha') or 'unknown'}`"
        + (" *(dirty tree)*" if tt_metal.get("dirty") else "") + " |",
    ]
    for key in ("vllm", "plugin"):
        entry = built.get(key)
        if isinstance(entry, dict):
            lines.append(f"| {key} | `{entry.get('sha')}` |")
    lines += [
        f"| code digest | `{str(built.get('code_sha256', ''))[:16]}` |",
        f"| built | {built.get('created_at', '')} by tt-model {built.get('tt_model_version', '')} |",
        "",
        "## Shipped code",
        "",
        "`code/` in this repo is byte-identical to what runs inside the image.",
        "",
    ]
    lines += [f"- `{e}`" for e in code_tree]
    return "\n".join(lines) + "\n"


def finalize(staged: Staged, *, echo: Optional[Callable[[str], None]] = None) -> Path:
    """After a successful build: freeze, export the OCI layout, write the repo dir."""
    from . import oci

    out = staged.out
    m = staged.manifest

    # requirements.lock: from the image when this build resolved live; passed through
    # unchanged when the build installed from an existing lock.
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
        render_model_card(m, staged.built, staged.code_tree)
    )
    wire = m.to_wire(
        image_tag=staged.image,
        tt_metal_version=str((staged.built.get("tt_metal") or {}).get("scm_version")
                             or "unknown"),
        tt_kernel_version=__version__,
        built=staged.built,
    )
    (out / MANIFEST_NAME).write_text(wire.to_json())

    shutil.rmtree(staged.ctx, ignore_errors=True)
    return out
