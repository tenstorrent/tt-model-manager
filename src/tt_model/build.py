# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""``tt-model package``: manifest -> Docker image -> staged HF repo directory.

Three jobs live here:

1. **Provenance.** Resolve ``source.tt_metal`` (a local checkout or a git ref) to a
   commit sha + version, and rewrite the manifest with a ``built:`` block so the
   published file is fully pinned even when the author wrote a path or a branch name.
2. **Staging.** Assemble the build context (Dockerfile, the ``code/`` allowlist, the
   generated per-type install/verify scripts, the lock file) and, after the build, the
   HF repo directory (manifest, README, code/, image/ as an exploded OCI layout).
3. **The long build, survivably.** ``docker build`` output streams live in the
   terminal it was fired in AND tees to a log file whose ``tail -f`` command is printed
   before the wait starts. The child runs in its own process group behind an interrupt
   guard: on a TTY the first Ctrl-C only *warns* (elapsed time, current stage, what a
   cancel would cost) and the build continues; a second Ctrl-C within the window
   cancels and cleans up — partial staging removed, BuildKit/ccache caches kept, the
   log kept, and the exact resume command printed.
"""

from __future__ import annotations

import hashlib
import json
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
from typing import Dict, List, Optional

from . import __version__
from .manifest import Manifest, ManifestError, load_manifest

BASE_IMAGE_DEFAULT = "ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-{ubuntu}-dev-amd64:latest"


class BuildError(RuntimeError):
    pass


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
    (``v[0-9]*.[0-9]*.[0-9]*``, dev tags excluded). We approximate the same scheme from
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
# tests/tt_metal/test_utils, so the default build needs the tree present (it still
# never reaches the runtime image: that stage COPYs named dirs only).
METAL_CONTEXT_EXCLUDES = (
    ".git", ".cpmcache", "python_env", "generated", "built", "models", "docs",
    "tech_reports", "tt-train", "model_tracer", ".github", "infra", "jobs",
    "releases", "dockerfile", "contributing", ".claude",
)


def _copy_metal_tree(metal: Path, dest: Path) -> None:
    def _ignore(dirpath, names):
        top = Path(dirpath) == metal   # build trees are pruned at the ROOT only:
        drop = set()                   # "build" must not eat tt_metal/.../build_*.cpp etc.
        for n in names:
            if n in METAL_CONTEXT_EXCLUDES or n.startswith(".venv"):
                drop.add(n)
            elif top and (n in ("build", "venv") or (n.startswith("build_") and
                                                     not n.endswith(".sh"))):
                drop.add(n)
            elif n.endswith(".log"):
                drop.add(n)
        return drop

    shutil.copytree(metal, dest, symlinks=True, ignore=_ignore)


@dataclass
class MetalSource:
    mode: str                 # "local" | "git"
    context: Path             # dir handed to --build-context metalsrc= (filtered copy)
    origin: Optional[Path]    # the author's actual checkout (local mode) — code/ stages from here
    sha: Optional[str]        # commit, when known
    describe: Optional[str]
    dirty: bool
    scm_version: str
    git_repo: Optional[str] = None  # git mode only
    git_ref: Optional[str] = None


def resolve_metal_source(m: Manifest, scratch: Path) -> MetalSource:
    src = m.source.tt_metal
    if isinstance(src, str):
        metal = Path(src).expanduser()
        if not (metal / "tt_metal").is_dir():
            raise BuildError(
                f"source.tt_metal: {metal} does not look like a tt-metal checkout "
                "(no tt_metal/ directory)"
            )
        sha = _git(metal, "rev-parse", "HEAD")
        status = _git(metal, "status", "--porcelain")
        # A filtered copy becomes the build context: handing BuildKit the raw checkout
        # would transfer the whole worktree (.git 4.6G, .cpmcache 3.6G, python_env
        # 3.8G, build trees ...) before the Dockerfile's excludes ever run. The tracked
        # source is ~400 MB; this copy takes seconds and IS the hermetic input.
        filtered = scratch / "metalsrc"
        _copy_metal_tree(metal, filtered)
        return MetalSource(
            mode="local", context=filtered, origin=metal, sha=sha,
            # tt-metal's OWN describe invocation (cmake/version.cmake), so the stub tag
            # the Dockerfile creates reproduces the exact PROJECT_VERSION
            describe=_git(metal, "describe", "--abbrev=10", "--first-parent",
                          "--dirty=-dirty"),
            dirty=bool(status), scm_version=scm_version(metal),
        )
    # git mode: the Dockerfile clones; metalsrc becomes an empty placeholder context.
    placeholder = scratch / "empty-metalsrc"
    placeholder.mkdir(parents=True, exist_ok=True)
    sha = None
    ls = subprocess.run(["git", "ls-remote", src.repo, src.ref, f"{src.ref}^{{}}"],
                        capture_output=True, text=True)
    if ls.returncode == 0 and ls.stdout.strip():
        # prefer the peeled tag line when present
        lines = [ln.split("\t") for ln in ls.stdout.strip().splitlines()]
        sha = lines[-1][0]
    elif re.fullmatch(r"[0-9a-f]{7,40}", src.ref):
        sha = src.ref
    if not sha:
        raise BuildError(
            f"could not resolve {src.ref!r} on {src.repo} (git ls-remote found nothing "
            "and it is not a commit sha)"
        )
    return MetalSource(
        mode="git", context=placeholder, origin=None, sha=sha, describe=None, dirty=False,
        scm_version="0.0.0.dev0", git_repo=src.repo, git_ref=sha,
    )


def resolve_git_ref(repo: str, ref: str) -> str:
    """A ref on a remote -> commit sha (used for runtime.plugin / runtime.vllm)."""
    if re.fullmatch(r"[0-9a-f]{40}", ref):
        return ref
    ls = subprocess.run(["git", "ls-remote", repo, ref, f"{ref}^{{}}"],
                        capture_output=True, text=True)
    if ls.returncode == 0 and ls.stdout.strip():
        return ls.stdout.strip().splitlines()[-1].split("\t")[0]
    if re.fullmatch(r"[0-9a-f]{7,39}", ref):
        return ref  # short sha: not resolvable via ls-remote, trust it
    raise BuildError(f"could not resolve ref {ref!r} on {repo}")


# ---------------------------------------------------------------------- staging
def stage_code(manifest: Manifest, metal: Path, dest: Path) -> List[str]:
    """Copy the source.code allowlist into dest/, refusing silently-missing paths.

    A missing path RAISES rather than shipping nothing: the silent miss is the failure
    mode where the bundle ImportErrors on a consumer long after the push looked fine.
    Returns a rendered file tree for the model card.
    """
    entries: List[str] = []
    for rel in manifest.source.code:
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
                ignore=shutil.ignore_patterns(
                    "__pycache__", "*.pyc", ".pytest_cache", "*.egg-info",
                    ".venv", "venv", "build", "generated", "logs", "*.log",
                    "*.safetensors", "*.pt", "*.pth", "*.ckpt", "*.bin",
                    # readiness/bring-up artifacts that litter real autoport dirs
                    "*.refpt", "readiness_*",
                ),
            )
        else:
            shutil.copy2(src, target)
        entries.append(rel + ("/" if src.is_dir() else ""))
    return entries


def _sha256_tree(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


@dataclass
class Staged:
    manifest: Manifest
    ctx: Path            # docker build context
    out: Path            # HF repo staging dir (manifest, README, code/, image/)
    image: str           # docker tag
    build_args: Dict[str, str] = field(default_factory=dict)
    metal: Optional[MetalSource] = None


def stage(manifest_path: Path, out_root: Optional[Path] = None) -> Staged:
    """Everything before `docker build`: resolve, pin, and lay out the context."""
    from .types import TYPES

    manifest_path = Path(manifest_path)
    m = load_manifest(manifest_path)
    mtype = TYPES[m.type]

    out = (out_root or Path.cwd() / "build") / m.name
    if out.exists():
        shutil.rmtree(out)
    ctx = out / "ctx"
    ctx.mkdir(parents=True)

    metal = resolve_metal_source(m, scratch=ctx)

    # -- pin every git ref in runtime: ------------------------------------------------
    for key in ("plugin", "vllm"):
        entry = m.runtime.get(key)
        if isinstance(entry, dict) and entry.get("repo") and entry.get("ref"):
            entry["sha"] = resolve_git_ref(entry["repo"], entry["ref"])

    # -- code/ -------------------------------------------------------------------------
    code_dir = ctx / "code"
    if metal.mode == "local":
        tree = stage_code(m, metal.origin, code_dir)
    else:
        # git mode: shallow-clone just to stage code/ (the image build re-clones)
        tmp = ctx / "metal-for-code"
        subprocess.run(["git", "clone", "--filter=blob:none", metal.git_repo, str(tmp)],
                       check=True)
        subprocess.run(["git", "-C", str(tmp), "checkout", metal.sha], check=True)
        tree = stage_code(m, tmp, code_dir)
        shutil.rmtree(tmp)

    # -- lock --------------------------------------------------------------------------
    lock_name = m.runtime.get("lock")
    if lock_name:
        lock_src = manifest_path.parent / lock_name
        if not lock_src.exists():
            raise BuildError(f"runtime.lock names {lock_src}, which does not exist")
        shutil.copy2(lock_src, ctx / "requirements.lock")

    # -- generated scripts ---------------------------------------------------------------
    header = "#!/bin/bash\nset -euxo pipefail\n"
    (ctx / "install_engine.sh").write_text(header + "\n".join(mtype.install_lines(m)) + "\n")
    (ctx / "verify.sh").write_text(header + "\n".join(mtype.verify_lines(m)) + "\n")
    pkg_docker = Path(__file__).parent / "docker"
    shutil.copy2(pkg_docker / "Dockerfile", ctx / "Dockerfile")
    shutil.copy2(pkg_docker / "entrypoint.sh", ctx / "entrypoint.sh")

    # -- built: block + image tag -------------------------------------------------------
    image = f"tt-model/{m.name}:{(metal.sha or 'dev')[:9]}"
    built: Dict[str, object] = {
        "image": image,
        "tt_model_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tt_metal": {
            "sha": metal.sha, "describe": metal.describe, "dirty": metal.dirty,
            "scm_version": metal.scm_version, "mode": metal.mode,
        },
        "code_sha256": _sha256_tree(code_dir),
    }
    for key in ("plugin", "vllm"):
        entry = m.runtime.get(key)
        if isinstance(entry, dict) and entry.get("sha"):
            built[key] = {"repo": entry["repo"], "sha": entry["sha"]}
    m.built = built

    # -- docker build args ----------------------------------------------------------------
    ext = m.runtime.get("extension")
    profiles = ",".join(m.profile_names())
    build_args = {
        "BASE_IMAGE": BASE_IMAGE_DEFAULT.format(ubuntu=m.source.ubuntu),
        "UBUNTU_VERSION": m.source.ubuntu,
        "PYTHON_VERSION": m.source.python,
        "METAL_MODE": metal.mode,
        "SCM_VERSION": metal.scm_version,
        "EXTRA_MODELS_DIR": f"/opt/tt-metal/{ext}/extra_models" if ext else "",
        # suppress builtin registration only when the model ships its own extension
        "TT_VLLM_BUILTIN_MODELS": "0" if ext else "",
        "TT_MODEL_TYPE": m.type,
        "MODEL_NAME": m.name,
        "MODEL_REPO": m.repo,
        "MODEL_WEIGHTS": m.weights,
        "MODEL_ARCH": m.arch,
        "MODEL_PROFILES": profiles,
    }
    build_args["METAL_DESCRIBE"] = metal.describe or "v0.0.0"
    if metal.mode == "git":
        build_args["METAL_GIT_REPO"] = metal.git_repo or ""
        build_args["METAL_GIT_REF"] = metal.git_ref or ""

    return Staged(manifest=m, ctx=ctx, out=out, image=image,
                  build_args=build_args, metal=metal)


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

    The child gets its own process group (a terminal Ctrl-C otherwise goes straight to
    it and there is nothing left to intercept). On a TTY, the first SIGINT prints a
    warning card — elapsed time, current stage, what a cancel costs — and the work
    CONTINUES; only a second SIGINT within `window_s` cancels. SIGTERM, or SIGINT with
    no TTY (nobody there to press twice), cancels immediately.
    """

    def __init__(self, describe: str, window_s: float = 10.0,
                 on_cancel_note: str = "", tty: Optional[bool] = None):
        self.describe = describe
        self.window_s = window_s
        self.on_cancel_note = on_cancel_note
        self.tty = sys.stdin.isatty() if tty is None else tty
        self.started = time.monotonic()
        self.armed_until = 0.0
        self.cancelled = False
        self.current_stage = ""
        self._proc: Optional[subprocess.Popen] = None
        self._old = {}

    # -- lifecycle ---------------------------------------------------------------
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
        return f"{s // 3600}h {s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m {s % 60:02d}s"

    def note_line(self, line: str) -> None:
        """Feed output lines through so the warning card can name the current stage."""
        m = _STAGE_RE.match(line)
        if m:
            self.current_stage = m.group("stage")

    # -- signals ---------------------------------------------------------------------
    def _handle(self, sig, frame):
        now = time.monotonic()
        if sig == signal.SIGINT and self.tty and now >= self.armed_until and not self.cancelled:
            self.armed_until = now + self.window_s
            stage = f"   ·   current stage: {self.current_stage}" if self.current_stage else ""
            sys.stderr.write(
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
def run_build(staged: Staged, quiet: bool = False) -> None:
    """docker build, streamed live + teed to a log, under the interrupt guard."""
    log_path = build_log_path(staged.manifest.name)
    argv = build_argv(staged)

    print(f"building {staged.image}")
    print("watch from another terminal:")
    print(f"  tail -f {log_path}\n")

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
                if not quiet:
                    sys.stdout.write(line)
                    sys.stdout.flush()
        finally:
            code = proc.wait()

        if guard.cancelled:
            raise KeyboardInterrupt(
                f"build cancelled after {guard.elapsed()}\n"
                f"    kept      BuildKit/ccache/CPM caches — a re-run resumes there\n"
                f"    log       {log_path}"
            )
        if code != 0:
            raise BuildError(
                f"docker build failed (exit {code}); full log: {log_path}"
            )


def freeze_from_image(image: str) -> str:
    """The exact python env inside a built image, as a requirements lock."""
    code = (
        "import importlib.metadata as md\n"
        "for d in sorted(md.distributions(), key=lambda d: (d.metadata['Name'] or '').lower()):\n"
        "    print(f\"{d.metadata['Name']}=={d.version}\")\n"
    )
    r = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "/opt/tt-venv/bin/python",
         image, "-c", code],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise BuildError(f"could not freeze the image env: {r.stderr[-800:]}")
    return r.stdout


def finalize(staged: Staged, code_tree: Optional[List[str]] = None) -> Path:
    """After a successful build: freeze, export the OCI layout, write the repo dir."""
    from . import MANIFEST_NAME, hub, oci

    out = staged.out
    m = staged.manifest

    # requirements.lock: from the image when this build resolved live; pass through
    # unchanged when the build installed from an existing lock.
    lock = staged.ctx / "requirements.lock"
    lock_text = lock.read_text() if lock.exists() else freeze_from_image(staged.image)
    (out / "requirements.lock").write_text(lock_text)
    m.runtime["lock"] = "requirements.lock"

    # code/ moves from the ctx into the repo dir — the SAME directory the image was
    # built from, so what a reviewer reads on HF is byte-identical to what runs.
    shutil.move(str(staged.ctx / "code"), str(out / "code"))

    print("exporting image to OCI layout (this is most of the repo's size) ...")
    oci.save(staged.image, out / "image")

    tree = code_tree or sorted(
        str(p.relative_to(out / "code")) + ("/" if p.is_dir() else "")
        for p in (out / "code").rglob("*") if p.is_dir() or p.suffix == ".py"
    )[:200]
    (out / "README.md").write_text(hub.render_model_card(m, tree))
    (out / MANIFEST_NAME).write_text(m.to_yaml())

    shutil.rmtree(staged.ctx, ignore_errors=True)
    return out
