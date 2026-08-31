# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Compose and drive ``docker`` invocations for a container (v5.1) package.

Everything here is either pure argv composition — unit-testable, and asserted through
``serve --print`` — or a thin subprocess wrapper. Nothing in this module imports ttnn,
reads a manifest file, or touches the Hub.

The docker flags are not folklore; each one is load-bearing:

- ``--device /dev/tenstorrent`` — the boards.
- ``--ipc host`` — shared memory with the host.
- the hugepages mount, **verbatim**: umd matches ``/proc/mounts`` against
  ``^(nodev|hugetlbfs) (/dev/hugepages-1G) hugetlbfs …$``, so binding a subdirectory or
  a different dst silently fails that regex and device-open fails minutes later.
- the HF cache mounted **read-write** at ``HF_HOME``: model classes call
  ``snapshot_download`` at load time, and ``--trust-remote-code`` writes to
  ``HF_MODULES_CACHE``. Weights are the only thing that ever touches the host.
- a per-model host dir at ``TT_METAL_CACHE``: the JIT kernel/trace cache. Persisting it
  across boots is the difference between a ~6-minute and a ~15-minute start.
- ``--user <host uid>:<host gid>``: everything the container writes lands in bind mounts
  the host user owns (the HF cache, the kernel cache), so it must write AS that user. A
  baked-in uid cannot work — 1000 and 1001 are both common — and mismatched ownership
  fails as ``Permission denied`` from the JIT, minutes into a boot.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .manifest import Manifest, ServeProfile

#: Docker label applied to every container and used to find ours again.
LABEL = "org.tenstorrent.tt-model"

# SIGTERM lets the server close the mesh on its way out; SIGKILL does not, and leaves the
# devices needing a reset before anything can open them again. Boot alone is ~10 minutes,
# so the grace period is deliberately generous.
STOP_TIMEOUT_S = 120

# 128 + SIGKILL(9): docker's grace period expired and it hard-killed the server.
SIGKILL_EXIT_CODE = "137"


class ContainerError(RuntimeError):
    """A docker operation that must not proceed. The message is user-facing."""


# --------------------------------------------------------------------------- preflight

#: Docker < 25 does not emit an OCI layout from `docker save`, which `oci.save` requires.
MIN_DOCKER_MAJOR = 25

#: umd matches /proc/mounts against this exact mount point. A subdirectory, a different
#: dst, or 2M hugepages all fail that match — and the failure surfaces as a device-open
#: error MINUTES into a boot, nowhere near the cause.
HUGEPAGES_MOUNT = "/dev/hugepages-1G"

TT_DEVICE = "/dev/tenstorrent"


@dataclass(frozen=True)
class Requirement:
    name: str
    ok: bool
    detail: str      # what was found
    fix: str = ""    # what to do about it, when not ok


def _docker_version() -> Optional[str]:
    if shutil.which("docker") is None:
        return None
    r = _run(["docker", "version", "--format", "{{.Server.Version}}"],
             capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None


def preflight(*, need_devices: bool, proc_mounts: Optional[Path] = None,
              dev_root: Optional[Path] = None) -> List[Requirement]:
    """Check what this host must have, before anything slow or opaque happens.

    Every one of these fails LATE and unhelpfully if left unchecked: a missing hugepages
    mount surfaces as a device-open error ten minutes into a boot, and an old Docker only
    shows up when ``oci.save`` finds no OCI layout after a multi-hour build.

    ``need_devices`` is False for operations that only move bytes (``pull``, ``push``):
    those work fine on a machine with no card attached.
    """
    out: List[Requirement] = []

    version = _docker_version()
    if version is None:
        out.append(Requirement(
            "docker", False, "not found or daemon unreachable",
            "install Docker and ensure the daemon is running; add yourself to the "
            "`docker` group (`sudo usermod -aG docker $USER`, then log out and in) so it "
            "works without sudo",
        ))
    else:
        m = re.match(r"(\d+)", version)
        major = int(m.group(1)) if m else 0
        ok = major >= MIN_DOCKER_MAJOR
        out.append(Requirement(
            "docker", ok, version,
            "" if ok else (
                f"Docker >= {MIN_DOCKER_MAJOR} is required: earlier versions do not emit "
                "an OCI layout from `docker save`, which packaging and loading rely on "
                "(installing `skopeo` is an alternative)"
            ),
        ))

    if not need_devices:
        return out

    dev = Path(dev_root) if dev_root is not None else Path(TT_DEVICE)
    out.append(Requirement(
        "tt devices", dev.exists(), str(dev) if dev.exists() else "missing",
        "" if dev.exists() else (
            f"{TT_DEVICE} does not exist — the Tenstorrent kernel driver (tt-kmd) is not "
            "loaded, or this is not a machine with a card"
        ),
    ))

    mounts = Path(proc_mounts) if proc_mounts is not None else Path("/proc/mounts")
    try:
        text = mounts.read_text()
    except OSError:
        text = ""
    mounted = any(
        len(parts) > 2 and parts[1] == HUGEPAGES_MOUNT and parts[2] == "hugetlbfs"
        for parts in (ln.split() for ln in text.splitlines())
    )
    out.append(Requirement(
        "hugepages", mounted,
        HUGEPAGES_MOUNT if mounted else f"{HUGEPAGES_MOUNT} not mounted",
        "" if mounted else (
            f"mount 1G hugepages at exactly {HUGEPAGES_MOUNT} — umd matches that path in "
            "/proc/mounts, so a different mount point silently fails and the container "
            "then dies on device open, minutes into the boot. This is normally set up by "
            "the tt-metal host provisioning (see tt-metal's installation docs)"
        ),
    ))
    return out


def preflight_failures(reqs: List[Requirement]) -> List[Requirement]:
    return [r for r in reqs if not r.ok]


def _require_container(m: Manifest):
    if m.container is None:
        raise ContainerError(
            f"{m.name} is not a container package (schema {m.schema_version}); "
            "this path serves v5.1 packages only"
        )
    return m.container


def image_ref(m: Manifest) -> str:
    """The image reference to run.

    For an HF-hosted image this is the local tag that ``docker load`` produced. For a
    package whose image lives in a real registry it is the ``docker pull`` reference, so
    the same composition works for both without the caller knowing which it got.
    """
    image = _require_container(m).image
    return image.pull_ref or image.tag


def container_name(m: Manifest, profile: ServeProfile) -> str:
    """One running container per (model, profile) — so two profiles of the same model
    are distinguishable in ``docker ps`` and stoppable independently."""
    return f"tt-model-{m.name}-{profile.name}"


def hf_home() -> Path:
    return Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))


def _safe_name(name: str) -> str:
    """Refuse a name that would escape the managed cache dir when used as a path component.

    ``model_cache_dir`` builds a path from ``manifest.name`` that ``remove_container``
    deletes with ``rmtree``. Authoring validates the name (see ``ContainerManifest``), but a
    hand-crafted *pulled* ``tt_kernel_manifest.json`` reaches ``Manifest`` directly, so guard
    here too — a ``../..`` or absolute name must never drive an rmtree outside the cache root.
    """
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", name or ""):
        raise ContainerError(
            f"unsafe container name {name!r}: refusing to derive a filesystem path from it "
            "(expected a lowercase slug). Re-publish the package with a valid name."
        )
    return name


def model_cache_dir(m: Manifest) -> Path:
    """Host-side JIT kernel cache, per model. Survives container removal on purpose."""
    return Path.home() / ".cache" / "tt-model" / _safe_name(m.name) / "cache"


def compose_run(
    m: Manifest,
    profile: ServeProfile,
    argv: List[str],
    env: Dict[str, str],
    *,
    detach: bool = True,
    hf_home_dir: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    include_hf_token: Optional[bool] = None,
) -> List[str]:
    """The full ``docker run`` argv for one serve profile.

    Pure: the only environment it reads is ``HF_HOME``/``HF_TOKEN``, and both can be
    overridden by the caller so tests never depend on the developer's shell.
    """
    _require_container(m)
    port = profile.port or 8000
    hf = Path(hf_home_dir) if hf_home_dir is not None else hf_home()
    cache = Path(cache_dir) if cache_dir is not None else model_cache_dir(m)
    if include_hf_token is None:
        include_hf_token = bool(os.environ.get("HF_TOKEN"))

    cmd = ["docker", "run"]
    if detach:
        cmd += ["--detach"]
    # Bind-mount sources must exist before `docker run`, or the daemon creates them as
    # ROOT and the container (running as the host user) cannot write them.
    cmd += [
        "--name", container_name(m, profile),
        # write as the host user: see the module docstring
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--label", f"{LABEL}={m.name}",
        "--label", f"{LABEL}.profile={profile.name}",
        "--device", "/dev/tenstorrent",
        "--ipc", "host",
        # verbatim src AND dst — umd regex-matches this line in /proc/mounts
        "--mount", "type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G",
        "--volume", f"{hf}:/hf",
        "--env", "HF_HOME=/hf",
        "--volume", f"{cache}:/cache",
        "--env", "TT_METAL_CACHE=/cache",
        "--publish", f"{port}:{port}",
    ]
    if include_hf_token:
        # name only: the value is inherited from the caller's environment rather than
        # being written into an argv that `--print` would display and `ps` would leak.
        cmd += ["--env", "HF_TOKEN"]
    for k, v in sorted(env.items()):
        cmd += ["--env", f"{k}={v}"]
    cmd += [image_ref(m)]
    cmd += argv
    return cmd


def ensure_mount_sources(m: Manifest, *, hf_home_dir: Optional[Path] = None,
                         cache_dir: Optional[Path] = None) -> None:
    """Create the bind-mount source dirs, as the host user, before ``docker run``.

    If they do not exist the docker daemon creates them itself — owned by ROOT — and the
    container, which runs as the host user, then cannot write them. The JIT fails with
    ``Permission denied`` several minutes into a boot, nowhere near the cause.

    Kept out of ``compose_run`` deliberately: that function is pure argv composition and
    is asserted through ``serve --print``, which must not touch the filesystem.
    """
    hf = Path(hf_home_dir) if hf_home_dir is not None else hf_home()
    cache = Path(cache_dir) if cache_dir is not None else model_cache_dir(m)
    for d in (hf, cache):
        d.mkdir(parents=True, exist_ok=True)


def compose_pull(m: Manifest) -> Optional[List[str]]:
    """``docker pull`` argv for a registry-hosted image, or None for an HF-hosted one
    (whose bytes arrive with the repo snapshot and are loaded from the OCI layout)."""
    ref = _require_container(m).image.pull_ref
    return ["docker", "pull", ref] if ref else None


# ------------------------------------------------------------------ thin wrappers


def _run(argv: List[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(argv, **kw)


def run_checked(argv: List[str]) -> str:
    """Run a docker command, raising ContainerError with its stderr on failure.

    docker's own messages are the useful diagnosis here (no such image, port in use,
    device busy), so they are surfaced verbatim rather than reworded.
    """
    r = _run(argv, capture_output=True, text=True)
    if r.returncode != 0:
        detail = (r.stderr or r.stdout or "").strip()
        raise ContainerError(detail or f"`{' '.join(argv[:2])}` failed (exit {r.returncode})")
    return r.stdout


def running(name_filter: Optional[str] = None) -> List[Dict[str, str]]:
    """tt-model containers present on this host (running or exited)."""
    fmt = "{{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"
    out = _run(
        ["docker", "ps", "--all", "--filter", f"label={LABEL}", "--format", fmt],
        capture_output=True, text=True,
    ).stdout
    rows = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3 and (not name_filter or name_filter in parts[0]):
            rows.append({
                "name": parts[0], "image": parts[1], "status": parts[2],
                "ports": parts[3] if len(parts) > 3 else "",
            })
    return rows


def images() -> List[Dict[str, str]]:
    """Locally loaded tt-model images."""
    fmt = "{{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}"
    out = _run(
        ["docker", "images", "--filter", f"label={LABEL}", "--format", fmt],
        capture_output=True, text=True,
    ).stdout
    rows = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            rows.append({
                "image": parts[0], "size": parts[1],
                "created": parts[2] if len(parts) > 2 else "",
            })
    return rows


def run_or_empty(argv: List[str]) -> str:
    """stdout of a docker query, or "" if it fails. For display only — never for control
    flow, where a silent empty string would hide a real problem."""
    r = _run(argv, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


def container_exists(name: str) -> bool:
    """Does a container with this exact name exist, in ANY state?

    ``docker run`` creates the container before it binds ports, so a failed start (a busy
    port, most commonly) leaves one behind in ``Created`` — holding the name and blocking
    every retry.
    """
    return _run(["docker", "container", "inspect", name],
               capture_output=True).returncode == 0


def is_running(name: str) -> bool:
    """Is it actually running? Asked via inspect rather than by matching `docker ps`
    status text, because "Created" and "Exited (1)" both mean "not running" and only the
    state field says so unambiguously."""
    r = _run(["docker", "inspect", "--format", "{{.State.Running}}", name],
             capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() == "true"


def remove(name: str, *, force: bool = False) -> bool:
    argv = ["docker", "rm"] + (["--force"] if force else []) + [name]
    return _run(argv, capture_output=True, text=True).returncode == 0


def loaded_digest(ref: str) -> Optional[str]:
    """The config digest of the image currently under ``ref``, or None if it is absent.

    ``image_present`` only answers "is something under this name". With digest tags that is
    usually enough, but a hand-tagged or hand-loaded image can sit under the right name and
    be the wrong image — so where correctness matters, compare digests.
    """
    out = run_or_empty(["docker", "image", "inspect", ref, "--format", "{{.Id}}"]).strip()
    return out or None


def remove_image(ref: str) -> bool:
    return _run(["docker", "image", "rm", ref], capture_output=True, text=True).returncode == 0


def image_present(ref: str) -> bool:
    return _run(["docker", "image", "inspect", ref], capture_output=True).returncode == 0


def stop(name: str, image: Optional[str] = None) -> bool:
    """SIGTERM-first stop. Returns True when the shutdown was clean.

    ``docker stop`` sends SIGTERM and escalates to SIGKILL after the timeout. A kill means
    the server never closed the mesh — eth cores are left dirty and the NEXT boot fails —
    so in that case the mesh is reset with ``tt-smi -r all`` in a throwaway container from
    the same image. That is why no host tt-smi is needed: the image already has one.
    """
    inspect = _run(
        ["docker", "inspect", "--format", "{{.State.Running}}", name],
        capture_output=True, text=True,
    )
    was_running = inspect.returncode == 0 and inspect.stdout.strip() == "true"

    _run(["docker", "stop", "--timeout", str(STOP_TIMEOUT_S), name],
         capture_output=True, text=True)

    clean = True
    if was_running:
        code = _run(
            ["docker", "inspect", "--format", "{{.State.ExitCode}}", name],
            capture_output=True, text=True,
        ).stdout.strip()
        clean = code not in (SIGKILL_EXIT_CODE, "")
    _run(["docker", "rm", name], capture_output=True, text=True)

    if not clean and image:
        reset_mesh(image)
    return clean


def compose_reset_mesh(image: str) -> List[str]:
    return [
        "docker", "run", "--rm",
        "--device", "/dev/tenstorrent",
        "--mount", "type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G",
        "--entrypoint", "tt-smi", image, "-r", "all",
    ]


def reset_mesh(image: str) -> bool:
    """Run ``tt-smi -r all`` inside a throwaway container from the given image."""
    return _run(compose_reset_mesh(image), capture_output=True, text=True).returncode == 0


def logs(name: str, follow: bool = False) -> int:
    argv = ["docker", "logs"]
    if follow:
        argv.append("--follow")
    argv.append(name)
    return _run(argv).returncode


@dataclass(frozen=True)
class ReadyResult:
    """Why the wait ended. ``ready`` alone cannot say WHY it failed, and "the server did
    not report ready" is a useless thing to print when the container actually crashed."""

    ready: bool
    exited: bool           # the log stream closed => the container is gone
    tail: List[str]        # last lines seen, for a failure card


def wait_ready(name: str, probe: str, timeout_s: int = 1800, echo=print) -> ReadyResult:
    """Follow the container's logs until the launcher's ready line appears.

    The generous default timeout is not padding: a cold boot JIT-compiles kernels, which
    is the ~10-minute cost the mounted TT_METAL_CACHE exists to avoid paying twice.
    """
    import queue
    import threading
    import time

    deadline = time.monotonic() + timeout_s
    proc = subprocess.Popen(
        ["docker", "logs", "--follow", name],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    assert proc.stdout is not None
    # Read the log on a thread so the deadline is honoured even while the container is
    # SILENT. The old `for line in proc.stdout` only checked the deadline after a line
    # arrived, so a booted-but-hung container (no output, no crash) blocked forever.
    lines: "queue.Queue[Optional[str]]" = queue.Queue()
    tail: List[str] = []

    def _pump() -> None:
        for line in proc.stdout:  # type: ignore[union-attr]
            lines.put(line)
        lines.put(None)  # EOF sentinel: the log stream closed => the container exited

    threading.Thread(target=_pump, daemon=True).start()
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return ReadyResult(False, False, tail[-20:])
            try:
                line = lines.get(timeout=min(remaining, 5.0))
            except queue.Empty:
                continue  # silent container: re-check the deadline and keep waiting
            if line is None:
                # EOF: the container exited before the probe appeared. The reason is in
                # what it printed on the way out, so hand that back rather than a bare
                # "not ready".
                return ReadyResult(False, True, tail[-20:])
            stripped = line.rstrip("\n")
            tail.append(stripped)
            echo(stripped)
            if probe in line:
                return ReadyResult(True, False, tail[-20:])
    finally:
        proc.terminate()
