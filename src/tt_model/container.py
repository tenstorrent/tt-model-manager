# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Compose and drive ``docker`` invocations for serving.

Everything here is either pure argv composition (unit-testable, asserted via
``serve --print``) or a thin subprocess wrapper. The docker flags are not folklore —
each one is load-bearing:

- ``--device /dev/tenstorrent`` — the boards.
- ``--ipc host`` — shared memory with the host.
- the hugepages mount, **verbatim**: umd matches ``/proc/mounts`` against
  ``^(nodev|hugetlbfs) (/dev/hugepages-1G) hugetlbfs …$``; binding a subdirectory or a
  different dst silently fails that regex and device-open fails minutes later.
- the HF cache mounted **read-write** at HF_HOME: model classes call
  ``snapshot_download`` at load time, and ``--trust-remote-code`` writes to
  HF_MODULES_CACHE. Weights are the only thing that ever touches the host.
- a per-model host dir at TT_METAL_CACHE: the JIT kernel/trace cache. Persisting it is
  the difference between a ~6-minute and a ~15-minute boot.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from .manifest import Manifest, ServeProfile

LABEL = "org.tenstorrent.tt-model"

# SIGTERM lets the server close the mesh on its way out; a SIGKILL does not, and leaves
# the devices needing a reset before anything can open them again. Boot alone is ~10
# minutes, so the grace period is generous.
STOP_TIMEOUT_S = 120


def image_tag(m: Manifest) -> str:
    """One image covers every profile, so the tag encodes the build, not the hardware."""
    built = m.built or {}
    if built.get("image"):
        return built["image"]
    sha = (built.get("tt_metal") or {}).get("sha") if isinstance(built.get("tt_metal"), dict) else None
    return f"tt-model/{m.name}:{(sha or 'dev')[:9]}"


def container_name(m: Manifest, profile: ServeProfile) -> str:
    return f"tt-model-{m.name}-{profile.name}"


def hf_home() -> Path:
    return Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))


def model_cache_dir(m: Manifest) -> Path:
    return Path.home() / ".cache" / "tt-model" / m.name / "cache"


def compose_run(
    m: Manifest,
    profile: ServeProfile,
    argv: List[str],
    env: Dict[str, str],
    *,
    detach: bool = True,
) -> List[str]:
    """The full ``docker run`` argv for one serve profile."""
    port = profile.port or 8000
    cmd = ["docker", "run"]
    if detach:
        cmd += ["--detach"]
    cmd += [
        "--name", container_name(m, profile),
        "--label", f"{LABEL}={m.name}",
        "--label", f"{LABEL}.repo={m.repo}",
        "--label", f"{LABEL}.profile={profile.name}",
        "--device", "/dev/tenstorrent",
        "--ipc", "host",
        "--mount", "type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G",
        "--volume", f"{hf_home()}:/hf",
        "--env", "HF_HOME=/hf",
        "--volume", f"{model_cache_dir(m)}:/cache",
        "--env", "TT_METAL_CACHE=/cache",
        "--publish", f"{port}:{port}",
    ]
    if os.environ.get("HF_TOKEN"):
        cmd += ["--env", "HF_TOKEN"]  # value comes from the caller's env, not the argv
    for k, v in env.items():
        cmd += ["--env", f"{k}={v}"]
    cmd += [image_tag(m)]
    cmd += argv
    return cmd


# ------------------------------------------------------------------ thin wrappers
def _run(argv: List[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(argv, **kw)


def running(name_filter: Optional[str] = None) -> List[Dict[str, str]]:
    """tt-model containers currently running (or exited but present)."""
    fmt = "{{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"
    argv = ["docker", "ps", "--all", "--filter", f"label={LABEL}", "--format", fmt]
    out = _run(argv, capture_output=True, text=True).stdout
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
    argv = ["docker", "images", "--filter", f"label={LABEL}", "--format", fmt]
    out = _run(argv, capture_output=True, text=True).stdout
    rows = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            rows.append({"image": parts[0], "size": parts[1],
                         "created": parts[2] if len(parts) > 2 else ""})
    return rows


def stop(name: str, image: Optional[str] = None) -> bool:
    """SIGTERM-first stop. Returns True when the shutdown was clean.

    ``docker stop`` sends SIGTERM and escalates to SIGKILL after the timeout. A kill
    means the mesh was not closed — eth cores are dirty and the next boot fails — so in
    that case the mesh is reset with ``tt-smi -r all`` in a throwaway container from the
    same image (no host tt-smi needed).
    """
    inspect = _run(["docker", "inspect", "--format", "{{.State.Running}}", name],
                   capture_output=True, text=True)
    was_running = inspect.returncode == 0 and inspect.stdout.strip() == "true"

    _run(["docker", "stop", "--timeout", str(STOP_TIMEOUT_S), name],
         capture_output=True, text=True)

    clean = True
    if was_running:
        code = _run(["docker", "inspect", "--format", "{{.State.ExitCode}}", name],
                    capture_output=True, text=True).stdout.strip()
        # 137 = 128+SIGKILL: docker's grace period expired and it hard-killed the server
        clean = code not in ("137", "")
    _run(["docker", "rm", name], capture_output=True, text=True)

    if not clean and image:
        reset_mesh(image)
    return clean


def reset_mesh(image: str) -> bool:
    """Run ``tt-smi -r all`` inside a throwaway container from the given image."""
    r = _run([
        "docker", "run", "--rm",
        "--device", "/dev/tenstorrent",
        "--mount", "type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G",
        "--entrypoint", "tt-smi", image, "-r", "all",
    ], capture_output=True, text=True)
    return r.returncode == 0


def logs(name: str, follow: bool = False) -> int:
    argv = ["docker", "logs"]
    if follow:
        argv.append("--follow")
    argv.append(name)
    return _run(argv).returncode


def wait_ready(name: str, probe: str, timeout_s: int = 1800) -> bool:
    """Follow the container's logs until the ready line appears (serve --follow)."""
    import time

    deadline = time.monotonic() + timeout_s
    proc = subprocess.Popen(["docker", "logs", "--follow", name],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            print(line, end="")
            if probe in line:
                return True
            if time.monotonic() > deadline:
                return False
        return False
    finally:
        proc.terminate()
