# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Validate the surrounding toolchain — and only ever *warn*, never install.

tt-model is the front door, but it is not a package installer for the platform: it
expects the serving stack (tt-metal and the vLLM fork/plugin) to already be present on the
system and merely checks it is *adequate*, warning (with the required version) when it is
not. This keeps tt-model's dependency surface tiny and never mutates the user's environment.
"""

from __future__ import annotations

import importlib.metadata as md
import importlib.util
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

from . import metal

# The default serving stack is tt-metal plus the vLLM fork/plugin. tt-lang and tt-api are
# leftovers from an earlier serving prototype and are not part of the vLLM path, so they are
# not checked here (checking them only produced spurious "missing dependency" warnings).
LOCK = {
    "tt-metal": "0.72.0",
}

# Distribution names each component may be installed under (first match wins).
_VLLM_DISTS = ("vllm",)


@dataclass
class ComponentReport:
    name: str
    found: bool
    version: Optional[str]
    required: str
    adequate: bool
    message: str


@dataclass
class ToolchainReport:
    components: List[ComponentReport]

    @property
    def ok(self) -> bool:
        return all(c.adequate for c in self.components)

    @property
    def problems(self) -> List[ComponentReport]:
        return [c for c in self.components if not c.adequate]


def _parse_version(s: Optional[str]) -> Optional[Tuple[int, ...]]:
    """Extract the leading dotted-numeric version from a string.

    Tolerates a ``v`` prefix and git-describe suffixes ("0.72.0-5-gabc" -> (0,72,0)).
    Returns None when there is no leading numeric component (e.g. a bare git sha).
    """
    if not s:
        return None
    s = s.strip().lstrip("vV")
    # Drop git-describe / build / prerelease suffixes ("0.72.0-5-gabc", "1.1.3+light").
    for sep in ("+", "-"):
        s = s.split(sep, 1)[0]
    nums: List[int] = []
    for part in s.split("."):
        if part.isdigit():
            nums.append(int(part))
        else:
            break
    return tuple(nums) if nums else None


def _meets(version: Optional[str], minimum: str) -> Optional[bool]:
    """True/False if ``version`` >= ``minimum``; None if ``version`` is unparseable."""
    v = _parse_version(version)
    if v is None:
        return None
    return v >= _parse_version(minimum)


# A ``git describe --tags`` tail: "-<N commits>-g<sha>", optionally "-dirty". This is the only
# decoration on a real tt-metal/ttnn version string that PEP 440 can't parse; everything else
# packaging handles natively (rc/pre/post/dev/local), so we strip ONLY this and let Version()
# see the rest verbatim — never truncating the patch level.
_GIT_DESCRIBE_TAIL = re.compile(r"-\d+-g[0-9a-f]+(?:-dirty)?$", re.IGNORECASE)


def _coerce_version(installed: str):
    """Parse ``installed`` into a ``packaging.Version``, or ``None`` if it isn't a version.

    Strips a leading ``v`` and a ``git describe`` tail, then hands the string straight to
    ``Version`` so PEP 440 semantics (rc/pre/post/dev, patch level) are preserved. A bare git
    sha or other non-version returns ``None`` (caller treats that as "assume OK").
    """
    from packaging.version import InvalidVersion, Version

    s = installed.strip().lstrip("vV")
    for candidate in (s, _GIT_DESCRIBE_TAIL.sub("", s)):
        try:
            return Version(candidate)
        except InvalidVersion:
            continue
    return None


def version_satisfies(installed: Optional[str], spec: str) -> Optional[bool]:
    """Whether ``installed`` satisfies a PEP 440 range ``spec`` (e.g. ``">=0.72,<0.76"``).

    Returns ``None`` — "assume OK, can't tell" — when the installed string is missing or not a
    real version (a bare git sha from ``git describe``), mirroring ``_meets``' ``None``
    semantics so an unpinnable dev checkout is never falsely reported out-of-range. A malformed
    ``spec`` also yields ``None`` rather than raising, so a bad manifest can't hard-crash a
    resolve. Uses ``packaging`` for correct range semantics — upper bounds AND pre-releases
    (``0.72.3rc1`` compares as a real point release, not truncated to ``0.72``).

    A falsy ``spec`` (``None``/empty) also returns ``None``: callers routinely resolve a
    manifest that declares only some of the ttnn/vLLM/plugin ranges, and an undeclared range
    is "no constraint", not a crash. (``SpecifierSet(None)`` raises ``TypeError``, which this
    guard prevents.)
    """
    if not installed or not spec:
        return None
    from packaging.specifiers import InvalidSpecifier, SpecifierSet

    parsed = _coerce_version(installed)
    if parsed is None:
        return None
    try:
        return SpecifierSet(spec, prereleases=True).contains(parsed)
    except InvalidSpecifier:
        return None


def _dist_version(dists: Tuple[str, ...]) -> Optional[str]:
    for dist in dists:
        try:
            return md.version(dist)
        except md.PackageNotFoundError:
            continue
        except Exception:  # noqa: BLE001 — never let metadata lookup break a check
            continue
    return None


def _spec_present(*module_names: str) -> bool:
    for m in module_names:
        try:
            if importlib.util.find_spec(m) is not None:
                return True
        except (ImportError, ValueError):
            continue
    return False


#: Emitted as JSON by :func:`_probe_interpreter`. Detection only — ``find_spec`` never
#: executes the module, so probing cannot open a device or pull in a multi-second import.
_PROBE_SRC = """
import importlib.util as u, json, sys
def present(name):
    try:
        return u.find_spec(name) is not None
    except (ImportError, ValueError):
        return False
def version(dists):
    import importlib.metadata as md
    for d in dists:
        try:
            return md.version(d)
        except Exception:
            continue
    return None
mods, dists = json.loads(sys.argv[1])
json.dump({"present": {m: present(m) for m in mods}, "version": version(dists)}, sys.stdout)
"""

#: A probe is detection in a fresh interpreter; 30s matches ``runtime.vllm_available``.
_PROBE_TIMEOUT = 30


def _probe_interpreter(python: str, modules: Tuple[str, ...],
                       dists: Tuple[str, ...]) -> Optional[dict]:
    """Ask *python* which of *modules* it can import, and the version of *dists*.

    Returns ``None`` when the interpreter cannot be probed at all (missing, broken, timed
    out). ``None`` is distinct from "probed, found nothing": the caller must not turn an
    unreachable interpreter into a confident "not installed".
    """
    try:
        out = subprocess.run([python, "-c", _PROBE_SRC, json.dumps([list(modules), list(dists)])],
                             capture_output=True, text=True, timeout=_PROBE_TIMEOUT, check=True)
        data = json.loads(out.stdout)
        return data if isinstance(data, dict) else None
    except (subprocess.SubprocessError, OSError, json.JSONDecodeError):
        return None


def _component(name: str, *, found: bool, version: Optional[str]) -> ComponentReport:
    required = LOCK[name]
    if not found:
        return ComponentReport(name, False, None, required, False,
                               f"not found — install {name} >= {required}")
    verdict = _meets(version, required)
    if verdict is None:
        return ComponentReport(name, True, version, required, True,
                               f"version {version!r} not comparable; assuming OK (require >= {required})")
    if verdict:
        return ComponentReport(name, True, version, required, True, "ok")
    return ComponentReport(name, True, version, required, False,
                           f"version {version} is older than required {required} — upgrade")


def _vllm_component(python: Optional[str] = None) -> ComponentReport:
    """Presence check for the Tenstorrent vLLM serving stack (fork + plugin).

    The fork tracks the ``dev`` branch, so this is presence-based rather than a strict
    version floor: both ``vllm`` and the ``vllm_tt_plugin`` package must be importable.

    With *python* set, the check runs in THAT interpreter. tt-model is routinely installed
    in a venv of its own (pipx, or a manager venv) while the build that actually serves
    lives in another — so an in-process ``find_spec`` reports the plugin missing for an
    instance that can serve perfectly well. ``serve`` already guards its hard error this
    way (:func:`runtime.vllm_available`); the warning path has to agree with it, or the two
    contradict each other on the same host.
    """
    required = "tenstorrent/vllm@dev + plugin"

    if python is not None and python != sys.executable:
        probed = _probe_interpreter(python, ("vllm", "vllm_tt_plugin"), _VLLM_DISTS)
        if probed is None:
            # Unreachable interpreter. Reported as inadequate (serving through it would
            # fail) but worded as what we actually know, not as "vLLM is not installed".
            return ComponentReport("vllm", False, None, required, False,
                                   f"could not probe the instance interpreter ({python})")
        present, version = probed.get("present") or {}, probed.get("version")
        found_vllm, found_plugin = bool(present.get("vllm")), bool(present.get("vllm_tt_plugin"))
        where = f" in {python}"
    else:
        found_vllm, found_plugin = _spec_present("vllm"), _spec_present("vllm_tt_plugin")
        version, where = _dist_version(_VLLM_DISTS), ""

    if not found_vllm:
        return ComponentReport(
            "vllm", False, None, required, False,
            f"not found{where} — install the Tenstorrent vLLM fork + plugin "
            "(see scripts/install.sh)",
        )
    if not found_plugin:
        return ComponentReport(
            "vllm", True, version, required, False,
            f"vllm present but the TT plugin (vllm_tt_plugin) is not importable{where} — "
            "pip install -e plugins/vllm-tt-plugin",
        )
    return ComponentReport("vllm", True, version, required, True, "ok (vllm + TT plugin present)")


@dataclass
class EnvConflict:
    """One unsatisfied requirement between two installed distributions."""
    package: str
    requirement: str
    installed: Optional[str]

    @property
    def message(self) -> str:
        have = f"have {self.installed}" if self.installed else "not installed"
        return f"{self.package} requires {self.requirement} ({have})"


def check_environment(python: Optional[str] = None) -> List[EnvConflict]:
    """Report installed distributions whose requirements are mutually unsatisfiable.

    Version checks alone said "toolchain adequate" on an environment pip had just called
    broken: installing ttnn (which pins numpy<2) into a venv holding the vLLM fork (whose
    opencv-python-headless wants numpy>=2) satisfies every individual check while leaving
    the env internally inconsistent. Three imports resolving is not the same as an
    environment that works, so ask pip.

    Advisory by design: a conflict may involve a package the TT serving path never imports,
    so callers should surface these and let the user judge, not block on them. Returns []
    when pip cannot be run — an unavailable check is not a passing check, but it is also not
    a conflict we can name.
    """
    exe = python or sys.executable
    try:
        proc = subprocess.run([exe, "-m", "pip", "check"], capture_output=True, text=True,
                              timeout=60)
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode == 0:
        return []
    conflicts: List[EnvConflict] = []
    # Real `pip check` output, captured from the venv in the bug report:
    #   opencv-python-headless 5.0.0.93 has requirement numpy>=2; python_version >= "3.9",
    #   but you have numpy 1.26.4.
    # "has requirement" and "requires" are both in the wild depending on pip version, and a
    # requirement may itself contain commas ("numpy>=1.24,<2"), so the split is on the
    # literal ", but you have" rather than on any comma.
    pattern = re.compile(
        r"^(?P<pkg>\S+)\s+\S+\s+(?:has requirement|requires)\s+(?P<req>.+?),\s+"
        r"but you have\s+(?P<have>.+?)\.?$"
    )
    for line in proc.stdout.splitlines():
        m = pattern.match(line.strip())
        if m:
            have = m.group("have").strip()
            conflicts.append(EnvConflict(
                package=m.group("pkg"),
                requirement=m.group("req").strip(),
                installed=None if have.endswith("not installed") else have,
            ))
    return conflicts


def check_toolchain(python: Optional[str] = None) -> ToolchainReport:
    """Inspect the local tt-metal + vLLM serving stack. Never imports the heavy modules and
    never installs anything — detection via metadata, find_spec, and the tt-metal version
    resolver already used by ``compare``. tt-lang and tt-api (leftovers from an earlier
    serving prototype) are not part of the vLLM path and are not checked.

    *python* is the interpreter to inspect — pass the selected tt-metal instance's, so the
    report describes the environment that will actually serve rather than the one running
    tt-model. Omitted (the default) keeps the in-process behaviour.
    """
    tt_metal_version = metal.resolve_version()
    ttnn_found = bool(tt_metal_version)
    if not ttnn_found:
        if python is not None and python != sys.executable:
            probed = _probe_interpreter(python, ("ttnn",), ())
            ttnn_found = bool((probed or {}).get("present", {}).get("ttnn"))
        else:
            ttnn_found = _spec_present("ttnn")
    return ToolchainReport(components=[
        _component("tt-metal", found=ttnn_found, version=tt_metal_version),
        _vllm_component(python),
    ])


__all__ = ["LOCK", "ComponentReport", "EnvConflict", "ToolchainReport",
           "check_environment", "check_toolchain"]
