# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""A deployment: several container packages served side by side on one host, from one file.

Each instance is its own server process, so instances never share a chip: only one process
can open a chip. The planner gives every instance its own chips (whole boards for a
multi-chip profile) and its own port, and refuses a plan that collides before anything starts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import yaml

from . import container
from .container_manifest import hardware_chip_count
from .manifest import DEFAULT_PORT, Manifest

#: Ports that other well-known services on a Tenstorrent host already use (tt-studio, telemetry).
BUSY_PORTS = frozenset({3000, 4000, 8000, 8002, 8080, 8111, 18080})


class DeployError(RuntimeError):
    """A deployment file or plan that cannot run as written."""


@dataclass
class Instance:
    name: str
    model: str
    profile: Optional[str] = None
    chips: Optional[List[int]] = None     # None = pick automatically
    port: Optional[int] = None            # None = pick automatically
    args: List[str] = field(default_factory=list)


@dataclass
class Deployment:
    name: str
    instances: List[Instance]


@dataclass
class Placement:
    instance: Instance
    manifest: Manifest
    profile: str
    chips: List[int]
    port: int

    @property
    def container(self) -> str:
        spec = self.manifest.container
        assert spec is not None
        return container.container_name(self.manifest, spec.resolve_profile(self.profile),
                                        self.instance.name)


def load(path: str) -> Deployment:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(raw, dict) or not isinstance(raw.get("instances"), list) or not raw["instances"]:
        raise DeployError(f"{path}: expected a mapping with a non-empty `instances:` list")
    known = {"name", "model", "profile", "chips", "port", "args"}
    out = []
    for i, item in enumerate(raw["instances"]):
        if not isinstance(item, dict) or not item.get("name") or not item.get("model"):
            raise DeployError(f"{path}: instance {i + 1} needs `name` and `model`")
        unknown = set(item) - known
        if unknown:
            raise DeployError(f"{path}: instance {item['name']}: unknown key(s) {', '.join(sorted(unknown))}")
        chips, port, args = item.get("chips", "auto"), item.get("port", "auto"), item.get("args") or []
        if chips != "auto" and not (isinstance(chips, list) and all(isinstance(c, int) for c in chips)):
            raise DeployError(f"{path}: instance {item['name']}: `chips` is `auto` or a list of chip ids, "
                              f"e.g. [2, 3] (got {chips!r})")
        if port != "auto" and not isinstance(port, int):
            raise DeployError(f"{path}: instance {item['name']}: `port` is `auto` or a number (got {port!r})")
        if not isinstance(args, list):   # a string would become one argument per character
            raise DeployError(f"{path}: instance {item['name']}: `args` is a list, e.g. "
                              f"[--max-model-len, \"8192\"] (got {args!r})")
        out.append(Instance(
            name=str(item["name"]), model=str(item["model"]), profile=item.get("profile"),
            chips=None if chips == "auto" else chips,
            port=None if port == "auto" else port, args=[str(a) for a in args],
        ))
    names = [i.name for i in out]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise DeployError(f"{path}: instance name(s) used twice: {', '.join(dup)}")
    return Deployment(name=str(raw.get("deployment") or Path(path).stem), instances=out)


def plan(dep: Deployment, manifests: Dict[str, Manifest], *, inventory: Sequence[int],
         claimed: Set[int], boards: Dict[int, str], port_free, free_hugepages: Optional[int],
         ports_needed=lambda manifest, profile: 1) -> List[Placement]:
    """Assign chips and ports to every instance, or raise ``DeployError`` naming the collision.

    Pure: the host facts (chips present and in use, boards, free ports and hugepages) are
    passed in, so a plan can be checked anywhere and unit-tested without a device.
    ``ports_needed`` says how many consecutive ports an instance serves on (one per model).
    """
    used: Set[int] = set(claimed)
    resolved = []
    for inst in dep.instances:
        container.check_instance_name(inst.name)
        m = manifests[inst.model]
        spec = m.container
        if spec is None:
            raise DeployError(f"{inst.name}: {inst.model} is not a container package")
        try:
            profile = spec.resolve_profile(inst.profile)
        except ValueError as e:
            raise DeployError(f"{inst.name}: {e}") from None
        count = hardware_chip_count(profile.hardware or "") or 1
        resolved.append((inst, m, profile.name, count, max(1, int(ports_needed(m, profile)))))

    chips_of: Dict[str, List[int]] = {}
    for inst, _, _, count, _ in resolved:  # explicit pins first, so auto picks route around them
        if inst.chips is None:
            continue
        if len(inst.chips) != count or len(set(inst.chips)) != count:
            raise DeployError(f"{inst.name}: chips {inst.chips} — this profile needs {count} distinct chip(s)")
        missing = sorted(set(inst.chips) - set(inventory))
        if missing:
            raise DeployError(f"{inst.name}: chip(s) {missing} are not on this host (it has {list(inventory)})")
        clash = sorted(set(inst.chips) & used)
        if clash:
            owner = next((n for n, c in chips_of.items() if set(c) & set(clash)), None)
            raise DeployError(
                f"{inst.name}: chip(s) {clash} are already taken"
                + (f" by {owner}" if owner else " by a running server")
                + " — one process per chip; models that must share chips go in one co-host package")
        chips_of[inst.name] = sorted(inst.chips)
        used |= set(inst.chips)
    for inst, _, _, count, _ in resolved:
        if inst.chips is not None:
            continue
        free = [c for c in inventory if c not in used]
        pick = container._choose_chips(free, count, boards) if len(free) >= count else None
        if not pick:
            raise DeployError(f"{inst.name}: needs {count} chip(s) on whole boards; free: {free or 'none'}")
        chips_of[inst.name] = pick
        used |= set(pick)

    total_chips = sum(len(c) for c in chips_of.values())
    if free_hugepages is not None and free_hugepages < total_chips:
        raise DeployError(f"the plan opens {total_chips} chip(s) but only {free_hugepages} free 1G "
                          "hugepage(s) remain (one per chip)")

    ports: Dict[str, int] = {}
    taken: Set[int] = set()

    def usable(first: int, span: int) -> bool:
        return all(p not in taken and p not in BUSY_PORTS and port_free(p) for p in range(first, first + span))

    for inst, *_, span in resolved:
        if inst.port is None:
            continue
        if any(p in taken for p in range(inst.port, inst.port + span)):
            raise DeployError(f"{inst.name}: port {inst.port} is given to two instances")
        if not usable(inst.port, span):
            raise DeployError(f"{inst.name}: port {inst.port}" + (f"-{inst.port + span - 1}" if span > 1 else "")
                              + " is in use on this host")
        ports[inst.name] = inst.port
        taken |= set(range(inst.port, inst.port + span))
    nxt = DEFAULT_PORT
    for inst, *_, span in resolved:
        if inst.port is not None:
            continue
        while not usable(nxt, span):
            nxt += 1
        ports[inst.name] = nxt
        taken |= set(range(nxt, nxt + span))
        nxt += span
    return [Placement(inst, m, prof, chips_of[inst.name], ports[inst.name]) for inst, m, prof, *_ in resolved]
