# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""The co-host config: which members share the mesh, how each is served, on which port."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from . import CONFIG_VERSION

KINDS = ("vllm", "app")


@dataclass
class Member:
    name: str
    kind: str
    port: int
    #: vllm: the weights id served (``vllm serve <model>``); app: informational.
    model: str = ""
    #: vllm: extra ``vllm serve`` arguments. app: unused.
    args: List[str] = field(default_factory=list)
    #: Env applied while this member loads (and set in its vLLM frontend process).
    env: Dict[str, str] = field(default_factory=dict)
    #: app: ``module:factory`` returning an ``apps.AppMember``.
    app: str = ""
    #: vllm: cap on this member's KV pool (tokens, all users), so the members' DRAM adds up.
    kv_tokens: int = 0
    #: ``module:function`` the host calls (in this member's env) before building it: an image-side
    #: hook that fits a member next to the others (e.g. a smaller-L1 program choice).
    preload: str = ""


@dataclass
class CohostConfig:
    #: MESH_DEVICE for the one mesh every member shares, e.g. ``P150x4``.
    mesh_device: str
    members: List[Member]
    #: The plugin's ``tt`` additional_config the mesh is opened with (fabric, trace region, ...).
    tt_config: Dict[str, object] = field(default_factory=dict)
    socket_dir: str = "/tmp/tt-cohost"
    status_port: Optional[int] = None

    def socket_path(self, member: str) -> str:
        return str(Path(self.socket_dir) / f"{member}.sock")


def load(path: str) -> CohostConfig:
    return loads(Path(path).read_text())


def loads(text: str) -> CohostConfig:
    raw = json.loads(text)
    version = raw.get("cohost_config_version")
    if version != CONFIG_VERSION:
        raise ValueError(f"cohost config version {version!r}; this runtime reads {CONFIG_VERSION}")
    members = [Member(**m) for m in raw["members"]]
    names = [m.name for m in members]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate member names: {names}")
    ports = [m.port for m in members] + ([raw["status_port"]] if raw.get("status_port") else [])
    if len(set(ports)) != len(ports):
        raise ValueError(f"members must have distinct ports: {ports}")
    for m in members:
        if m.kind not in KINDS:
            raise ValueError(f"member {m.name}: kind {m.kind!r} is not one of {KINDS}")
        if m.kind == "app" and ":" not in m.app:
            raise ValueError(f"member {m.name}: an app member needs app = 'module:factory'")
    if len({m.kind for m in members}) > 1:
        raise ValueError("a group is all vllm members or all app members (mixed groups are not supported yet)")
    return CohostConfig(mesh_device=raw["mesh_device"], members=members,
                        tt_config=raw.get("tt_config") or {},
                        socket_dir=raw.get("socket_dir") or "/tmp/tt-cohost",
                        status_port=raw.get("status_port"))
