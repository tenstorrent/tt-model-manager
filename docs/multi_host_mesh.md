# Multi-host (cross-box) mesh — design of record

Tracking issue: **#87**. This is the design note that issue asks for — a cross-box mesh is "a
design effort," and everything below is grounded in the launch contract as it actually exists on
Tenstorrent Blackhole hardware, not inferred from docs.

> **Status.** The hard *hardware* unknown is settled: cross-host fabric **trains and computes**.
> A `1x4` mesh spanning two boxes (node4 + node5, 2× Blackhole each, 800G card cables) was opened
> under `FABRIC_1D` and ran a real op with **PCC 0.9999** — `Fabric Initialized`, `MESH OPENED
> shape=[1,4] num_devices=4`, no handshake timeout. So this is **not** a fabric question. What
> remains is **packaging + deployment + one vLLM-build dependency** — see the critical path.

---

## 1. Why the current model can't express it

Both tt-model launch paths assume **one process, on one host**:

- v6 thin `run.sh` → `python -m vllm.entrypoints.openai.api_server` (one process).
- v5.1 container → `docker run` (one container).

PCIe does not cross the chassis: a process on node4 can only open node4's local chips. A mesh whose
chips span two boxes therefore needs **one process per host**, joined into one job — which neither
generator can produce.

## 2. The verified launch contract

A cross-box mesh is launched by tt-metal's MPI launcher **`tt-run`** (already present in the `ttnn`
wheel at `bin/tt-run`, so a v6 venv has it). Two deploy-specific artifacts drive it, both **cluster-
specific and therefore NOT bakeable into a portable bundle**:

**a) Mesh Graph Descriptor** (`.textproto`, via `TT_MESH_GRAPH_DESC_PATH`) — maps the logical mesh
to physical `(host, chip)` positions. The one that passed on node4+node5:

```
mesh_descriptors {
  name: "M0"
  arch: BLACKHOLE
  device_topology { dims: [ 1, 4 ] }   # the logical 1x4
  host_topology   { dims: [ 1, 2 ] }   # 2 hosts, each owning a 1x2 slice
  channels        { count: 4 policy: STRICT }
}
top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
```

**b) Rank binding** (`rank_bindings.yaml`) — one entry per host-rank, plus a pointer to the MGD.
`tt-run` generates this from the MGD + `--hosts`; the vLLM plugin also accepts it directly:

```
rank_bindings:
  - { rank: 0, mesh_id: 0, mesh_host_rank: 0, env_overrides: { TT_VISIBLE_DEVICES: "0,1" } }
  - { rank: 1, mesh_id: 0, mesh_host_rank: 1, env_overrides: { TT_VISIBLE_DEVICES: "0,1" } }
mesh_graph_desc_path: /abs/path/to/mgd_node45_1x4.textproto
```

**Two entry points, same fabric underneath:**

- Bare bring-up (what proved the fabric): `tt-run --mesh-graph-descriptor <mgd> --hosts node4,node5 <program>`.
- **vLLM serve** (what tt-model must drive): the plugin already has the multi-host path.
  `vllm_tt_plugin` installs `TTCoreEngineLauncher` when it sees explicit MPI launch, driven through
  `--additional-config` under the `tt` key: **`tt.rank_binding`** (path to the YAML above),
  `tt.mpi_args`, and the `nnodes` / `node_rank` parallel-config fields
  (`launcher.py: parse_tt_mpi_params`, `_requires_tt_mpi_rank_binding`; `platform.py:
  _uses_explicit_tt_mpi_launch`). It shells out to `tt_run_launch(...)` from the **rank-0 host**,
  which SSHes into the peer and launches the process there.

So the serving logic is **largely already in the plugin**. This is a packaging + deployment gap,
not a plugin gap — with one exception (the critical path below).

## 3. The critical path — one hard dependency

The plugin's multi-host path **refuses to run** unless the vLLM build exposes an engine-core
launcher hook:

> `NotImplementedError`: "TT MPI multi-host launch … needs an engine-core launcher hook
> (`ParallelConfig.engine_core_launcher_cls` and `vllm.v1.engine.utils.CoreEngineLauncher`) that
> this vLLM build does not provide."

**Nothing tt-model emits can matter until a vLLM build with that hook is pinned and a real
multi-host serve is validated end to end.** That is tracked in `tenstorrent/vllm-tt-plugin#23`, and
it is the gate for this whole effort. Implementing packaging vocabulary before it is speculative —
so the sequence below puts validation first.

## 4. What tt-model must add (packaging design)

The portable bundle can only carry the **authored intent** (the mesh is multi-host, and its shape).
The **specific hosts, MGD, and rank binding are the operator's deployment**, supplied at serve time
— they name real machines and cannot live in a repo anyone can pull.

**Manifest vocabulary (portable):**
- `mesh.hosts: int = 1` — the number of hosts the mesh spans. `> 1` marks a multi-host bundle.
- (`mesh.topology` / a `host_topology` already describe the shape; the MGD is derived from these
  plus the operator's cluster, not stored.)

**Launch generation:**
- v6 thin `run.sh`: when `mesh.hosts > 1`, extend the exact `--additional-config` emit point #86
  added — add `tt.rank_binding` alongside `tt.fabric_config`. The value is an **operator-supplied
  path** (a `TT_RANK_BINDING` env / serve flag), guarded so a multi-host bundle served without it
  fails with one clear sentence rather than silently single-hosting. The launch still runs **once on
  rank 0**; the plugin fans out.
- v5.1 container: harder — MPI must cross the container boundary (host networking + an MPI-reachable
  `sshd`, or launching both ranks from one side). The **v6 venv path is friendlier** and should go
  first; the non-container venv has no boundary for `tt-run` to punch through.

**Deployment (documented, not automated by the bundle):**
- `pull` the same bundle on **both** hosts → identical venv + weights at a **matching absolute path**
  (MPI launches the same path remotely). node4/node5 already satisfy this: same user `mando222` @
  `/home/mando222`.
- Host prerequisites `tt-run` needs and the wheel can't carry: **system OpenMPI** (`mpirun` is not in
  the wheel), **passwordless SSH** between hosts, the fabric **trained** (proven), and the
  **vLLM-build hook** from §3. This breaks v6's "nothing on the box but card+fw+SFPI" promise, so
  multi-host must be documented as a **distinct deployment mode** with its own prereqs, not the
  "just pull and run" contract single-host v6 offers.

## 5. Sequenced slices

| Slice | What | Depends on | Testable now? |
|---|---|---|---|
| **0** | Single-host multi-chip fabric emit (`fabric_config`/`trace_region`) | — | ✅ done — #86 |
| **1** | **Validate a real multi-host vLLM serve on node4+node5** (pin the vLLM build with the launcher hook; confirm correct output, not just launch) | plugin **#23** | ❌ hardware + #23 |
| **2** | Manifest vocabulary (`mesh.hosts`) + v6 `run.sh` emits `tt.rank_binding` (extends #86's block) | Slice 1 (so the emitted config is known-correct) | ✅ offline once the contract from Slice 1 is fixed |
| **3** | Deployment docs + a `serve`-time rank-binding input (peer discovery: a `--rank-binding` flag / small deploy config) | Slice 2 | ✅ |
| **4** | v5.1 container multi-host (MPI across the container boundary) | Slices 1–3 | later |

**Slice 1 is the linchpin and must come before Slice 2.** Emitting `tt.rank_binding` before a real
serve is validated risks encoding a contract that turns out subtly wrong; the fabric is proven, the
serve is not.

## 6. Open design questions
- **Peer discovery.** How does a pulled bundle learn its second box? A `serve`-time `--rank-binding
  <file>` (operator hands over the YAML + MGD) is the minimal answer; auto-generating the rank
  binding from `--hosts a,b` via `tt-run`'s own generator is the richer one.
- **Container multi-host.** Whether to solve the container boundary at all, or declare multi-host a
  v6-venv-only deployment mode.
- **Topology validity.** A 2+2 commodity-desktop `1x4` matched a committed `2x2_Mesh_flat`; whether
  arbitrary cross-box shapes are supported MGD templates is a fabric-team question, not a packaging
  one.

## References
- Proven on-box: `tt-metal/tests/scale_out/node45/` (`mgd_node45_1x4.textproto`, `test_mesh_1x4.py`),
  `tt-metal/python_env/bin/tt-run`, `tt_metal/fabric/MGD_README.md`.
- Plugin: `vllm-tt-plugin/src/vllm_tt_plugin/{platform,launcher}.py`
  (`_uses_explicit_tt_mpi_launch`, `parse_tt_mpi_params`, `_requires_tt_mpi_rank_binding`,
  `tt_run_launch`, `TTCoreEngineLauncher`).
- Blocking dependency: `tenstorrent/vllm-tt-plugin#23`. Single-host precursor: #86.
