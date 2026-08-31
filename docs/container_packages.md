# Container (v5.1) model packages

> **Status.** The v5.1 container path is **merged and on `main`**
> ([PR #37](https://github.com/tenstorrent/tt-model-manager/pull/37), with
> [#50](https://github.com/tenstorrent/tt-model-manager/pull/50) adding the `tt-dit-server`
> kind and [#51](https://github.com/tenstorrent/tt-model-manager/pull/51) making the image
> digest its identity). The design is additive: `container` is a new optional block on the
> existing manifest, so v5/v6 are untouched. A few items are called out below as
> **planned**; everything else is implemented.

A **container package** ships the whole platform *inside an OCI image*: Ubuntu, the built
tt-metal tree, vLLM, the Tenstorrent vLLM plugin, and the model's own code — all pinned,
all inside. A consumer needs only **Docker and a Tenstorrent card**: no tt-metal, no vLLM,
no venv, no matching Python or OS. The image is the wall.

## Where it sits among the three packaging paths

| | ships | consumer must have | who assembles the platform |
|---|---|---|---|
| **v5** self-contained | the author's `ttnn`/vLLM/plugin **wheels** + a `tt-metal-community` tree | a TT card + firmware | the consumer, at `pull` (wheels installed into the bundle's own venv) |
| **v6** thin | a pinned pip spec (no wheels) | a TT card + a matching platform | the consumer, at `pull` (a pinned venv resolved from an index) |
| **v5.1** container | an **OCI image** with OS + tt-metal + vLLM + plugin + code baked in | **Docker** + a TT card | the **author**, once, at `package` (build time) |

v5 rebuilds the world on the consumer's host, which is why the host's glibc, Python,
tt-metal and vLLM all have to cooperate — most of `packaging.py`, `provision.py` and
`toolchain.py` exist to negotiate that. v6 trims the payload to a pinned spec but still
resolves a venv on the consumer's box. **v5.1 moves the assembly to the author**: the image
is built once, and nothing about the consumer's host can matter because there is no host
interpreter or host platform in the picture. `compare()` needed no container branch at
all: it already checked only the two facts an image cannot carry — the **arch** its binaries
were built for (fatal) and whether the box has **enough chips** for the chosen profile
(forceable). Wheel interpreter/platform tags are checked separately at install, on the v5
path only.

## Weights stay a pointer

The image never bakes in weights. `weights:` is an HF repo id (optionally pinned to a
revision and filtered by patterns); at `pull` time it is downloaded into the **consumer's
own HF cache** under the consumer's own token. A small image serves a large model — a ~2 GB
image can front a 57 GB checkout — and the same image works for anyone regardless of which
weights revision they are entitled to.

## Provenance: authored `main`, published `<sha>`

The other half of the idea is trust. The manifest an author writes may say `ref: main`; the
manifest that gets **published** pins `ref: 9350f5ae…`. `package` resolves every floating
ref (the tt-metal tree, the plugin, the weights revision) to a commit before staging and
records it in the wire manifest's `built:` block. A plugin that moved under a
validated model is the exact failure this guards against.

## Authoring: one YAML file, one command

The entire authoring interface is a single YAML file — `tt-model.yaml`, committed next to
the model in the tt-metal fork so the serving recipe is reviewed in the same PR as the model
code (recommended, but it can live anywhere). There are no per-field flags:

```bash
tt-model package --container tt-model.yaml          # builds the image, stages the repo dir
tt-model serve   build/my-model/tt_kernel_manifest.json --follow   # prove it locally
tt-model push    build/my-model --private           # publish to the Hub
```

To also list it in the community catalog, push it public and opt in — the same
`tt-model-catalog` tag the v5/v6 path writes, added after the upload so it lands on the
model card `package` generated:

```bash
tt-model push build/my-model --public --publish     # upload + list
tt-model publish   you/my-model                     # list one pushed earlier
tt-model unpublish you/my-model                     # delist (repo untouched)
```

### Writing the YAML with Claude Code

You do not have to write `tt-model.yaml` from scratch. This repo ships a skill that reads a
model directory in a tt-metal checkout, works out its import closure and serve recipe, and
interviews you for what the directory cannot tell you (hardware label, mesh, context length,
tool-call parsers):

```bash
/tt-model-yaml models/demos/blackhole/my_model
```

It is scoped to v5.1 **only** — v5 "fat" and v6 "thin" bundles are authored with CLI flags
and have no manifest file, so the skill declines those rather than emitting a YAML they
cannot read. Source: [`.claude/skills/tt-model-yaml/`](../.claude/skills/tt-model-yaml/).

Validation is front-loaded: everything knowable without hardware — arch, kind, the mesh vs
hardware chip-count cross-check, required launch fields, that `runtime.extra_models_dir` is
covered by the `source.code` allowlist, that every `source.code` path actually exists — is
checked at *load* time (`ContainerManifest.validate_semantics` /
`validate_sources_exist`), because the alternative is finding out ten minutes into a build.

### The YAML schema

Fields, from `src/tt_kernel/container_manifest.py`:

| field | meaning |
|---|---|
| `schema` | `"5.1"` — a string so YAML cannot turn it into a float. Same number the published manifest carries as `schema_version`. |
| `repo` | the namespaced HF id `push` publishes to, e.g. `you/my-model`. |
| `name` | the model name; also the default image repository. |
| `weights` | an HF id **or** a `{repo, revision, allow_patterns, ignore_patterns}` mapping. A pointer — never baked in. Pin a revision so a consumer gets the weights you validated, not whatever the default branch points at today. |
| `kind` | the serving stack + launch command: `vllm-plugin` (default), `vllm-fork`, or `tt-dit-server`. See below. |
| `arch` | `blackhole` or `wormhole_b0` — fixed by the build; every profile shares it. |
| `source.tt_metal` | a local checkout path (default, hermetic — packages exactly the tree you validated, uncommitted work included) or `{repo, ref}` to clone in CI. |
| `source.code` | an **allowlist** (min one entry) of paths, relative to the tt-metal tree, that are **exactly** what ships. Staged to `code/`, uploaded to HF as browsable files, and `COPY`'d into the image as the *only* `models` package. Under-listing fails the image's own build-time import check, on the author's machine. |
| `source.ubuntu` | base image version, e.g. `"22.04"`. |
| `source.python` | interpreter, e.g. `"3.12"` — independent of ubuntu; `uv` provides it. |
| `runtime` | kind-specific (see below): the vLLM form, the plugin form, `extra_models_dir`, an optional `lock`, extra `wheels`, resolution `overrides`. |
| `serve` | the launch defaults every profile inherits (`ServeSettings`): `port`, `mesh_device`, `hardware`, `max_num_seqs`, `block_size`, `max_model_len`, `capabilities` (tool/reasoning parsers), `additional_config`, `env`, `args`. |
| `serve_profiles` | optional list of named `ServeProfile`s; each deep-merges over `serve:`. Omit entirely for a single-configuration model. |
| `default_profile` | required when more than one profile is declared — the author decides the default, not the consumer's luck. |
| `verify` | build-time Python assertions run **inside the finished image**, on top of the launcher's own import checks. |
| `image` | where the built image is published: `registry: hf` (default) or a real registry namespace. |
| `card.quickstart` | optional Markdown appended to the generated model card. |

`max_num_seqs` and `block_size` are **required after profile merge** — the TT backend
rejects vLLM's own defaults. `mesh_device` must be a value from the plugin's closed
`MESH_DEVICE` table (or a literal `"(rows, cols)"` tuple); anything else is refused at load
rather than raising ~10 minutes into a boot. The `hardware`↔`mesh_device` cross-check is the
only relationship tt-model asserts between the two: a label like `p300x2` means 2 dual-chip
boards → 4 chips, and the mesh must open exactly that many chips. Both are stated because one
cannot be derived from the other in general (`P150x4` and `P300x2` are both a `(1, 4)` mesh).

`hardware` names **boards**, never the box they sit in — `<board>[xN]`, board one of `p100`,
`p150`, `n150`, `e150`, `p300`, `n300` (a board-revision letter like `p300c` is fine). A label
outside that grammar is refused at load, because a label tt-model cannot read is one whose
`device_count` it would have to invent and whose cross-check it would skip — a 4-chip model
publishing `device_count: 1` with nothing said. Box names (`QB2`, `T3K`, `TG`) belong in
`mesh_device`: a QB2 is `hardware: p300x2`, a T3000 is `hardware: n300x4`.

### `kind`: the serving stack

- **`vllm-plugin`** (default) — stock `vllm==X.Y.Z` from PyPI (built from sdist in the image
  with `VLLM_TARGET_DEVICE=empty`) plus the standalone `tenstorrent/vllm-tt-plugin`.
  Upstream vLLM's platform-plugin API means a hardware backend no longer needs a fork; this
  is where Tenstorrent is heading. Launched with `vllm serve`. Its `runtime:` block wants a
  `vllm` source (`{version}`, `{wheel}`, or `{path}`), a `plugin` source (`{path}` — the
  default — or `{repo, ref}` or `{version}`), and `extra_models_dir` (the directory the
  plugin scans for per-model `vllm_metadata.json` files, which must be covered by
  `source.code`). This is the only kind that has run on hardware.
- **`vllm-fork`** — the `tenstorrent/vllm` fork with the plugin in-tree, both installed
  editable, launched through tt-metal's readiness runner. The older arrangement, and what
  this repo's own `install`/`provision` set up. Its `runtime:` wants `vllm: {repo, ref}` (the
  fork) and `model_dir` instead of a plugin block. Argv-tested but not yet run on hardware.
- **`tt-dit-server`** — a diffusion model behind its own HTTP app, launched with
  `python -m uvicorn`. A diffusion transformer has no tokens, no KV cache and no continuous
  batching, so vLLM has nothing to do: this kind installs a small HTTP stack (fastapi /
  uvicorn / pydantic / pillow) instead of an engine, and the serving code is the model's
  own — the ASGI app tt-metal ships under `models/tt_dit/server/<model>`. Its `runtime:`
  wants `app` (`"module.path:attribute"`, which `source.code` must ship) and optionally
  `packages`, `lock`, and `mesh_shape_env`. Because there is no engine, `max_num_seqs` and
  `block_size` are **not** required for this kind — only `hardware` and `mesh_device` are.
  Run on hardware: FLUX.2-dev on a QB2 (2× p300c, 4 chips).

  These servers read the resolved mesh **shape** (`"2x2"`) from an environment variable that
  each one names for itself. `mesh_shape_env` says which; it defaults to `FLUX2_MESH_SHAPE`,
  the model this kind was built for, so a *second* diffusion model should set its own. The
  value is always derived from `mesh_device`, so the SKU and the shape cannot drift — never
  hand-write the shape into `serve.env`.

### Example

```yaml
schema: "5.1"
repo: you/my-model
name: my-model
weights: org/Weights-7B          # a pointer; pin a revision to freeze it
kind: vllm-plugin
arch: blackhole

source:
  tt_metal: /path/to/your/tt-metal    # or {repo, ref} to clone in CI
  code:                               # EXACTLY what ships
    - models/common
    - models/autoports/my_model       # must cover runtime.extra_models_dir
  ubuntu: "22.04"
  python: "3.12"

runtime:
  vllm: {version: "0.24.0"}                 # or {wheel: ...} / {path: ...}
  plugin: {path: /path/to/vllm-tt-plugin}   # or {repo, ref} / {version}
  extra_models_dir: models/autoports/my_model/vllm_bundle
  lock: requirements.lock                   # optional but strongly recommended

serve:
  port: 8000
  block_size: 64                            # required — TT backend rejects vLLM's default
  capabilities:
    tool_parser: hermes
    reasoning_parser: deepseek_r1
  additional_config:
    tt:
      sample_on_device_mode: all
      trace_region_size: 50331648
      fabric_config: FABRIC_1D_RING
  env:
    ARCH_NAME: blackhole
  args: [--trust-remote-code]

serve_profiles:                             # OPTIONAL — omit for a single config
  - name: p150x2
    description: One interactive user at full speed.
    hardware: p150x2
    mesh_device: P150x2
    max_num_seqs: 8
    max_model_len: 65536
  - name: p150x4
    hardware: p150x4
    mesh_device: P150x4
    max_num_seqs: 32
    max_model_len: 131072
default_profile: p150x4

verify:
  - "import models.autoports.my_model.tt as m; assert m"
```

A flat manifest (no `serve_profiles:`) is not a special case: `serve:` alone becomes a
single synthesized profile named `default`, so a one-config model never has to learn what a
profile is. One image serves *all* of a model's profiles — kernels JIT-compile against
whatever mesh is opened at launch, so a device target (`p150x2` vs `p150x4`) and a deployment
shape (latency vs capacity) are both just launch arguments, not separate builds.

The full annotated template is at
[`examples/container-example.yaml`](../examples/container-example.yaml).

## From authored YAML to the wire manifest

The authored YAML is deliberately *not* the published document. `ContainerManifest.to_wire()`
renders a `schema_version: "5.1"` `Manifest` — the same `tt_kernel_manifest.json` filename
every command already resolves — and *that* JSON lands on the Hub. Two reasons for the split:

- `Manifest.from_json` gates on `SUPPORTED_SCHEMAS` (`{"5", "5.1", "6"}`), which is what
  makes an older tt-model refuse a newer package loudly instead of half-reading it. Adding
  the v5.1 path required adding `"5.1"` to that set.
- YAML is a better *authoring* surface (comments, block scalars, no commas); JSON is a better
  *wire* format. Authors get the former, the Hub gets the latter.

The wire `Manifest` carries a `container: ContainerSpec` block — present iff this is a
container package, absent for every v3/v4/v5/v6 bundle, which is what keeps those paths
byte-for-byte unaffected. Inside it: an `ImageRef` (registry, repository, tag, digest), the
`kind`, the opaque `runtime` dict, the merged-once `serve` defaults and `serve_profiles`, a
`code_dir` pointing at the browsable copy, the `verify` list, and the pinned `built:`
provenance block. `to_wire()` also fills the top-level `device_count` from the default
profile's hardware label, and sets `build_key = None` / `kernel_count = 0` because kernels
JIT inside the container into a mounted cache dir rather than shipping precompiled.

## The image on the wire: an exploded OCI layout

A container package carries its image as an **exploded OCI layout** under `image/` in the HF
repo (`image/blobs/sha256/…`) rather than one giant tarball. Layers are content-addressed
files, so HF/xet dedupes the blobs shared between models built on the same tt-metal commit, a
re-push uploads only what changed, and an interrupted multi-GB transfer resumes per blob. The
PR reports a 10.2 GB image publishing as 2.1 GB in 27 blobs. `oci.py` does the conversion
with `skopeo` when present, falling back to `docker save`/`docker load` (Docker ≥ 25 emits and
accepts the OCI layout) — and is deliberately free of any tt-model concepts so it is testable
against a fake `docker` on PATH with no daemon.

Setting `image.registry` to a real registry (e.g. `ghcr.io/tenstorrent`) instead makes the
repo carry only a pointer, so plain `docker pull` works for consumers who never install
tt-model (k8s, CI).

## Consuming and serving

On any box with Docker and a card:

```bash
tt-model serve you/my-model --follow          # auto-pulls the image + weights, then serves
```

Or split it — `tt-model pull you/my-model` moves bytes only and needs **no card**, so it can
run on a build host; a later `serve` starts the container. Around them:

- `tt-model profiles you/my-model` — list the serve profiles.
- `tt-model serve --profile <name> --port <n> --print you/my-model` — `--profile` picks a
  non-default profile; `--port` must come *before* the target because it moves in two places
  (docker's `--publish` and the server's own `--port`); `--print` composes the full `docker
  run` argv without running it (the test surface for every flag).
- `tt-model logs you/my-model -f` — follow the boot (a cold first boot JIT-compiles kernels,
  ~10 min).
- `tt-model stop you/my-model` — a clean `SIGTERM` closes the mesh; a `SIGKILL` would leave
  the devices needing `tt-smi -r`.
- `tt-model rm you/my-model` — removes a *pulled* container package, including its HF
  snapshot. `--keep-cache` keeps the JIT/weight caches for a fast re-pull;
  `--include-weights` also deletes the weights from the HF cache (off by default — they
  are shared and can be tens of GB to re-download).

`pull` loads the image into the local docker daemon (or `docker pull`s it from a real
registry) and records the package in the local db. **It does not fetch weights unless you
pass `--with-weights`** — the flag defaults to off, so a bare `pull` moves the image only and
the model downloads its weights at first load instead. `serve`'s *auto*-pull (the one that
fires when nothing is installed yet) does fetch them, so the two entry points differ:

| | image | weights |
|---|---|---|
| `tt-model pull org/name` | yes | **no** |
| `tt-model pull org/name --with-weights` | yes | yes |
| `tt-model serve org/name` (nothing installed) | yes | yes |

If weights can't be fetched the image still loads and the model fetches them at first boot.

`serve` also reloads the image from the staged `image/` layout if docker no longer has it —
but note this only helps a package you **built** locally. A *pulled* package keeps just
`tt_kernel_manifest.json` (`pull_container` loads the image from a temporary snapshot and
lets the multi-GB layout go).
Finally, `serve` refuses to point at the *authoring* YAML (it needs the built package), and
— with `--follow` — waits on the launcher's readiness probe before reporting the endpoint.

### How the container runs

`container.compose_run()` builds the `docker run` argv (pure — its only environment inputs
are `HF_HOME`/`HF_TOKEN`, both overridable, so `--print` and tests are deterministic):

- `--device /dev/tenstorrent` — the boards.
- `--mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G` — **verbatim** src and dst,
  because umd regex-matches that exact line in `/proc/mounts`; a subdirectory or 2M
  hugepages fails the match and surfaces as a device-open error.
- `--user <host uid>:<host gid>` — everything the container writes lands in bind mounts owned
  by the person who ran it, not root.
- `--ipc host`, `--volume <hf>:/hf`, `--volume <cache>:/cache` (the JIT kernel/trace cache,
  so the ~10 min first compile is paid once), `--volume <weights>:/weight-cache` with
  `TT_DIT_CACHE_DIR` pointing at it, `--publish <port>:<port>`, and the profile's env.
  `HF_TOKEN` is passed by name only so its value never enters an argv `--print`/`ps`
  would leak.

Both host caches live under one per-model parent, `~/.cache/tt-model/<name>/` — `cache/` for
JIT kernels, `weights/` for weights already converted to device layout. The second is what
keeps a diffusion model from reconverting on every start: FLUX.2 measured 291 s to ready
while writing it against 80 s reading it. It is a tradeoff, not a free win — it costs disk
roughly equal to the weights (105 GB for FLUX.2, on top of what the HF cache already holds),
and nothing reports it, because `tt_dit` silently reconverts when `TT_DIT_CACHE_DIR` is unset
rather than failing. `tt-model rm` removes the whole parent; `--keep-cache` keeps it.

### Inside the image

`docker/Dockerfile` is a builder + runtime multi-stage build, kind-agnostic: everything
stack-specific arrives through two generated scripts (`install_engine.sh`, `verify.sh`) so
the Dockerfile never changes per kind. The builder clones or copies the tt-metal tree
(excluding `models/`, so the staged `code/` allowlist becomes the *only* `models` package),
runs `build_metal.sh` under `ccache`/CPM cache mounts (the ~1.5–2.5 h cold C++ build), does
an editable `ttnn` install, then runs the generated engine install. The runtime stage starts
from a bare `ubuntu:<version>`, installs only the built artifacts' actual link closure, sets
up a writable `$HOME`/cache story that survives an arbitrary `--user` uid, and gives the
image a default `CMD` (`serve-default.sh`) so `docker run <image>` with the right host flags
serves correctly. `entrypoint.sh` prepares the runtime dirs and `exec`s the composed serve
command as PID 1 so `docker stop`'s SIGTERM reaches the server for a clean mesh close.

## Two things that make it trustworthy

**The image verifies itself at build time.** `verify.sh` runs inside the finished image,
after the user switch: imports resolve, torch is `+cpu` and matches tt-metal's own pin, and
every model registered through `EXTRA_MODELS_DIR` is *actually resolved* — not just checked
for a metadata file. Registration is lazy, so a shim that computes its root by directory
depth can resolve on the author's machine and fail in the image; this catches it on the
author's machine instead.

**Under-shipping is an error, never a skip.** `source.code` promises exactly what ships, so a
missing path fails and anything the ignore list drops is reported. Runtime *data* is the easy
thing to forget — one model reads a precision config whose absence silently falls back to
in-code defaults — which is what the manifest's own `verify:` assertions are for.

## Proven on hardware

The PR brought up `qwen3-coder-30B-A3B` on a QB2 (p300x2, 4 chips) end to end: `package` →
`serve` → coherent completions at ~50 tok/s single-stream → clean SIGTERM `stop` → `docker
image rm` → reload from the OCI layout → `serve` → **byte-identical output** → `push` (272
files on the Hub). That run turned eight defects into checks that now fail early: a uid-1000
collision on Ubuntu 24.04, uninitialised submodules, an ignore pattern eating a real package,
an unresolvable lazy registration, bind-mount ownership, a uid with no passwd entry, `$HOME`
cache perms, and a tqdm stand-in that would have silently downloaded nothing.

## Testing

The suite is 581 offline tests — no hardware, no daemon, no network — and a large share of
them cover this path. `oci.py` runs against a fake `docker` on PATH; argv composition is pure
so `serve --print` exercises every flag; the manifest's front-loaded validation is checkable
on a machine that does not have the author's tt-metal tree. Run the whole suite with
`pytest -q`.

## Not covered yet (planned)

- Image size is unoptimised: `tools/triage` (~75 MB) and `tt_metal/pre-compiled` (~187 MB) are
  flagged in the Dockerfile as candidates to prune once a build is green.
- `package` leaves a ~10 GB image per build and never cleans up older tags of the same model;
  `tt-model rm` only undoes a *pull*, so an author's images accumulate. No `--prune-previous`
  yet.
- The `vllm-plugin` and `tt-dit-server` kinds have run on hardware; `vllm-fork` is
  argv-tested.
