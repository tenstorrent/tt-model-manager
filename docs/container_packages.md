# Container (v5.1) model packages

> **Status.** The v5.1 container path is merged and on `main`. The base path landed in
> [PR #37 (OCI path to serve and distribute models)](https://github.com/tenstorrent/tt-model-manager/pull/37);
> [PR #50 (`tt-dit-server` kind)](https://github.com/tenstorrent/tt-model-manager/pull/50) added the
> `tt-dit-server` kind and
> [PR #51 (image digest as identity)](https://github.com/tenstorrent/tt-model-manager/pull/51) made
> the image digest its identity. The design is additive: `container` is a new optional block on the
> existing manifest, so v5 and v6 are untouched. A few items are called out below as **planned**;
> everything else is implemented. Design rationale, the wire format, and the image layout are in
> [Design notes](#design-notes) at the end of this page.

A **container package** ships the whole platform *inside an Open Container Initiative (OCI)
image*: Ubuntu, the built TT-Metalium™ tree, vLLM, the Tenstorrent vLLM plugin, and the model's own
code, all pinned, all inside. A consumer needs only **Docker and a Tenstorrent PCIe card**: no
TT-Metalium, no vLLM, no venv, no matching Python or OS. Everything needed to serve is inside the
image.

## Where it sits among the three packaging paths

| | ships | consumer must have | who assembles the platform |
|---|---|---|---|
| **v5** self-contained | the author's `ttnn`/vLLM/plugin **wheels** + a `tt-metal-community` tree | a Tenstorrent card + firmware | the consumer, at `pull` (wheels installed into the bundle's own venv) |
| **v6** thin | a pinned pip spec plus a small models wheel (no engine wheel) | a Tenstorrent card + firmware + SFPI | the consumer, at `pull` (a pinned venv built inside the bundle from pip pins) |
| **v5.1** container | an **OCI image** with OS + TT-Metalium + vLLM + plugin + code baked in | **Docker** + a Tenstorrent card | the **author**, once, at `package` (build time) |

v5 and v6 assemble the platform on the consumer's host, so the host's glibc and architecture have
to cooperate. v5.1 moves the assembly to the author: the image is built once, and nothing about
the consumer's host can matter because there is no host interpreter or host platform in the
picture. The compatibility check (`compare()`) covers the two facts an image cannot carry: the
**arch** its binaries were built for (fatal) and whether the host has **enough chips** for the
chosen profile (forceable).

## Weights stay a pointer

The image does not bake in weights. `weights:` is a Hugging Face (HF) repo id (optionally pinned
to a revision and filtered by patterns). At `pull` time it is downloaded into the **consumer's
own HF cache** under the consumer's own token. A small image serves a large model (a roughly 2 GB
image can front a 57 GB checkout) and the same image works for anyone regardless of which weights
revision they are entitled to.

## Provenance: authored `main`, published `<sha>`

The manifest an author writes may say `ref: main`; the manifest that gets **published** pins
`ref: 9350f5ae…`. `package` resolves every floating ref (the TT-Metalium tree, the plugin, the
weights revision) to a commit before staging and records it in the wire manifest's `built:` block.
This guards against a plugin that moved under a validated model.

## Authoring: one YAML file, one command

The entire authoring interface is a single YAML file, `tt-model.yaml`, committed next to
the model in the `tt-metal` fork so the serving recipe is reviewed in the same PR as the model
code (recommended, but it can live anywhere). There are no per-field flags:

```bash
tt-model package --container tt-model.yaml          # builds the image, stages the repo dir
tt-model serve   build/my-model/tt_kernel_manifest.json   # prove it locally
tt-model push    build/my-model --private           # publish to the Hub
```

To also list it in the community catalog, push it public and opt in. This is the same
`tt-model-catalog` tag the v5/v6 path writes, added after the upload so it lands on the
model card `package` generated:

```bash
tt-model push build/my-model --public --publish     # upload + list
tt-model publish   you/my-model                     # list one pushed earlier
tt-model unpublish you/my-model                     # delist (repo untouched)
```

### Generating the YAML with the `tt-model-yaml` skill

You do not have to write `tt-model.yaml` from scratch. This repo ships a Claude Code skill that
reads a model directory in a `tt-metal` checkout, works out its import closure and serve recipe,
and asks you for what the directory cannot tell it (hardware label, mesh, context length,
tool-call parsers):

```bash
/tt-model-yaml models/demos/blackhole/my_model
```

It is scoped to v5.1 **only**. v5 "fat" and v6 "thin" bundles are authored with CLI flags
and have no manifest file, so the skill declines those rather than emitting a YAML they
cannot read. Source: [`.claude/skills/tt-model-yaml/`](../.claude/skills/tt-model-yaml/).

Validation is front-loaded. Everything knowable without hardware is checked at *load* time
(`ContainerManifest.validate_semantics` / `validate_sources_exist`): arch, kind, the mesh vs
hardware chip-count cross-check, required launch fields, that `runtime.extra_models_dir` is
covered by the `source.code` allowlist, and that every `source.code` path exists. The
alternative is finding out ten minutes into a build.

### The YAML schema

Fields, from `src/tt_kernel/container_manifest.py`:

| field | meaning |
|---|---|
| `schema` | `"5.1"`, a string so YAML cannot turn it into a float. Same number the published manifest carries as `schema_version`. |
| `repo` | the namespaced HF id `push` publishes to, for example `you/my-model`. |
| `name` | the model name; also the default image repository. |
| `weights` | an HF id **or** a `{repo, revision, allow_patterns, ignore_patterns}` mapping. A pointer; the weights are not baked in. Pin a revision so a consumer gets the weights you validated rather than whatever the default branch points at when they pull. |
| `kind` | the serving stack and launch command: `vllm-plugin` (default), `vllm-fork`, or `tt-dit-server`. See below. |
| `arch` | `blackhole` or `wormhole_b0`. Fixed by the build; every profile shares it. |
| `source.tt_metal` | a local checkout path (default, hermetic: packages exactly the tree you validated, uncommitted work included) or `{repo, ref}` to clone in CI. |
| `source.code` | an **allowlist** (min one entry) of paths, relative to the `tt-metal` tree, that are **exactly** what ships. Staged to `code/`, uploaded to HF as browsable files, and `COPY`'d into the image as the *only* `models` package. Under-listing fails the image's own build-time import check, on the author's machine. |
| `source.extra_code` | code from outside the `tt-metal` tree. See [Code that does not live in `tt-metal`](#code-that-does-not-live-in-tt-metal). |
| `source.ubuntu` | base image version, for example `"22.04"`. |
| `source.python` | interpreter, for example `"3.12"`. Independent of ubuntu; `uv` provides it. |
| `runtime` | kind-specific (see below): the vLLM form, the plugin form, `extra_models_dir`, an optional `lock`, extra `wheels`, resolution `overrides`. |
| `serve` | the launch defaults every profile inherits (`ServeSettings`): `port`, `mesh_device`, `hardware`, `max_num_seqs`, `block_size`, `max_model_len`, `capabilities` (tool and reasoning parsers), `additional_config`, `env`, `args`. |
| `serve_profiles` | optional list of named `ServeProfile`s; each deep-merges over `serve:`. Omit entirely for a single-configuration model. |
| `default_profile` | required when more than one profile is declared. The author decides the default rather than leaving it to the consumer's environment. |
| `verify` | build-time Python assertions run **inside the finished image**, on top of the launcher's own import checks. |
| `image` | where the built image is published: `registry: hf` (default) or a real registry namespace. |
| `card.quickstart` | optional Markdown appended to the generated model card. |

`max_num_seqs` and `block_size` are **required after profile merge**. The Tenstorrent backend
rejects vLLM's own defaults. `mesh_device` must be a value from the plugin's closed
`MESH_DEVICE` table (or a literal `"(rows, cols)"` tuple); anything else is refused at load
rather than raising about 10 minutes into a boot. The `hardware` to `mesh_device` cross-check is
the only relationship `tt-model` asserts between the two: a label like `p300x2` means 2 dual-chip
boards, so 4 chips, and the mesh must open exactly that many chips. Both are stated because one
cannot be derived from the other in general (`P150x4` and `P300x2` are both a `(1, 4)` mesh).

`hardware` names **boards** and not the system they sit in: `<board>[xN]`, board one of `p100`,
`p150`, `n150`, `e150`, `p300`, `n300` (a board-revision letter like `p300c` is fine). A label
outside that grammar is refused at load, because a label `tt-model` cannot read is one whose
`device_count` it would have to invent and whose cross-check it would skip, leaving a 4-chip model
publishing `device_count: 1` with nothing said. System names such as TT-QuietBox® 2, TT-LoudBox™,
and Tenstorrent Galaxy™ belong in `mesh_device` (as the plugin's `MESH_DEVICE` table spells them,
for example `QB2`, `T3K`, `TG`): a TT-QuietBox 2 is `hardware: p300x2`, a TT-LoudBox is
`hardware: n300x4`.

### `kind`: the serving stack

- **`vllm-plugin`** (default): stock `vllm==X.Y.Z` from PyPI (built from sdist in the image
  with `VLLM_TARGET_DEVICE=empty`) plus the standalone `tenstorrent/vllm-tt-plugin`.
  Upstream vLLM's platform-plugin API means a hardware backend no longer needs a fork, and
  this is the direction of Tenstorrent's vLLM support. Launched with `vllm serve`. Its
  `runtime:` block wants a `vllm` source (`{version}`, `{wheel}`, or `{path}`), a `plugin`
  source (`{path}`, the default, or `{repo, ref}` or `{version}`), and `extra_models_dir` (the
  directory the plugin scans for per-model `vllm_metadata.json` files, which must be covered by
  `source.code`). This is the only kind that has run on hardware for an LLM.
- **`vllm-fork`**: the `tenstorrent/vllm` fork with the plugin in-tree, both installed
  editable, launched through TT-Metalium's readiness runner. This is the pre-plugin
  arrangement. Its `runtime:` wants `vllm: {repo, ref}` (the fork) and `model_dir` instead of a
  plugin block. Argv-tested but not yet run on hardware.
- **`tt-dit-server`**: a diffusion model behind its own HTTP app, launched with
  `python -m uvicorn`. A diffusion transformer has no tokens, no key-value (KV) cache and no
  continuous batching, so vLLM has nothing to do. This kind installs a small HTTP stack (fastapi,
  uvicorn, pydantic, pillow) instead of an engine, and the serving code is the model's own: the
  Asynchronous Server Gateway Interface (ASGI) app TT-Metalium ships under
  `models/tt_dit/server/<model>`. Its `runtime:` wants `app` (`"module.path:attribute"`, which
  `source.code` must ship) and optionally `packages`, `lock`, and `mesh_shape_env`. Because there
  is no engine, `max_num_seqs` and `block_size` are **not** required for this kind; only
  `hardware` and `mesh_device` are. Run on hardware: FLUX.2-dev on a TT-QuietBox 2 (2× p300c,
  4 chips).

  These servers read the resolved mesh **shape** (`"2x2"`) from an environment variable that
  each one names for itself. `mesh_shape_env` says which. It defaults to `FLUX2_MESH_SHAPE`,
  the model this kind was built for, so a *second* diffusion model should set its own. The
  value is always derived from `mesh_device`, so the SKU and the shape cannot drift. Do not
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
  extra_code:                         # optional; code from OUTSIDE the tt-metal tree
    - root: {repo: https://github.com/org/my-model, ref: v1.2.0}
      paths: [my_model_pkg]
  ubuntu: "22.04"
  python: "3.12"

runtime:
  vllm: {version: "0.24.0"}                 # or {wheel: ...} / {path: ...}
  plugin: {path: /path/to/vllm-tt-plugin}   # or {repo, ref} / {version}
  extra_models_dir: models/autoports/my_model/vllm_bundle
  lock: requirements.lock                   # optional but strongly recommended

serve:
  port: 8000                                # bare-docker default; `tt-model serve` uses 20000+
  block_size: 64                            # required; the Tenstorrent backend rejects vLLM's default
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

serve_profiles:                             # OPTIONAL; omit for a single config
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
single synthesized profile named `default`, so a one-config model does not have to learn what a
profile is. One image serves *all* of a model's profiles. Kernels JIT-compile against
whatever mesh is opened at launch, so a device target (`p150x2` vs `p150x4`) and a deployment
shape (latency vs capacity) are both launch arguments rather than separate builds.

The full annotated template is at
[`examples/container-example.yaml`](../examples/container-example.yaml).

### Code that does not live in `tt-metal`

`source.code` is relative to `source.tt_metal`. That is right for a model whose code *is*
a `tt-metal` file and wrong for one that is not. `kind: tt-dit-server` made the second
case real, because a diffusion server's ASGI app need not be a `tt-metal` module at all.

`source.extra_code` ships from other trees. Each entry takes the same two forms as
`tt_metal` itself, a local checkout (hermetic: exactly the tree you validated) or a
`{repo, ref}` to clone (reproducible from a sha, CI-friendly), and the same allowlist
rule, relative to *its own* root:

```yaml
source:
  tt_metal: {repo: https://github.com/tenstorrent/tt-metal, ref: v0.78.0}
  code: [models/common]
  extra_code:
    - root: {repo: https://github.com/tenstorrent/tt-animatediff, ref: v0.10.0}
      paths: [animatediff_ttnn]

runtime:
  kind: tt-dit-server
  app: animatediff_ttnn.server.app:app   # covered by extra_code above
```

Everything staged lands in the **same** `code/` tree and is COPY'd to the same place in
the image, so a path here and a path in `code` are indistinguishable downstream; only the
root differs. That is why `runtime.app` resolves either way, and why the allowlist checks
(`extra_models_dir`, `model_dir`, `app`) consult both.

Prefer the `{repo, ref}` form for anything you publish. A package whose code can only be
staged from one person's working copy is not reproducible by the consumer reading it,
which is the same reason `tt_metal` accepts a GitSource.

## Consuming and serving

On any host with Docker and a card:

```bash
tt-model serve you/my-model                   # auto-pulls the image + weights, then serves
```

`serve` starts the container and then watches it boot, as a short checklist of the boot's
landmarks parsed out of the container log, without showing the log itself:

```
  ✓ host ready  docker 28.1 · /dev/tenstorrent · hugepages
  ✓ image tt-model/my-model:7c2773460298
  ✓ container started  1.4s
  ✓ engine initialised  vLLM 0.24.0
  ✓ Tenstorrent device opened  2 chips · mesh (1, 2)  4.6s
  ✓ weights loaded  32/32  1m 21s
  ✓ KV cache configured  133,120 tokens
  ⠹ warming up the model  0:42
```

A percentage appears only where the log states a real total (weight shards, per-layer KV
allocation); everything else shows its running time. When the server reports ready the list
is erased and replaced by one `✓ you/my-model ready  4m 12s` line and a card with the
endpoint, a `tt-model curl` example, and the `logs`/`stop` hints. A boot that fails leaves
the list in place, marks the step it died in, and renders a diagnosis card (the cause,
one log line of evidence, what to try) instead of dumping the log. `-v` keeps every row;
a piped run prints each row once with no escape codes. Pass `--detach` to return as soon
as the container is started (`--follow` is accepted for compatibility and has no effect).
Ctrl-C stops the watching only; the container keeps booting.

The watch gives up after **4 hours** by default. Everything after `docker run` is on that
clock: a cold JIT, a large multi-chip load, and (with `--no-weights` / `--local-only`) a
weight download inside the container on whatever link the host has. When it expires only the
watch stops; the container keeps booting and the card points at `tt-model logs -f` and
`tt-model stop`. `TT_MODEL_READY_TIMEOUT=<seconds>` changes the bound for one run
(`TT_MODEL_READY_TIMEOUT=28800 tt-model serve you/my-model` waits 8 h).

Or split it: `tt-model pull you/my-model` moves bytes only and needs **no card**, so it can
run on a build host; a later `serve` starts the container. Around them:

- `tt-model profiles you/my-model` lists the serve profiles.
- `tt-model serve you/my-model --profile <name> --port <n> --print`: `--profile` picks a
  non-default profile; `--port` is `serve`'s own option and is applied in two places (docker's
  `--publish` and the server's own `--port`); `--print` composes the full `docker run` argv
  without running it (the test surface for every flag). See the pass-through rule in
  [cli.md](cli.md#run-a-model).
- `tt-model logs you/my-model -f` follows the boot (a cold first boot JIT-compiles kernels,
  about 10 min).
- `tt-model stop you/my-model`: a clean `SIGTERM` closes the mesh. A `SIGKILL` leaves it
  dirty. `stop` then attempts a `tt-smi -r` scoped to that container's own chips (read back
  from the label `serve` set, and skipped if another container has taken one of them since),
  but that is best-effort recovery and is not guaranteed. A force-killed teardown can leave a device
  that only a host reboot restores (issue #107).
- `tt-model rm you/my-model` removes a *pulled* container package, including its HF
  snapshot. `--keep-cache` keeps the JIT and weight caches for a fast re-pull;
  `--include-weights` also deletes the weights from the HF cache (off by default, because they
  are shared and can be tens of GB to re-download).

`pull` loads the image into the local docker daemon (or `docker pull`s it from a real
registry) and records the package in the local db. **It does not fetch weights unless you
pass `--with-weights`.** The flag defaults to off, so a bare `pull` moves the image only and
the model downloads its weights at first load instead. `serve` is the opposite: it always
makes sure the weights are on the host first, whether or not the package was already
installed. So `pull` is the opt-in case and `serve` the automatic one:

| | image | weights |
|---|---|---|
| `tt-model pull org/name` | yes | **no** |
| `tt-model pull org/name --with-weights` | yes | yes |
| `tt-model serve org/name` | yes | yes |
| `tt-model serve org/name --no-weights` | yes | **no** |

**`serve` always puts the weights on the host before it starts the container.** They get
downloaded either way, since the HF cache is bind-mounted and a model fetching its own weights
writes to the same place. Doing it inside the container, though, hides a multi-hundred-GB
transfer behind the readiness probe, with no progress and no error the user ever sees. So
`serve` checks the host cache and fetches what is missing, as a step you can watch:

```
✓ weights org/Weights-7B@a1b2c3d4 on host   /home/you/.cache/huggingface/hub/...
```

It **resumes unconditionally** rather than deciding for itself whether the cache is complete.
`snapshot_download` holds the revision's real file list, so it knows exactly which files are
missing and fetches only those. Anything derived locally (index files, shard-name arithmetic) is
a guess at that list and is wrong for some layout.

The cost of always asking is small and measured: with a complete cache it is a metadata
no-op (about a second), and with the Hub unreachable `huggingface_hub` falls straight back to
the cache, so a fully cached model still serves air-gapped. A partial cache is resumed, and
says so: `weights org/Weights-7B@a1b2c3d4 - resuming a partial download`.

The one thing it refuses to do quietly:

- **It won't start a download that cannot fit.** `serve` compares the pinned revision's size
  against free space on the cache filesystem and stops before anything else runs:
  `not enough disk space for the weights org/Weights-7B: needs 297.0 GB more, 24.0 GB free`.
  This matters because a fetch that fills the disk does *not* fail loudly. It leaves a
  half-populated cache, and the next run resumes into the same wall. Byte accounting is scoped
  to the pinned revision, so an unrelated cached checkpoint of the same repo cannot be
  counted as already-present. Skipped for a spec pinned with `allow_patterns`/`ignore_patterns`,
  where a whole-repo total would over-count.

Not detected, in any version: a file that is present but truncated. `huggingface_hub` verifies
what it downloads and does not re-hash what is already on disk, so neither does this. Catching
it would mean re-reading every byte of the weights on every serve.

`--no-weights` (and `--local-only`, which forbids the network by definition) lets the model
fetch its own weights at first load, with an advisory note instead of a silent boot:

```
⚠ weights org/Weights-7B@a1b2c3d4 are not in your local HF cache; the model will download
  them inside the container at first load (slower, and no progress is shown here)
→ to fetch them first instead:  tt-model pull org/name --with-weights
→ or directly:  hf download org/Weights-7B --revision a1b2c3d4
```

That in-container download counts against `serve`'s readiness watch (4 h by default; see
above), so on a slow link either prefetch with `tt-model pull --with-weights` or raise
`TT_MODEL_READY_TIMEOUT`.

A *failed* fetch is still non-fatal: the image is loaded and the model can try for itself, so
a gate you can click through does not cost you the serve. The exception is a full disk, which
stops the run rather than leaving a partial cache behind. `serve` and `pull --with-weights`
share one code path here, so both behave identically.
Weights handling is suppressed entirely under `--print`.

`serve` also reloads the image from the staged `image/` layout if docker no longer has it,
but note this only helps a package you **built** locally. A *pulled* package keeps just
`tt_kernel_manifest.json` (`pull_container` loads the image from a temporary snapshot and
lets the multi-GB layout go).
Finally, `serve` refuses to point at the *authoring* YAML (it needs the built package), and
waits on the launcher's readiness probe before reporting the endpoint (see the boot
checklist above; `--detach` skips the wait).

### How the container runs

`container.compose_run()` builds the `docker run` argv. It is pure: its only environment inputs
are `HF_HOME`/`HF_TOKEN`, both overridable, so `--print` and tests are deterministic.

- `--device /dev/tenstorrent/<n>`: one flag per chip the profile needs, and **only** those
  chips. See [Which chips a serve gets](#which-chips-a-serve-gets) below.
- `--mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G`: **verbatim** src and dst,
  because the user-mode driver (UMD) matches that exact line in `/proc/mounts` with a regular
  expression; a subdirectory or 2M hugepages fails the match and surfaces as a device-open error.
- `--user`: everything the container writes lands in bind mounts owned by the person who
  ran it, not root. The value depends on the daemon: `<host uid>:<host gid>` for a rootful
  one, `0:0` under rootless, where the invoking user is already mapped to container root
  (`container_user()`; the mode comes from `docker_is_rootless()`, asked of the daemon).
  Rootless also makes the preflight's device and hugepage checks stricter, because a
  container's identity there does not carry the host user's supplementary groups.
- `--ipc host`, `--volume <hf>:/hf`, `--volume <cache>:/cache` (the JIT kernel and trace cache,
  so the roughly 10 min first compile is paid once), `--volume <weights>:/weight-cache` with
  `TT_DIT_CACHE_DIR` pointing at it, `--publish <port>:<port>`, and the profile's env.
  `HF_TOKEN` is passed by name only so its value does not enter an argv `--print`/`ps`
  would leak.

Both host caches live under one per-model parent, `~/.cache/tt-model/<name>/`: `cache/` for
JIT kernels, `weights/` for weights already converted to device layout. The second is what
keeps a diffusion model from reconverting on every start: FLUX.2 measured 291 s to ready
while writing it against 80 s reading it. This has a cost. It uses disk roughly equal to the
weights (105 GB for FLUX.2, on top of what the HF cache already holds), and nothing reports it,
because `tt_dit` silently reconverts when `TT_DIT_CACHE_DIR` is unset rather than failing.
`tt-model rm` removes the whole parent; `--keep-cache` keeps it.

### Which chips a serve gets

A container is scoped to exactly as many chips as its profile needs. The count comes from
the resolved profile's `hardware` (cross-checked against `mesh_device` at package time), and
the specific chips are picked from whatever is free on the host at launch:

```
tt-model serve org/model                 # auto-picks free chip(s)
tt-model serve org/model --device-id 0,1 # pins specific ones instead
```

This is more than bookkeeping. TT-Metalium/UMD takes a host-wide lock on every chip it can
*see* during cluster bring-up, not only the one it computes on, and holds it for the
container's whole life. A container handed the whole `/dev/tenstorrent` directory therefore
locks the entire board, and the next serve (even one wanting a different chip) blocks forever
on that lock with no error and no timeout. Scoping the mount is what makes several models
coexist on one board at all.

What the picker does:

- **Counts what's in use host-wide**, including containers `tt-model` did not launch. The lock is shared
  through `--ipc host` regardless of who launched the other container, so a `tt-studio` or
  `tt-inference-server` container counts too. A claim is read from a container's actual grant
  (`--device`, a bind mount, or `--privileged`, which reaches every node without listing
  any), unioned with the `org.tenstorrent.tt-model.devices` label `tt-model`'s own containers
  carry. A label can only ever over-claim; it cannot hide a held chip. A container that does not
  share the host ipc namespace claims nothing.
- **Refuses instead of hanging.** Not enough free chips is an immediate error naming what is
  busy, raised ahead of the expensive steps (the first-time auto-pull, the image
  self-heal, the weights prefetch) and re-checked immediately before `docker run`.
  (`--refresh` is the exception: the candidate manifest does not exist yet, and checking
  the installed one would reject a profile the new revision adds.)
- **Validates `--device-id` against the real inventory.** `--device-id 99` on a four-chip
  host is refused up front rather than minutes later inside docker.

The pick is advisory rather than a reservation. Nothing is held between the scan and the
container starting (about 100 ms), so two `serve` commands launched within that window could
choose the same chip; the second then hangs on the UMD lock, recoverable with `tt-model stop`. A
host-wide lock would close that window, at the cost of every docker call inside it being able to
pin a lock that blocks *every* serve on the host. That is the worse failure, so it is not taken.
Two serves of the same model and profile are excluded regardless, by docker's own container
name uniqueness.

One wrinkle worth knowing: a single chip that is physically one ASIC of a fused multi-chip
board (half a P300) reports the *board's* type to TT-Metalium, which cannot match "P300 board,
one chip visible" to any built-in preset and refuses to open a mesh at all. For a single-chip
scope, `compose_run` therefore also sets `TT_MESH_GRAPH_DESC_PATH` to TT-Metalium's own generic
1×1 descriptor (never overriding one the author set). Multi-chip profiles keep their real
fabric topology.

## Not covered yet (planned)

- Image size is unoptimised: `tools/triage` (about 75 MB) and `tt_metal/pre-compiled` (about
  187 MB) are flagged in the Dockerfile as candidates to prune once a build is green.
- `package` leaves a roughly 10 GB image per build and does not clean up older tags of the same
  model; `tt-model rm` only undoes a *pull*, so an author's images accumulate. No
  `--prune-previous` yet.
- The `vllm-plugin` and `tt-dit-server` kinds have run on hardware; `vllm-fork` is
  argv-tested.

## Design notes

The sections above are the user-facing reference. The rest of this page records why the format
is shaped the way it is: how the authored YAML becomes the published manifest, how the image
travels, what the image checks about itself at build time, and how the path was validated.

### Why assembly moved to the author

v5 rebuilds the world on the consumer's host, so the host's glibc, Python,
TT-Metalium™ and vLLM all have to cooperate. Most of `packaging.py`, `provision.py` and
`toolchain.py` exist to negotiate that. v6 trims the payload to a pinned spec but still
resolves a venv on the consumer's host. v5.1 moves the assembly to the author: the image is
built once, and nothing about the consumer's host can matter because there is no host
interpreter or host platform in the picture. `compare()` needed no container branch at all: it
already checked only the two facts an image cannot carry, the **arch** its binaries were built
for (fatal) and whether the host has **enough chips** for the chosen profile (forceable). Wheel
interpreter and platform tags are checked separately at install, on the v5 path only.

### Provenance

The other half of the idea is trust. The manifest an author writes may say `ref: main`; the
manifest that gets **published** pins `ref: 9350f5ae…`. `package` resolves every floating
ref (the TT-Metalium tree, the plugin, the weights revision) to a commit before staging and
records it in the wire manifest's `built:` block. A plugin that moved under a validated model is
the exact failure this guards against.

### From authored YAML to the wire manifest

The authored YAML is *not* the published document. `ContainerManifest.to_wire()`
renders a `schema_version: "5.1"` `Manifest`, the same `tt_kernel_manifest.json` filename
every command already resolves, and *that* JSON lands on the Hub. Two reasons for the split:

- `Manifest.from_json` gates on `SUPPORTED_SCHEMAS` (`{"5", "5.1", "6"}`), which is what
  makes an older `tt-model` refuse a newer package loudly instead of half-reading it. Adding
  the v5.1 path required adding `"5.1"` to that set.
- YAML is a better *authoring* surface (comments, block scalars, no commas); JSON is a better
  *wire* format. Authors get the former, the Hub gets the latter.

The wire `Manifest` carries a `container: ContainerSpec` block. It is present if and only if this
is a container package, and absent for v5 and v6 bundles (pre-v5 schemas are refused), which is
what keeps those paths byte-for-byte unaffected. Inside it: an `ImageRef` (registry, repository,
tag, digest), the `kind`, the opaque `runtime` dict, the merged-once `serve` defaults and
`serve_profiles`, a `code_dir` pointing at the browsable copy, the `verify` list, and the pinned
`built:` provenance block. `to_wire()` also fills the top-level `device_count` from the default
profile's hardware label, and sets `build_key = None` / `kernel_count = 0` because kernels
JIT inside the container into a mounted cache dir rather than shipping precompiled.

### The image on the wire: an exploded OCI layout

A container package carries its image as an **exploded Open Container Initiative (OCI) layout**
under `image/` in the Hugging Face (HF) repo (`image/blobs/sha256/…`) rather than one giant
tarball. Layers are content-addressed files, so HF/xet dedupes the blobs shared between models
built on the same TT-Metalium commit, a re-push uploads only what changed, and an interrupted
multi-GB transfer resumes per blob. In the PR #37 validation run, a 10.2 GB image published as
2.1 GB in 27 blobs. `oci.py` does the conversion with `skopeo` when present, falling back to
`docker save`/`docker load` (Docker 25 or newer emits and accepts the OCI layout). It has
no `tt-model` concepts in it, so it is testable against a fake `docker` on PATH
with no daemon.

Setting `image.registry` to a real registry (for example `ghcr.io/tenstorrent`) instead makes the
repo carry only a pointer, so plain `docker pull` works for consumers who do not install
`tt-model` (k8s, CI).

### Inside the image

`docker/Dockerfile` is a builder plus runtime multi-stage build, kind-agnostic: everything
stack-specific arrives through two generated scripts (`install_engine.sh`, `verify.sh`) so
the Dockerfile does not change per kind. The builder clones or copies the `tt-metal` tree
(excluding `models/`, so the staged `code/` allowlist becomes the *only* `models` package),
runs `build_metal.sh` under `ccache`/CPM cache mounts (the 1.5 to 2.5 h cold C++ build), does
an editable `ttnn` install, then runs the generated engine install. The runtime stage starts
from a bare `ubuntu:<version>`, installs only the built artifacts' actual link closure, sets
up a writable `$HOME` and cache layout that survives an arbitrary `--user` uid, and gives the
image a default `CMD` (`serve-default.sh`) so `docker run <image>` with the right host flags
serves correctly. `entrypoint.sh` prepares the runtime dirs and `exec`s the composed serve
command as PID 1 so `docker stop`'s SIGTERM reaches the server for a clean mesh close.

### Two things that make it trustworthy

**The image verifies itself at build time.** `verify.sh` runs inside the finished image,
after the user switch: imports resolve, torch is `+cpu` and matches TT-Metalium's own pin, and
every model registered through `EXTRA_MODELS_DIR` is *actually resolved* rather than only checked
for a metadata file. Registration is lazy, so a shim that computes its root by directory
depth can resolve on the author's machine and fail in the image; this catches it on the
author's machine instead.

**Under-shipping is an error rather than a skip.** `source.code` promises exactly what ships, so a
missing path fails and anything the ignore list drops is reported. Runtime *data* is the easy
thing to forget (one model reads a precision config whose absence silently falls back to
in-code defaults), which is what the manifest's own `verify:` assertions are for.

### Why weights are resumed unconditionally

`serve` asks `snapshot_download` to resume every time rather than deciding for itself whether
the cache is complete. `snapshot_download` holds the revision's real file list, so it knows
exactly which files are missing. Anything derived locally (index files, shard-name arithmetic) is
a guess at that list and is wrong for some layout. An earlier version of this check guessed, and
a half-downloaded repo it did not recognize read as complete, which is the failure the check
exists to prevent. The behavior is described from the user's side in
[Consuming and serving](#consuming-and-serving).

### Validated on hardware

The PR #37 validation brought up `qwen3-coder-30B-A3B` on a TT-QuietBox® 2 (p300x2, 4 chips)
end to end: `package`, `serve`, coherent completions at about 50 tokens/s single-stream, clean
SIGTERM `stop`, `docker image rm`, reload from the OCI layout, `serve` again with
**byte-identical output**, then `push` (272 files on the Hub). Eight defects found in that run
are now checks that fail early:

- a uid-1000 collision on Ubuntu 24.04
- uninitialised submodules
- an ignore pattern eating a real package
- an unresolvable lazy registration
- bind-mount ownership
- a uid with no passwd entry
- `$HOME` cache perms
- a tqdm stand-in that would have silently downloaded nothing

### Testing

The offline suite needs no hardware, daemon, or network, and a large share of it covers this
path. `oci.py` runs against a fake `docker` on PATH; argv composition is pure so
`serve --print` exercises every flag; the manifest's front-loaded validation is checkable
on a machine that does not have the author's `tt-metal` tree. Run the whole suite with
`pytest -q`.
