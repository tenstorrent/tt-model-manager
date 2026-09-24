<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc. -->
# End-to-end recipe (v5.1): package, push, pull, serve

This is the copy-paste recipe for the **v5.1 container** path. Take a model you have brought up
on `tt-metal`, describe it in one `tt-model.yaml`, build it into an Open Container Initiative
(OCI) image, publish it to the Hugging Face (HF) Hub, and let anyone with Docker and a
Tenstorrent PCIe card pull and serve it. The whole platform (Ubuntu, the built TT-Metalium™
tree, vLLM, the Tenstorrent vLLM plugin, and your model code) ships inside the image.

> Scope: **v5.1 container packages only.** The author builds the platform once, at `package`
> time. A consumer needs only **Docker and a Tenstorrent card**: no TT-Metalium, no vLLM, no
> venv, no matching Python or OS. For the v6 thin path, which builds a venv on the consumer's
> host instead, see [E2E_RECIPE_V6.md](E2E_RECIPE_V6.md).

There are two roles:

- **Producer**: the host where the model already serves from a `tt-metal` checkout. It needs
  Docker and time: a cold build is 2.5 to 4 hours.
- **Consumer**: any host with Docker and a Tenstorrent card. Nothing else is required.

---

## Prerequisites

**Both roles**
- Linux **x86_64**.
- **Docker 25 or newer** (`docker version`), with the daemon running and your user in the
  `docker` group so it works without `sudo`. Earlier versions do not emit an OCI layout from
  `docker save`; installing `skopeo` is an alternative.
- `tt-model` installed. It is **not on PyPI**; install it from a clone:
  ```bash
  git clone https://github.com/tenstorrent/tt-model-manager && cd tt-model-manager
  python -m venv .venv && source .venv/bin/activate
  pip install -e .
  ```
  There is nothing else to provision on the host. No `ttnn`, TT-Metalium, or vLLM is needed
  outside the image, and none on the host is touched. See the [README](../README.md#install).
- Authenticated to HF for push and pull of private repos, either with `tt-model login` or:
  ```bash
  export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  ```

**Anyone who serves** (the consumer, and the producer when proving the build locally)
- A Tenstorrent card with firmware and driver installed (`/dev/tenstorrent/*` present).
- 1G hugepages mounted at **exactly** `/dev/hugepages-1G`. The user-mode driver matches that
  path in `/proc/mounts`; a different mount point or 2M pages fails silently and surfaces as a
  device-open error minutes into a boot. The `tt-metal` host provisioning normally sets this up.
- Enough free disk on the HF cache filesystem for the weights. `serve` checks before it
  downloads anything.

`tt-model` runs these checks itself, up front, before anything slow. `pull` and `push` only move
bytes and need no card, so they run fine on a build host.

**Producer only**
- A `tt-metal` checkout that serves your model. Uncommitted work is fine: the image ships the
  tree as-is.
- A checkout of `tenstorrent/vllm-tt-plugin` (or a pushed `{repo, ref}` / a PyPI version).
- The exact launch command the model was validated with (a `serve*.sh`, a README, or a
  bring-up note). Every flag in it has to land in the manifest.
- Optional but strongly recommended: a `requirements.lock` from the validated venv
  (`uv pip freeze --python <venv>/bin/python`), with editable installs, `file://` wheels, the
  `vllm==*+*` line, and any CUDA-only `nvidia-*` / `cuda-*` / `flashinfer-*` packages stripped.

---

## Step 0: Bring up your model on `tt-metal` (producer)

Not covered in depth here. See `tt-metal`'s `CONTRIBUTING.md` and `docs/BRINGUP.md`. Two cases:
- **Stock HF architecture** (Llama, Qwen, Mistral, and others): no code. Serve it with the
  plugin's `vllm serve` recipe for that model.
- **New architecture**: implement the novel block against ttnn ops under `models/`, and register
  it with the plugin through a `vllm_metadata.json` in a directory the plugin scans.

When `vllm serve` produces coherent text against your card, you are ready to package.

---

## Step 1: Write `tt-model.yaml` (producer)

The entire authoring interface is one YAML file. Commit it next to the model in your `tt-metal`
fork so the serving recipe is reviewed in the same PR as the model code (recommended; it can
live anywhere). There are no per-field flags.

```yaml
schema: "5.1"
repo: <your-org>/<model-name>          # the HF repo `push` publishes to
name: <model-name>
weights: unsloth/Llama-3.2-3B-Instruct  # a POINTER; pin {repo, revision} to freeze it
kind: vllm-plugin
arch: blackhole                          # or wormhole_b0; fixed by the build

source:
  tt_metal: /path/to/your/tt-metal       # or {repo, ref} to clone in CI
  code:                                  # EXACTLY what ships; an allowlist
    - models/common
    - models/tt_transformers
  ubuntu: "22.04"
  python: "3.12"

runtime:
  vllm: {version: "0.24.0"}              # or {wheel: ...} / {path: ...}
  plugin: {path: /path/to/vllm-tt-plugin}  # or {repo, ref} / {version}
  extra_models_dir: models/tt_transformers/vllm_bundle   # must be covered by source.code
  lock: requirements.lock                # optional but strongly recommended

serve:
  port: 8000
  hardware: p150                         # boards: <board>[xN]; a QuietBox 2 is p300x2
  mesh_device: P150                      # the plugin's MESH_DEVICE table
  max_num_seqs: 8                        # required; the TT backend rejects vLLM's default
  block_size: 64                         # required, same reason
  max_model_len: 65536
  capabilities:
    tool_parser: llama3_json             # omit if the model has no tool calling
  env:
    ARCH_NAME: blackhole
  args: [--trust-remote-code]

verify:
  - "import models.tt_transformers.tt.generator_vllm as m; assert m"
```

What the fields mean:

| Field | Meaning |
|---|---|
| `schema` | `"5.1"`, quoted so YAML does not turn it into a float |
| `repo` / `name` | the HF repo `push` publishes to, and the model name (also the default image repository) |
| `weights` | **pointer** to the weights. Never baked in. Pin a `revision` so consumers get what you validated |
| `kind` | `vllm-plugin` (default, the only kind run on hardware for an LLM), `vllm-fork`, or `tt-dit-server` for diffusion models |
| `arch` | the card's instruction set architecture (ISA): `blackhole` or `wormhole_b0`. The one **fatal** compatibility gate |
| `source.tt_metal` | your checkout (hermetic, uncommitted work ships) or `{repo, ref}` for CI |
| `source.code` | the **allowlist** of paths that ship. Under-listing fails the image's own build-time import check, on your machine |
| `runtime.vllm` / `runtime.plugin` | where vLLM and the plugin come from. `{path}` is the default for the plugin: your checkout goes into the image |
| `runtime.extra_models_dir` | the directory the plugin scans for per-model `vllm_metadata.json` (its children). Must be covered by `source.code` |
| `runtime.lock` | the frozen pins. Without it the build resolves live and can pull in something you never ran with |
| `serve.hardware` / `serve.mesh_device` | how many boards, and the mesh to open on them. Both are stated because one cannot be derived from the other |
| `serve.capabilities` | tool and reasoning parser **names**; `tt-model` renders the flags |
| `serve_profiles` / `default_profile` | *(optional)* several launch configs from one image. Omit for a single-configuration model |
| `verify` | Python assertions run **inside the finished image** at build time |

Rather than writing it from scratch, run the `tt-model-yaml` Claude Code skill against the model
directory. It reads the import closure and serve recipe and asks for what the directory cannot
tell it:

```bash
/tt-model-yaml models/demos/blackhole/my_model
```

Validate the manifest before you spend hours on a build. Everything knowable without hardware
is checked at load time, including that every `source.code` path exists:

```bash
python -c "
from tt_kernel.container_manifest import load_container_manifest
m = load_container_manifest('tt-model.yaml', check_sources=True)
p = m.resolve_profile()
print('VALID:', m.name, m.kind, '|', p.hardware, p.mesh_device)"
```

For every field, the three `kind`s, multi-profile manifests, and code that lives outside the
`tt-metal` tree, see **[docs/container_packages.md](container_packages.md#authoring-one-yaml-file-one-command)**.
The full annotated template is [`examples/container-example.yaml`](../examples/container-example.yaml).

---

## Step 2: Package (producer)

`tt-model package --container` resolves every floating ref to a commit, stages the code
allowlist, builds the image, verifies it from the inside, and exports it as an OCI layout into
a staging directory. It does **not** push.

```bash
tt-model package --container tt-model.yaml --out ~/tt-model-builds
```

| Flag | Meaning |
|---|---|
| `--container <yaml>` | the manifest. It is the whole interface; every other `package` flag is ignored |
| `--out <dir>` | the staging root. The package lands in `<dir>/<name>/`. Default: `./build/<name>/` |

A cold build is **2.5 to 4 hours**, most of it the C++ build of TT-Metalium. `package` prints
a `tail -f ~/.cache/tt-model/build/<name>.log` command before the wait starts, so watch it from
another terminal. On a TTY the first Ctrl-C only warns; a second within the window cancels and
keeps the build caches, so a re-run is cheap. The in-image verify stage runs **last**, after
every layer is cached, so a verify failure re-runs in minutes.

The result is the staged package directory:

```
~/tt-model-builds/<name>/
  tt_kernel_manifest.json     the published (wire) manifest, refs pinned to commits
  README.md                   the generated model card
  code/                       the source.code allowlist, browsable on the Hub
  image/                      the OCI layout: image/blobs/sha256/... content-addressed
  requirements.lock           the pins the image was built with
```

The authored YAML said `ref: main`; the wire manifest's `built:` block pins the TT-Metalium sha,
the plugin sha, and the weights revision it actually built from.

### Step 2b: Prove it locally (producer, needs a card)

`serve` and `stop` take the **built** manifest, never the authoring YAML:

```bash
tt-model serve ~/tt-model-builds/<name>/tt_kernel_manifest.json
```

Watch the boot checklist until it ends in a ready card, run the checks in
[Step 5](#step-5-verify-consumer) against it, then stop it cleanly:

```bash
tt-model stop ~/tt-model-builds/<name>/tt_kernel_manifest.json
```

Publishing a multi-gigabyte image you have not served is the expensive mistake this step exists
to prevent.

---

## Step 3: Push (producer)

`tt-model push` uploads the staged directory. The repo id comes from the manifest's `repo:`;
`--repo` overrides it. `push` needs no card.

```bash
tt-model push ~/tt-model-builds/<name>                   # private repo (the default)
tt-model push ~/tt-model-builds/<name> --public          # public, shared by link, not listed
tt-model push ~/tt-model-builds/<name> --publish         # public AND listed in the community catalog
tt-model push ~/tt-model-builds/<name> --repo other-org/other-name
```

Visibility is tri-state and **a push does not flip it implicitly**: a new repo is created
private, and an existing repo keeps its visibility unless you pass the flag, in which case the
change is announced. `--publish` implies `--public` and adds the catalog tag after the upload.
See [publishing.md](publishing.md#push-vs-public-vs-publish).

The image travels as **content-addressed blobs** under `image/`, so a re-push uploads only the
layers that changed, blobs shared with other models built on the same TT-Metalium commit are
deduplicated by the Hub, and an interrupted transfer resumes per blob. A 10 GB image published
as about 2 GB in the validation run.

To list a repo pushed earlier, or to delist one:

```bash
tt-model publish   <org>/<model-name>     # list (makes it public if needed)
tt-model unpublish <org>/<model-name>     # delist; the repo is untouched
```

---

## Step 4: Pull (consumer)

On any host with Docker, whether or not it has a card:

```bash
tt-model pull <org>/<model-name>                  # image only
tt-model pull <org>/<model-name> --with-weights   # image + pre-download the weights
```

`pull` snapshots the repo, loads the image into the local Docker daemon (or `docker pull`s it
when the author published to a real registry), and records the package in the local db. It
compares the image **digest** the manifest records against what Docker holds, so a re-pull of
an unchanged image is a no-op and a republished one is reloaded.

**`pull` does not fetch weights unless you pass `--with-weights`.** The weights go into the
consumer's own HF cache under the consumer's own token, so a small image can front a large
checkpoint and the same image works for anyone entitled to the weights.

Check compatibility before you serve. `tt-model info <org>/<model-name>` prints the manifest
and the required-vs-detected verdict; an `arch` mismatch is fatal, because the image's binaries
target a different ISA. `serve` itself refuses up front when the host does not have enough free
chips for the chosen profile.

---

## Step 5: Serve (consumer)

```bash
tt-model serve <org>/<model-name>                      # pulls if needed, fetches weights, serves
tt-model serve <org>/<model-name> --profile <name>     # a non-default serve profile
tt-model serve <org>/<model-name> --port 8001          # serve's own option
tt-model serve <org>/<model-name> --device-id 0,1      # pin specific chips instead of auto-picking free ones
tt-model serve <org>/<model-name> --detach             # start the container and return at once
tt-model serve <org>/<model-name> --print              # print the full docker run argv, don't run
tt-model serve <org>/<model-name> --no-weights         # let the model fetch weights inside the container
```

`serve` on a fresh host does everything: pulls the image, **puts the weights on the host
first** (checking free disk before it starts, resuming a partial download, and saying so), picks
exactly as many free chips as the profile needs, starts the container, and watches the boot as
a checklist parsed from the container log:

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

When the server reports ready the list is replaced by one `✓ <org>/<model-name> ready` line and
a card with the endpoint, a `tt-model curl` example, and the `logs` and `stop` hints. A cold first
boot JIT-compiles kernels (about 10 minutes); the compiled kernels are cached on the host under
`~/.cache/tt-model/<name>/` so the second boot is fast. A boot that fails marks the step it died
in and renders a diagnosis card instead of dumping the log. Ctrl-C stops the watching only; the
container keeps booting.

Only the profile's chips are handed to the container, because TT-Metalium locks every chip it
can see for the container's whole life. That scoping is what lets several models coexist on one
board. Not enough free chips is an immediate error naming what is busy, rather than a hang.

The pass-through rule is the same as for every format: `--port`, `--profile`, `--detach`,
`--print`, `--no-weights`, `--device-id`, `--refresh`, `--no-update-check`, `--local-only`, and
`--force` belong to `serve`; anything else after the id goes to vLLM. See
[cli.md](cli.md#run-a-model).

### Around a running server

```bash
tt-model profiles <org>/<model-name>          # the serve profiles and the default
tt-model logs     <org>/<model-name> -f       # follow the server log
tt-model stop     <org>/<model-name>          # clean SIGTERM; closes the mesh
tt-model rm       <org>/<model-name>          # remove the pulled package, image, and caches
tt-model rm       <org>/<model-name> --keep-cache        # keep JIT kernels for a fast re-pull
tt-model rm       <org>/<model-name> --include-weights   # also delete the weights from the HF cache
```

Always `stop`, never `docker kill`. A `SIGKILL` leaves the device dirty; `stop` attempts a
`tt-smi -r` scoped to the container's own chips, but a force-killed teardown can leave a device
that only a host reboot restores.

### Updating to a newer published revision

When an installed package is served, `serve` makes one short (3 s bounded) Hub check and, if
the author has pushed a newer revision, prints a non-blocking advisory. To update:

```bash
tt-model serve <org>/<model-name> --refresh    # re-pull if newer, reload the image if its digest changed, then serve
tt-model pull  <org>/<model-name>              # or update without serving
```

A refresh that fails for any reason warns and serves the installed package unchanged. Disable
the check per-serve with `--no-update-check`, or entirely offline with `--local-only`. A pinned
`<org>/<model-name>@<revision>` is never refreshed.

---

## Step 6: Verify (consumer)

```bash
tt-model curl "Say hello in one sentence."
```

`tt-model curl` reads the model id from the running server and builds the request for you.
The equivalent raw request, on the port the ready card printed (default 20000):

```bash
curl -s http://localhost:20000/v1/models
curl -s http://localhost:20000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"unsloth/Llama-3.2-3B-Instruct",
       "messages":[{"role":"user","content":"Say hello in one sentence."}],
       "max_tokens":64}'
```

Coherent text back means the full producer-to-consumer loop works. (`model` is the weights repo
id from the manifest, which `/v1/models` also reports.)

If the manifest declares `capabilities.tool_parser`, check tool calling too. Send a real
`tools` request and read the verdict, not the prose:

```bash
curl -s http://localhost:20000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "unsloth/Llama-3.2-3B-Instruct",
  "messages": [{"role": "user", "content": "What is the weather in Paris right now, in Celsius?"}],
  "tools": [{"type": "function", "function": {"name": "get_weather",
    "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {
      "city": {"type": "string"}, "metric": {"type": "boolean"}},
      "required": ["city"]}}}],
  "tool_choice": "auto", "max_tokens": 256, "temperature": 0
}' | python3 -c 'import sys, json; c = json.load(sys.stdin)["choices"][0]; \
print(c["finish_reason"], json.dumps(c["message"].get("tool_calls")))'
```

Pass: `finish_reason` is `tool_calls` and the arguments are valid JSON matching the schema.
Fail: prose in `content` with `finish_reason` `stop`, which means the parser is not in the
launch (check `--print` for `--enable-auto-tool-choice --tool-call-parser <name>`).

---

## The self-containment guarantee (why this is safe to hand around)

A v5.1 package is **hermetic at build time**: the OS, the built TT-Metalium tree, vLLM, the
plugin, and the model code are all inside the image, pinned to the commits the `built:` block
records. Nothing about the consumer's host can matter because there is no host interpreter or
host platform in the picture. Serving depends on nothing outside the image **except Docker, the
Tenstorrent device, the hugepages mount, and the bind-mounted caches** (the HF cache for
weights, and the per-model JIT and converted-weight caches under `~/.cache/tt-model/<name>/`).
The steps that touch the network are `pull` (the image) and the weights fetch on `serve` or
`pull --with-weights`; a fully cached model serves air-gapped.

The image also **verifies itself at build time**: imports resolve, torch is the `+cpu` build
TT-Metalium pins, and every model registered through `extra_models_dir` is actually resolved,
on the author's machine, before anything is pushed.

---

## Troubleshooting (the non-obvious parts)

| Symptom | Cause / fix |
|---|---|
| `package`: a `source.code` path does not exist | The allowlist names a path that is not in the `tt-metal` tree. Fix the path; a missing one is refused rather than shipped empty. |
| `package`: `ModuleNotFoundError` in the verify stage for a package nothing declares | The live resolve never installed it (for example `pytest` imported at module level by a `models/common` helper). Seed `runtime.lock` from the validated venv and strip editable, `file://`, `vllm==*+*`, and CUDA-only lines. |
| `package`: a `models/...` file missing at verify | The allowlist under-lists. Watch for lazy imports and data files read with a silent fallback; ship them and assert them in `verify:`. |
| `package`: circular `current_platform` ImportError in a verify line | Import `vllm` **first** in any check that touches `vllm_tt_plugin.platform`. |
| `push`: "not a staged container package" | `push` takes the directory `package --container` produced (the one holding `tt_kernel_manifest.json` with a `container` block), not the YAML and not a v6 bundle. |
| `tt-model info`: arch mismatch | The image's binaries target a specific ISA. Serve on a matching host, or repackage for this one. |
| `serve` refuses: hugepages | Mount 1G hugepages at exactly `/dev/hugepages-1G`. A subdirectory or 2M pages fails the driver's match. |
| `serve` refuses: not enough free chips | Another container (including one `tt-model` did not launch) holds them. `docker ps`, then `tt-model stop` or wait. |
| `serve`: not enough disk space for the weights | Free space on the HF cache filesystem, or point `HF_HOME` at a larger one. `serve` stops before a fetch that would fill the disk. |
| boot dies on mesh open: `Timed out while waiting for active ethernet core` | The devices are dirty from an earlier hard kill, not the package. Confirm nothing else holds them, `tt-smi -r all`, serve again. |
| boot hangs with no error after another serve | Two serves picked the same chip inside the roughly 100 ms window between scan and start. `tt-model stop` the second one. |
| serve: "Address already in use" | Another server holds the port. Use `tt-model serve <id> --port 8001`. |
| Tool calling not working | The manifest must declare `capabilities.tool_parser`; the launcher emits `--enable-auto-tool-choice --tool-call-parser <name>`. Check with `--print`. |
| Nothing registers in vLLM | `vllm_metadata.json` must live in a **child** of `runtime.extra_models_dir`, and that directory must be covered by `source.code`. |
| `serve` after `docker image rm` on a pulled package | A pulled package keeps only the manifest; re-run `tt-model pull`. A locally built package reloads from its staged `image/`. |

See [docs/container_packages.md](container_packages.md) for the design details, how the
container runs, which chips a serve gets, and the hardware validation run.
