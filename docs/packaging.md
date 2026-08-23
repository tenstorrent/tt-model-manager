# Packaging a model

The whole authoring interface is **one YAML file**. You point `tt-model package` at it;
there are no per-field flags. This walkthrough uses the real, validated example in
`examples/laguna-xs-2.1.yaml`.

## Prerequisites

- A tt-metal fork checkout in which the model **already serves** (that is the input —
  tt-model packages working models, it does not port them).
- Docker with BuildKit (any recent Docker; plain `docker build` is used, never bake).
- `hf auth login` (or `HF_TOKEN`) for `push`.

Nothing else. Packaging installs nothing on your host.

## 1. Write the manifest

Put `tt-model.yaml` next to your model in the fork (recommended — the serving recipe
then travels through review with the model code) or anywhere else:

```yaml
schema: 1
repo: you/my-model            # the HF repo push publishes to
name: my-model
weights: org/Weights-7B       # HF id; downloaded to the HOST HF cache, never baked
type: vllm                    # see docs/model_types.md
arch: blackhole

source:
  tt_metal: /path/to/your/tt-metal      # or {repo: ..., ref: ...} for CI
  code:                                  # EXACTLY the files that ship. An allowlist.
    - models/common
    - models/autoports/my_model/tt
    - models/autoports/my_model/vllm_ext
  ubuntu: "22.04"
  python: "3.12"

runtime:
  vllm: {version: "0.24.0"}
  plugin: {repo: https://github.com/tenstorrent/vllm-tt-plugin, ref: main}
  extension: models/autoports/my_model/vllm_ext

serve:                        # defaults every profile inherits
  port: 8000
  block_size: 64              # required — the TT backend rejects vLLM's default
  args: [--trust-remote-code]

serve_profiles:               # one image serves ALL profiles; pick at `serve` time
  - name: p150x4
    hardware: p150x4
    mesh_device: P150x4       # the plugin's closed enum — validated at load
    max_num_seqs: 8           # required — ditto
    max_model_len: 131072
default_profile: p150x4
```

Three rules that catch real mistakes at load time instead of ten minutes into a boot:

- `mesh_device` must be one of the plugin's presets (or a `"(rows, cols)"` tuple), and
  `rows*cols` must match the chip count implied by `hardware`.
- `max_num_seqs` and `block_size` are required on every (merged) profile.
- Every `source.code` entry must exist — a missing entry is an **error**, never a
  silent skip. The allowlist is also enforced structurally: the image's tt-metal tree
  contains **no** `models/` except what you listed, so an under-specified list fails
  the image's own build-time import check, on your machine.

**The allowlist includes data your model loads at runtime.** Laguna's precision config
lives under `doc/datatype_sweep/` and the model falls back *silently* to in-code
defaults without it — which is why its manifest ships that directory and why the image
verifies the file is present at build time. Audit your model for the same pattern.

## 2. Package

```bash
tt-model package path/to/tt-model.yaml
# watch from another terminal:
#   tail -f ~/.cache/tt-model/build/<name>.log
```

Expect **2.5–4 hours cold** (tt-metal C++ build + vLLM from sdist); minutes on a warm
BuildKit cache. Output streams live and tees to the log. A first Ctrl-C on a TTY only
*warns* (elapsed time, current stage, what a cancel costs); a second within 10 s
cancels, keeps the caches, and prints the resume command.

What it does:

1. Pins provenance: the checkout's HEAD sha (uncommitted changes are packaged — that is
   the point of the hermetic default — and recorded as `dirty: true`), plus every
   `ref:` in `runtime:` resolved to a sha.
2. Stages `code/` from your checkout and builds the image (see `docker/Dockerfile` via
   `src/tt_model/docker/`). The image contains no git metadata and no reference to
   your fork.
3. Runs the build-time verification inside the image: imports (`ttnn`, `vllm`,
   `vllm_tt_plugin`, your generator), CPU torch, plugin registration from
   `EXTRA_MODELS_DIR`, and any model-specific assertions.
4. Freezes the resolved python env to `requirements.lock`. Commit that file next to
   your manifest and name it under `runtime.lock:` — every later build then installs
   exactly that set and nothing resolves at build time.
5. Exports the image as an exploded OCI layout and writes the repo directory:

```
build/my-model/
├── README.md            the generated model card (profiles table, provenance)
├── tt-model.yaml        the PINNED manifest (shas, built: block)
├── requirements.lock
├── code/                byte-identical to what is inside the image
└── image/               OCI layout — content-addressed blobs, deduped on the Hub
```

## 3. Serve it locally before pushing

```bash
tt-model serve build/my-model/tt-model.yaml --follow
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"org/Weights-7B","messages":[{"role":"user","content":"hi"}]}'
tt-model stop build/my-model/tt-model.yaml
```

`serve` composes the `docker run` (inspect it with `--print`): TT devices, `--ipc
host`, the hugepages mount, your HF cache read-write at `HF_HOME`, and a per-model
host directory for the JIT kernel cache (`~/.cache/tt-model/<name>/cache`) so the
~10-minute compile is paid once, not per boot.

## 4. Push

```bash
tt-model push build/my-model              # new repos are created PRIVATE
tt-model push build/my-model --public     # explicit opt-in to public
```

Visibility is tri-state: with no flag, an existing repo's visibility is **never
touched**. The upload is blob-by-blob with resume, so a second model built on the same
tt-metal commit re-uses the base/venv/sfpi layers and uploads only its own.

## 5. Consume

```bash
tt-model pull  you/my-model     # image -> docker, weights -> host HF cache
tt-model list  you/my-model     # profiles, and which fit this machine
tt-model serve you/my-model --follow
tt-model stop  you/my-model     # SIGTERM-first; mesh reset only if a kill was needed
```

The host ends up with: the docker image, the weights in its own HF cache, and nothing
else. No tt-metal, no vLLM, no venv, no version conflicts.
