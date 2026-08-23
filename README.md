# tt-model

Package a working Tenstorrent model as a Docker image, publish it on Hugging Face, and
serve it anywhere with a TT card — with nothing installed on the host but `tt-model`
itself.

```
   AUTHOR                                                  CONSUMER
   ------                                                  --------
   tt-metal fork (built, working)                           tt-model pull org/name
        │                                                        │ image  -> docker load
        │  tt-model package tt-model.yaml                        │ code/  -> readable on the Hub
        ▼                                                        │ weights-> host HF cache
   docker image  ──┐                                             ▼
   code/         ──┼── tt-model push ──►  HF model repo ──►  tt-model serve org/name
   tt-model.yaml ──┤                                             │
   requirements  ──┘                                             ▼
                                                          docker run --device /dev/tenstorrent
```

A model is a container: Ubuntu, the tt-metal build (ttnn + the JIT's RISC-V toolchain),
vLLM + the TT plugin, and the model's own code — all pinned, all inside the image.
**Weights are the only thing that touches the host**, in the host's own Hugging Face
cache, bind-mounted in. No host tt-metal, no venv provisioning, no version conflicts.

## Install

```bash
uv tool install tt-model        # or: pipx install tt-model / pip install tt-model
```

Requires Docker (with BuildKit) and, for gated weights, `hf auth login`.

## Serve a published model

```bash
tt-model pull  org/model            # image -> docker, weights -> ~/.cache/huggingface
tt-model list  org/model            # its serve profiles, and which fit this machine
tt-model serve org/model --follow   # boots in ~10 min; --follow waits for ready
curl localhost:8000/v1/models
tt-model stop  org/model            # SIGTERM-first: the server closes the mesh itself
```

`serve` runs the container with the flags the hardware actually needs
(`--device /dev/tenstorrent`, `--ipc host`, the hugepages mount, a persistent per-model
kernel-cache dir so later boots are fast). See exactly what it will run with
`serve --print`.

A model can ship several **serve profiles** — device targets (`p150x2` vs `p150x4`)
and deployment shapes (a latency profile for one interactive user vs a capacity profile
for 32) — all served by the *same* image. `serve` uses the author's default and says
so; `--profile <name>` picks another.

## Package your model

Your model already serves from your tt-metal fork. Describe it in one YAML file:

```yaml
schema: 1
repo: you/my-model
name: my-model
weights: org/Weights-7B
type: vllm                    # or vllm-legacy — docs/model_types.md
arch: blackhole
source:
  tt_metal: /path/to/tt-metal
  code:                       # EXACTLY what ships — an allowlist
    - models/common
    - models/autoports/my_model/tt
    - models/autoports/my_model/vllm_ext
  ubuntu: "22.04"
  python: "3.12"
runtime:
  vllm: {version: "0.24.0"}
  plugin: {repo: https://github.com/tenstorrent/vllm-tt-plugin, ref: main}
  extension: models/autoports/my_model/vllm_ext
serve:
  port: 8000
  block_size: 64
serve_profiles:
  - {name: p150x4, hardware: p150x4, mesh_device: P150x4, max_num_seqs: 8, max_model_len: 131072}
default_profile: p150x4
```

```bash
tt-model package tt-model.yaml      # 2.5-4 h cold; streams live + tail command printed
tt-model serve build/my-model/tt-model.yaml --follow    # prove it serves locally
tt-model push  build/my-model                            # private by default
```

Full walkthrough: [docs/packaging.md](docs/packaging.md). Worked examples for both
model types: [examples/](examples/).

## Commands

| command | does |
| --- | --- |
| `package <yaml>` | build the image + staged repo dir from one manifest |
| `push <dir>` | upload to HF (`--public` to opt in; existing visibility never touched) |
| `pull <org/name>` | image → docker, weights → host HF cache |
| `serve <target>` | run it (`--profile`, `--print`, `--follow`, `--force`) |
| `stop <target>` | SIGTERM-first stop; mesh reset only if a kill was needed |
| `logs <target>` | server logs (`-f` to stream) |
| `list [target]` | local images/containers, or a model's profiles |

## What lands where

| | author's machine | HF repo | consumer's machine |
| --- | --- | --- | --- |
| image | docker daemon | `image/` (OCI blobs, deduped) | docker daemon |
| model code | your fork | `code/` — browsable, byte-identical to the image | inside the image |
| manifest | your YAML | `tt-model.yaml`, fully pinned | `~/.cache/tt-model/pulled/` |
| weights | host HF cache | — (a pointer) | host HF cache |

## Development

```bash
uv venv && uv pip install -e ".[test]"
pytest            # 129 offline tests: no hardware, no docker daemon needed
```

Design notes live in [docs/packaging.md](docs/packaging.md) and
[docs/model_types.md](docs/model_types.md).
