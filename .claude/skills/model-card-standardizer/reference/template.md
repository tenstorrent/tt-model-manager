# The `card:` block, field by field

The generator renders every section of the published card, in a fixed order, from this
block of `tt-model.yaml`. Fill in what only you know; leave out a field rather than
writing filler. `performance` and `limitations` are required for the package to be
listed in the community catalog, and render as "Not provided by the package author."
when missing.

Describe the model as it is now. No history of the port, and no implementation detail
unless a user needs it to run the model or trust its output.

```yaml
card:
  # One dense paragraph, shown right under the title: what the model is (size and
  # architecture), what it does, and which hardware it runs on.
  description: >-
    Qwen3-32B, a 32B-parameter dense decoder-only transformer, served with vLLM on a
    single p150x4 (four Blackhole chips). Text only, tested at up to 32 concurrent users.

  # The model this package is derived from, rendered as a block quote.
  attribution: >-
    This package serves Qwen/Qwen3-32B by the Qwen team, unmodified, on Tenstorrent
    hardware.

  # Anything needed outside this package to serve it.
  prerequisites: |
    - tt-cli (`pip install tenstorrent`)
    - Docker

  # What the model is for.
  intended_use: Chat and tool calling for English and Chinese text.

  # Added after the generated `tt model pull` / `tt serve` commands. Only what they do
  # not cover: first-boot time, the log line that means the server is ready, a required
  # env var, a demo URL or test script. For a server that does not speak the OpenAI API,
  # also the endpoint, request schema and an example response.
  quickstart: |
    The first boot compiles kernels and takes about 10 minutes; later boots take
    about 1. The server is ready when the log shows `Application startup complete`.

  # Required to list. Accuracy against the reference model, then one row per serving
  # profile. For an LLM every row carries ISL, OSL, concurrency, N (number of runs),
  # TTFT, prefill and decode tok/s/user, and E2EL. Other modalities: report the metrics
  # the original Hugging Face card reports, measured the same way, side by side.
  performance: |
    Top-1 token agreement with the HF reference: 98.7%.

    | profile | ISL | OSL | concurrency | N | TTFT (ms) | prefill tok/s/u | decode tok/s/u | E2EL (s) |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | default | 128 | 512 | 1 | 5 | 180 | 710 | 21 | 24.6 |
    | default | 128 | 512 | 32 | 5 | 950 | 135 | 12 | 43.6 |

    Measured on p150x4 with the tt-inference-server benchmark prompt set.

  # Required to list. Three things: what is not implemented or not tested (including
  # hardware), what the model should not be used for, and failure modes or precision
  # differences from the reference a user would notice.
  limitations: |
    - No image or audio input; the vision tower is not ported.
    - Contexts above 32k tokens are declared but not validated.
    - Occasional repetition loops at temperature 0; use the default sampling settings.

  # Optional. Only when the weights carry a restriction beyond the port code's own
  # license: the weights, any vendored upstream code, and the port code, each with its
  # commercial and redistribution terms.
  licensing: |
    Weights: Apache-2.0. Port and serving code: Apache-2.0.

  # Optional. Sibling packages for the same base model on other hardware or precision,
  # and whether one replaces another.
  related: |
    - [someone/qwen3-32b-p300x2](https://huggingface.co/someone/qwen3-32b-p300x2): same model on p300x2.

  # Frontmatter. `license.id` is an SPDX id, or "other" with `name` and `link`.
  license:
    id: apache-2.0
  pipeline_tag: text-generation      # defaults to the runtime's own tag
  base_model:                        # defaults to the weights repo
    - Qwen/Qwen3-32B
```

All values above are illustrations, not real measurements. Performance guidance for
image and video generation is still to be written.

## What the generator adds on its own

- **Tags**: hardware, board, catalog and runtime tags. Never add a tag claiming the model
  is verified; verified means Tenstorrent has copied it into its own Hugging Face org.
- **Quickstart commands**: `tt model pull <repo>` and `tt serve <repo>`.
- **Serve profiles** table, when there is more than one profile.
- **Using it**: whether the server speaks the OpenAI API, and how to call it.
- **Feedback**: the repo's Discussions page, `tt report issue` for problems with the `tt`
  tool itself, and support@tenstorrent.com.
- **Provenance**: what the image was built from.

## Not supported yet

- **Changelog**: deferred until users ask for one. The card is rendered fresh on every
  build.
- **A machine-readable performance summary** in the frontmatter: not emitted.
