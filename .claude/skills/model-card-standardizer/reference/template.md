# The `card:` block, field by field

The generator renders every section of the published card, in a fixed order, from this
block of `tt-model.yaml`. Fill in what only you know; leave out a field rather than
writing filler. `performance` and `limitations` are required for the package to be
listed in the community catalog, and render as "Not provided by the package author."
when missing.

```yaml
card:
  # One dense paragraph, shown right under the title: what the model is, what it does,
  # which hardware it runs on, and how mature the port is.
  description: >-
    Qwen3-32B served with vLLM on a single p150x4 (four Blackhole chips).
    Experimental community bring-up: text only, tested at up to 32 concurrent users.

  # At a glance
  architecture: 32B-parameter dense decoder-only transformer
  status: Experimental community bring-up   # your port's maturity; never "verified"

  # Intended use. The out-of-scope half is the one most often left out, and the one that
  # costs: a text-only port of a multimodal checkpoint looks like the checkpoint until
  # someone sends an image.
  intended_use: Chat and tool calling for English and Chinese text.
  out_of_scope_use: Image or audio input; the vision tower is not ported.

  # Added after the generated `tt` / `tt-model` commands. Only what they do not cover:
  # first-boot time, the log line that means the server is ready, a required env var.
  quickstart: |
    The first boot compiles kernels and takes about 10 minutes; later boots take
    about 1. The server is ready when the log shows `Application startup complete`.

  # Added after the generated API description. For a chat model: tool-calling and
  # sampling notes. For anything else: the real endpoint, request schema and an example
  # response.
  usage: |
    Tool calling works with `"tool_choice": "auto"`. Default sampling is temperature 0.6.

  # Required to list. Accuracy against the reference (PCC, exact match, or the
  # modality's equivalent) and throughput/latency, plus one sentence on how it was
  # measured: hardware, concurrency, prompt set.
  performance: |
    | metric | value |
    | --- | --- |
    | Top-1 agreement with the HF reference | 98.7% |
    | Decode, batch 1 | 21 tok/s/user |
    | Decode, batch 32 | 12 tok/s/user |

    Measured on p150x4 with 128-token prompts and 512-token completions.

  # Required to list. What is not implemented, known defects and their status, which
  # hardware was actually validated, and anything declared but untested.
  limitations: |
    - No image input.
    - Contexts above 32k tokens are declared but not validated.

  # Recommended. Failure modes with user-facing consequences, uses this model should not
  # be trusted for, and any precision difference from the reference that could quietly
  # change results.
  risks: |
    Occasional repetition loops at temperature 0; use the default sampling settings.

  # When the weights carry any restriction beyond the port code's own license: the
  # weights, any vendored upstream code, and the port code, each with its commercial
  # and redistribution terms stated plainly. Skip it when the weights are unrestricted.
  licensing: |
    Weights: Apache-2.0. Port and serving code: Apache-2.0.

  # Sibling packages for the same base model on other hardware or precision, and
  # whether one replaces another.
  related: |
    - [someone/qwen3-32b-p300x2](https://huggingface.co/someone/qwen3-32b-p300x2): same model on p300x2.

  # Frontmatter. `license.id` is an SPDX id, or "other" with `name` and `link`.
  license:
    id: apache-2.0
  pipeline_tag: text-generation      # defaults to the runtime's own tag
  base_model:                        # defaults to the weights repo
    - Qwen/Qwen3-32B
```

All values above are illustrations, not real measurements.

## What the generator adds on its own

- **Tags**: hardware, board, catalog and runtime tags. Never add a tag claiming the model
  is verified; verified means Tenstorrent has copied it into its own Hugging Face org.
- **Quickstart commands**: `tt model pull` / `tt serve`, then the `tt-model`-only
  equivalent, so the card never requires tt-cli.
- **Serve profiles** table, when there is more than one profile.
- **Using it**: whether the server speaks the OpenAI API, and how to call it.
- **Feedback**: the repo's Discussions page, `tt report issue` for problems with the `tt`
  tool itself, and support@tenstorrent.com.
- **Provenance**: what the image was built from.

## Not supported yet

- **Changelog**: there is no field, and the card is rendered fresh on every build.
- **A machine-readable performance summary** in the frontmatter: not emitted.
