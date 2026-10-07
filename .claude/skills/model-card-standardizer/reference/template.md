# The `card:` block, field by field

The generator renders the published card, in a fixed order, from this block of
`tt-model.yaml` plus what it already knows from the manifest. Fill in what only you know;
leave out a field rather than writing filler, and never repeat what the generator prints.
`performance` and `limitations` are required for the package to be listed in the
community catalog, and render as "Not provided by the package author." when missing.

Describe the model as it is now. No history of the port, and no implementation detail
unless a user needs it to run the model or trust its output.

```yaml
card:
  # Block quote directly under the title: the model this package serves and who made it.
  # Not the same as the note a verified copy in the Tenstorrent org opens with; that one
  # is added by `tt-model verify`.
  attribution: >-
    Derived from [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) by the Qwen team,
    unmodified. See the original model card for training and evaluation details.

  # One dense paragraph: what the model is (size and architecture) and what it does.
  # The hardware sentence is generated right after it, so do not name the board here.
  description: >-
    Qwen3-32B, a 32B-parameter dense decoder-only transformer, served with vLLM. Text
    only, tested at up to 32 concurrent users.

  # What the model is for.
  intended_use: Chat and tool calling for English and Chinese text.

  # Extra bullets only. The generator already lists the tt CLI (or tt-model alone),
  # Docker and the board. Indent a bullet's continuation lines.
  prerequisites: |
    - 64 GB of host RAM for the first weight conversion

  # Added after the generated `tt serve` command, which already says how long the first
  # boot takes and which log line means ready. Only what it does not cover: a required
  # env var, a demo URL or test script, a prompting convention.
  quickstart: |
    Run `python demo/chat.py --port <port>` for an interactive chat in the terminal.

  # Added under Capabilities, after the generated note on the API and tool calling. For
  # a chat model: tool-calling and sampling notes. For a server that does not speak the
  # OpenAI API: the endpoint, request schema and an example response.
  usage: |
    Tool calling works with `"tool_choice": "auto"`. Default sampling is temperature 0.6.

  # Required to list. First an eval table: each accuracy or eval result against the
  # reference model, with the reference's own figure beside it and the prompt set. Then
  # for an LLM a speed table, one row per serving profile, with these columns (package
  # warns when one is missing), then one sentence on how it was measured: hardware,
  # prompt set, tool. N is the number of benchmark runs behind each row.
  performance: |
    | eval | this package | reference | prompts |
    | --- | --- | --- | --- |
    | Top-1 token agreement with the HF reference | 98.7% | — | 32 |
    | GSM8K (exact match, 8-shot) | 78.1 | 79.0 | 1,319 |

    | profile | ISL | OSL | concurrency | N | TTFT (ms) | prefill tok/s/u | decode tok/s/u | E2EL (s) |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | default | 128 | 512 | 1 | 1 | 180 | 710 | 21 | 24.6 |
    | default | 128 | 512 | 32 | 1 | 950 | 135 | 12 | 43.6 |

    Medians, measured on p150x4 with the tt-inference-server benchmark workflow.

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

All values above are illustrations, not real measurements.

## Performance for other modalities

For a modality without its own guidance, report the metrics the original Hugging Face
card reports, measured the same way, side by side with the original's numbers.

**Image and video generation (draft, pending sign-off):**

- **Speed:** seconds per image (or per second of video) at a stated resolution, number of
  denoising steps and guidance scale, at concurrency 1 and at the profile's maximum, one
  row per serving profile.
- **Quality against the reference:** the same prompts and seeds on the reference
  implementation and on Tenstorrent. Report CLIP score for each, and FID or LPIPS where a
  reference image set exists. Say how many prompts.
- **Comparison:** any speed or quality figure the original card reports, beside ours.

## What the generator adds on its own

- **Tags**: hardware, board, catalog and runtime tags. Never add a tag claiming the model
  is verified; tt-cli decides that, from its curated list and the Tenstorrent org.
- **Title**: the package name and the boards it runs on.
- **Hardware sentence** and **At a glance**: hardware, context and license, from the
  manifest.
- **Prerequisites**: the tt CLI (or tt-model alone), Docker and the board.
- **Quickstart**: `tt serve <repo>`, how long the first boot takes and the ready line,
  then a one-line `tt-model serve <repo>` path for a box without tt-cli.
- **Serving profiles** table, when there is more than one profile.
- **Capabilities**: whether the server speaks the OpenAI API, how to call it, and the
  tool-calling and reasoning notes.
- **Feedback**: the repo's Discussions page, `tt report issue` for problems with the `tt`
  tool itself, and support@tenstorrent.com.
- **Provenance**: what the image was built from.

## Not supported yet

- **Changelog**: deferred until users ask for one. The card is rendered fresh on every
  build.
- **A machine-readable performance summary** in the frontmatter: not emitted.
