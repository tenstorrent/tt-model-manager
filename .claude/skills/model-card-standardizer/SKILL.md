---
name: model-card-standardizer
description: Check a tt-model card — an authored tt-model.yaml's `card:` block, a locally built package's README.md, or a published Hugging Face repo's README — against the standard card layout, separate what the model's author fills in from what only a change to the card generator can fix, and interview the author to draft the missing `card:` fields. Use for "does this model card follow the template", "review this model card", or "help me write the card for this model". Advisory only — never pushes to Hugging Face, never edits a published repo, and only edits a local tt-model.yaml's `card:` block after showing the diff and getting confirmation.
---

# Standardize a tt-model card

## How a card is made

Authors never write the README. `render_model_card()` in `src/tt_kernel/build.py`
renders it from the manifest, and the author's input is the `card:` block of their
`tt-model.yaml` (`CardSettings` in `src/tt_kernel/container_manifest.py`, built on
`CardSpec` in `src/tt_kernel/manifest.py`). Unknown keys in `card:` are rejected.

Read those before relying on the table below. If the fields or sections have changed,
the code wins.

| Card section | Filled by |
| --- | --- |
| Frontmatter `tags` | the generator (hardware, board, catalog and runtime tags) |
| Frontmatter `license`, `pipeline_tag`, `base_model` | `card.license`, `card.pipeline_tag`, `card.base_model` |
| Lead paragraph | `card.description`, including size and architecture |
| Attribution | `card.attribution` |
| Prerequisites | `card.prerequisites` |
| Intended use | `card.intended_use` |
| Quickstart | the generator (`tt model pull`, `tt serve`), then `card.quickstart` |
| Serve profiles | the generator, when there is more than one profile |
| Using it | the generator (what the API is) |
| Expected performance | `card.performance` — required to list in the catalog |
| Limitations | `card.limitations`, including out-of-scope uses and risks — required to list in the catalog |
| Licensing | `card.licensing` (optional) |
| Related packages | `card.related` (optional) |
| Feedback, Provenance | the generator |

What a good value looks like for each field is in `reference/template.md`.

This field set is the team's agreed standard, and the generator does not match it yet.
Report each of these as a generator gap, not an author fix:

- `attribution` and `prerequisites` are rejected by `CardSettings`.
- `architecture`, `status`, `out_of_scope_use`, `usage` and `risks` are still accepted
  and rendered, as At a glance, Out-of-scope use, Using it notes and Risks.
- The Quickstart still prints the `tt-model`-only commands after the `tt` ones.
- `performance` is free text, so its required columns are not checked.
- There is no changelog, and no machine-readable performance summary in the frontmatter.

Verified status is never part of a card. A model is verified when Tenstorrent copies
it into the `Tenstorrent` Hugging Face org; the org is the signal. Anyone can edit their
own card, so a card that calls itself verified (in a tag or prose) is wrong and should
be flagged.

## Step 1 — Find what you are checking

- **An authored `tt-model.yaml`, not yet built**: read its `card:` block (it may be
  missing entirely).
- **A locally built package**: read `<out>/README.md`, the generator's real output.
- **A published repo**: fetch it read-only, no login needed for a public repo:
  `curl -sL "https://huggingface.co/<repo>/raw/main/README.md"`.

## Step 2 — Score every section

For each row of the table, classify it as:

- **OK**: present, and the content meets the guidance in `reference/template.md`.
- **Author fix**: missing or weak. Name the `card:` field it belongs in.
- **Generator gap**: the generator cannot express it. Cite the file and line you checked.

Pay most attention to `performance` and `limitations`: the catalog will not list a
package without them. A `limitations` that does not say what the model should not be
used for is weak, and is the gap most often left.

A `card:` block that still uses `architecture`, `status`, `out_of_scope_use`, `usage` or
`risks` is an author fix: move architecture into `description`, out-of-scope uses and
risks into `limitations`, and non-OpenAI endpoint details into `quickstart`. Drop
`status`; the Tenstorrent org is the only maturity signal.

## Step 3 — Interview for the author fixes

Ask for everything missing in one batch, in the style of the `tt-model-yaml` skill:
give your best guess and where it came from, so the author confirms rather than writes
from scratch. Then draft a `card:` block with each answer in its own field. Use
`card.quickstart` only for what the generated commands do not cover, such as first-boot
time, the log line that means the server is ready, or a demo URL. Describe the model as it
is now: no history of the port, and no implementation detail a user does not need.

Until the generator accepts `attribution` and `prerequisites`, a `card:` block carrying
them fails to build. Draft them, but write attribution into the end of `description` and
prerequisites into the start of `quickstart`, and list both as generator gaps.

## Step 4 — Offer to write it

- **Unbuilt `tt-model.yaml`**: show the diff to the `card:` block and write it only after
  the author confirms.
- **Already built or published**: a card change needs a rebuild and re-push
  (`tt-model package --container`, then `tt-model push`). Say so. Never patch a published
  README directly; the manifest is the source of truth.

## Step 5 — Report

Two lists:

- **Author fixes**: what you drafted, and which field each part goes in.
- **Generator gaps**: what needs a tt-model-manager code change, each with the file and
  line you checked.

Read-only against Hugging Face throughout: never push, publish, or comment unless the
user explicitly asks.
