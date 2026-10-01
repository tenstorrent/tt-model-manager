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
| Lead paragraph | `card.description` |
| At a glance | `card.architecture`, `card.status`, the license; hardware and context come from the serve profiles |
| Intended use | `card.intended_use`, `card.out_of_scope_use` |
| Quickstart | the generator (`tt` and `tt-model` commands), then `card.quickstart` |
| Serve profiles | the generator, when there is more than one profile |
| Using it | the generator (what the API is), then `card.usage` |
| Expected performance | `card.performance` — required to list in the catalog |
| Limitations | `card.limitations` — required to list in the catalog |
| Risks and safety considerations | `card.risks` |
| Licensing | `card.licensing` |
| Related packages | `card.related` |
| Feedback, Provenance | the generator |

What a good value looks like for each field is in `reference/template.md`.

Two things a card cannot carry today, because the generator has no field for them: a
changelog (the card is rendered fresh on every build) and a machine-readable
performance summary in the frontmatter. Report these as generator gaps, not author
fixes.

Verified status is never part of a card. A model is verified when Tenstorrent copies
it into the `Tenstorrent` Hugging Face org; the org is the signal. Anyone can edit their
own card, so a card that calls itself verified (in `status`, a tag, or prose) is wrong
and should be flagged.

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

A missing section is almost always an author fix now: every section above has its own
field. Pay most attention to `performance` and `limitations` (the catalog will not list
a package without them) and to `out_of_scope_use`, the one most often left out.

## Step 3 — Interview for the author fixes

Ask for everything missing in one batch, in the style of the `tt-model-yaml` skill:
give your best guess and where it came from, so the author confirms rather than writes
from scratch. Then draft a `card:` block with each answer in its own field. Use
`card.quickstart` only for what the generated commands do not cover, such as first-boot
time or the log line that means the server is ready.

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
