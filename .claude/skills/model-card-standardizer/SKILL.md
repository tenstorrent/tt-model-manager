---
name: model-card-standardizer
description: Check a tt-model card — an authored tt-model.yaml's `card` fields, a locally built package's README.md, or an already-published HF repo's README — against the proposed model-card template, separate what the model's author can fix from what the `render_model_card` generator itself hardcodes, and interview the author to draft the missing free-text sections as ready-to-paste `card.quickstart` markdown. Use for "does this model card follow the template", "review/standardize this model card", "help me write the card for this model", or "check this repo against the template". Advisory only — never pushes to HF, never edits a published repo, and only edits a local tt-model.yaml's `card:` block after showing the diff and getting confirmation.
---

# Standardize a tt-model card against the proposed template

## Read this first: what actually generates a card

Two things combine into the published `README.md`, and only one of them is
something a model's author can edit per package:

1. **`render_model_card()` in `src/tt_kernel/build.py`** is fixed code, identical
   for every package regardless of `kind`. It emits the `tags:` frontmatter, the
   `# <name>` title, the hardware/mesh line, the `## Quickstart` heading with a
   **hardcoded `tt-model pull`/`tt-model serve`** block, the `## Serve profiles`
   table with **hardcoded LLM-shaped columns** (`max_num_seqs`/`max_model_len`)
   whenever there is more than one profile, and the `## Provenance` table. None
   of this is author-editable per model — a gap here needs a `tt-model-manager`
   code change, not different card text.

   The Quickstart command: `tt` (the `tenstorrent` PyPI package, i.e. "tt-cli")
   is real, published at v1.0.1, and `tt model pull`/`tt serve` already install
   and drive `tt-model pull`/`tt-model serve` themselves (checked
   `tenstorrent/tt-cli`, `src/tenstorrent/backends/serving/model_manager.py`).
   But the card must show **both** flows, `tt` first: `AGENTS.md` invariant 1
   ("tt-model alone must do the whole job; never introduce a step needing
   tt-cli") is binding on this repo, so a card naming only `tt` violates it.
   Note `tt model pull` has **no** `--with-weights` flag (only
   `--bundle`/`--weights-only`/`--offline`) — it pulls a bundle's weights by
   default; that flag belongs to the lower-level `tt-model pull`, which is why
   the two fences differ. Re-verify against the live `tt-cli` repo before
   relying on this, in case something's since moved.
2. **`CardSettings` in `src/tt_kernel/container_manifest.py`** is the only
   per-model input, and it has exactly two fields: `description` (a short lead
   paragraph, rendered right under the title) and `quickstart` (a markdown blob
   rendered right after the tool's own Quickstart block, before Serve profiles).
   `CardSettings` uses `extra="forbid"` — an author cannot invent a third field
   like `card.license`. Everything else in the template today has to be
   additional `## Heading` markdown packed by hand into `card.quickstart`, in
   the right order.

**Re-read both of those before relying on anything below.** This skill was
written against a specific snapshot of the code; later work may have since
added dedicated `CardSettings` fields or changed the hardcoded command, in
which case some "generator-owned" gaps listed here may already be fixed or
author-fixable. Don't carry a stale verdict forward.

## Step 1 — Get the template and figure out what you're checking

Read `reference/template.md` in this skill directory — the template as the
DX team proposed it. **Status: PROPOSED**, pending sign-off on the open
questions the team logged internally.
`license` is under an explicit A/B/C vote (stays optional /
required to list publicly / required to package at all), and who retrofits the
existing ~50 catalog cards is still unassigned — so treat both as unsettled.

Two questions are settled and recorded in the template's own status note: the
card shows both consumer flows with `tt` first (AGENTS.md invariant 1), and
whitelist status is not a card concern at all — a reviewed model is one copied
into the Tenstorrent HF org. Check the ticket for movement before treating any
other section as settled fact.

Then identify the input:

- **An authored `tt-model.yaml`, not yet built** — read its `card:` block (it
  may be absent entirely).
- **A locally staged/built package** — read `<out>/README.md`, the actual
  `render_model_card()` output for that manifest.
- **An already-published HF repo** — fetch it read-only, no auth needed for a
  public repo: `curl -s -L "https://huggingface.co/<repo>/raw/main/README.md"`.

## Step 2 — Score every template section

Go section by section through `reference/template.md` against what you're
checking. Classify each as:

- **OK** — present and matches.
- **Author-fixable** — missing or wrong, and belongs in `card.description` /
  `card.quickstart`.
- **Generator-owned** — missing or wrong, and current code cannot express it
  per-card. Cite the exact line you checked.

Most of what this skill originally listed as generator-owned has since been
built (PRs #129 and #133–#135). `CardSpec` now carries `description`,
`quickstart`, `architecture`, `status`, `intended_use`, `out_of_scope_use`,
`usage`, `performance`, `limitations`, `risks`, `licensing`, `related`,
`license`, `pipeline_tag` and `base_model`, and the renderer emits *At a
glance*, *Intended use*, *Quickstart*, *Serve profiles*, *Using it*,
*Licensing*, *Risks and safety considerations*, *Related packages*, *Feedback*
and *Provenance*. So a missing section is now usually **author-fixable**, not a
generator gap — which inverts the default this skill started with.

What is still genuinely generator-owned:

| Template section | What the generator does today |
| --- | --- |
| `tt_perf_summary` frontmatter | Not emitted at all. Deferred until the perf-target schema is settled |
| Model CI v0 row | Not emitted at all. Deferred until the model CI gate exists and produces a result worth citing |
| Changelog | No field and no section. `render_model_card()` rewrites the card wholesale on every build, so there is no history to draw on |
| Whitelist status | Not a card concern at all: a reviewed model is one Tenstorrent has copied into its own HF org (`tt-model whitelist` does the copy), so the repo it sits in *is* the signal. A `tt-whitelisted` tag in a card's frontmatter means nothing — anyone can add one to their own repo — and the renderer must never emit one |

Verify this table against the live source each time — don't trust it blindly.
It has already gone stale once, when the PR stack above landed.

## Step 3 — Interview for the author-fixable gaps

For everything classified author-fixable, ask the user for the missing
content in ONE batch (mirroring the `tt-model-yaml` skill's interview style)
— state your best guess and where it came from, so they are confirming
rather than composing from scratch. Then draft it as a single ready-to-paste
`card.quickstart` markdown block, sections in the template's order, each
under its own `## Heading` — since `CardSettings` has no separate field per
section today, all of it lands in one blob.

## Step 4 — Offer to write it

- **Unbuilt `tt-model.yaml`**: show the diff to the `card:` block and write it
  only after the user confirms.
- **Already built or published**: editing `card.quickstart` requires a
  rebuild and re-push (`tt-model package --container` then `tt-model push`)
  to take effect. Say so explicitly — do not attempt to patch a published
  `README.md` directly; the manifest is the source of truth, not the
  rendered file.

## Step 5 — Report

Two lists, and be exact:

- **Author-fixable** — what you drafted, and which manifest field it goes in.
- **Generator-owned** — what needs a `tt-model-manager` code change, each
  with the file:line you checked. Hand these to the team as follow-up work
  rather than treating them as resolved just because you found them.

## Notes

- Read-only against HF and Jira by default: never pushes, publishes, or
  comments unless the user explicitly asks.
- The template is not finalized. If a decision on one of the open questions
  lands (including posting the `tt-cli` resolution back to the ticket),
  update `reference/template.md` to match rather than letting this skill
  drift out of sync with the ticket.
