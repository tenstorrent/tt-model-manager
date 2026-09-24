# Publishing a bundle: push, visibility, catalog, compatibility

Author on the box where you built/brought up the model, then push. The full authoring
recipe — every flag, the resulting repo layout, and the offline + hardware tests — lives in the
per-format guides: **[self_contained_packages.md](self_contained_packages.md)** (v5 fat),
**[container_packages.md](container_packages.md)** (v5.1 container) and
**[thin_packages.md](thin_packages.md)** (v6 thin, beta).

```bash
# v5 fat: embeds your built ttnn wheel + vLLM/plugin wheels + your tt-metal-community tree
tt-model package you/mymodel --public \
  --from-metal ./tt-metal-community \
  --ttnn-wheel dist/ttnn-*.whl \
  --arch-name LlamaForCausalLM \
  --main-class models.tt_transformers.tt.generator_vllm:LlamaForCausalLM \
  --weights unsloth/Llama-3.2-3B-Instruct         # POINTER — weights are not embedded

# v5.1 container: build the OCI image described by tt-model.yaml, then push the staged dir
tt-model package --container tt-model.yaml
tt-model push build/mymodel --public                # repo id comes from the manifest (--repo overrides)

# v6 thin: builds the venv from pip pins (ttnn / tt-metal-models) + bundled wheels
tt-model package-thin you/mymodel --public ...

tt-model pull you/mymodel                          # lay the bundle in + build its venv
tt-model pull you/mymodel --with-weights           # ...and also pre-download the weights
```

`--out <dir>` stages the running folder locally without pushing. Large wheels go to git-LFS
automatically on push.

## push vs. public vs. publish

These are three independent things — keeping them straight is the whole model:

| Concept | What it does | Flag |
|---|---|---|
| **push** | Upload the bundle's files to an HF repo. | `package` / `package-thin` / `push` |
| **public / private** | The repo's **visibility**. | `--public` / `--private` |
| **publish** | List the repo in the community **catalog** (a public pointer index). | `--publish`, or `tt-model publish` later |

Three rules make this predictable and safe:

- **Private by default.** `push`, `package`, and `package-thin` all create a **new** repo
  **private** — a bundle can point at proprietary weights, so nothing is ever made public by
  omission. Pass `--public` to share openly; pass `--private` to be explicit. Pushing to a repo
  that **already exists never changes its visibility** unless you pass the flag, and a change is
  always announced (`! Changed visibility …`).
- **`--publish` implies `--public`.** The catalog is a public index that can only see public
  repos, so publishing makes the repo public and lists it in one step — you never pass both flags.
  `--public` on its own makes the repo public **without** listing it (shared by link, not
  advertised). `--publish` combined with `--private` is a contradiction and is refused. Publishing
  is a pure pointer opt-in: the catalog stores none of your content and the repo stays under your
  governance.
- **A listing needs a card that says something.** A container package must carry
  `card.performance` and `card.limitations` to be listed — how the model performs and where it
  falls short are the two questions a stranger picking from the catalog asks, and the two only its
  author can answer. `package` warns when they are missing; `tt-model publish` and `push --publish`
  refuse, and so does a plain `push` of a bundle that is *already* listed (a re-push re-applies the
  listing, so it is held to the same bar — add the sections, or `tt-model unpublish` first).
  Pushing an unlisted bundle and serving are unaffected, so an experimental or private bundle is
  never blocked. Bundles published before cards were carried in the manifest are exempt from
  `tt-model publish` (there is nothing to read, which is not the same as nothing to say) and pick
  up the template on their next `tt-model package --container`. That exemption is for
  `tt-model publish` only: a **staged directory** from before card sections is refused on `push`,
  because there re-packaging is local and a manifest is a JSON file anyone can hand-edit — and if
  the listing should go rather than the sections be added, `tt-model unpublish` settles it in one
  command instead of a rebuild.

  The check fails closed: if the published manifest cannot be read, nothing is listed and nothing
  is made public. That includes a manifest that will not *parse* — most likely one pushed by a
  newer tt-model, which says so and asks you to upgrade rather than blaming the network.

  Known gap, tracked separately: the v5/v6 `package`/`package-thin` paths turn `--mesh`/`--arch`
  into repo tags without validation, so a value of `tt-model-catalog` there lists a bundle with no
  card check at all.

```bash
tt-model package you/mymodel               # private repo, not listed  (the default)
tt-model package you/mymodel --public      # public repo, not listed   (shared by link)
tt-model package you/mymodel --publish     # public repo, listed in the catalog (implies --public)
```

## Community catalog

Published bundles can opt into a searchable community catalog of community-published models.
The catalog — its web front end and its indexer — lives in a dedicated repo,
**[tenstorrent/model-manager-site](https://github.com/tenstorrent/model-manager-site)**
(formerly the `web/` directory of this repo). Every listing is a pointer to a public HF repo
that remains under its author's governance.

Listing is an explicit opt-in, separate from the push:

```bash
tt-model package   you/mymodel --publish   # push, make public, and list — one step (implies --public)
tt-model publish   you/mymodel             # list a repo pushed earlier (makes it public if needed)
tt-model unpublish you/mymodel             # delist (repo stays public)
```

`--publish` implies `--public` and adds the `tt-model-catalog` tag
(`TT_MODEL_CATALOG_TAG` in [`tt_kernel/__init__.py`](../src/tt_kernel/__init__.py)); the catalog
indexes only repos carrying it, and reads each repo's `tt_kernel_manifest.json` to render it.
`tt-model search <term> --catalog` restricts a search to listed bundles.

## Whitelisting (Tenstorrent reviewers)

The catalog is open — anyone can list a bundle, and that is the point. The **whitelist** is
the curated subset of it that a Tenstorrent reviewer has looked at, so a developer can pick a
model that is expected to work on the hardware it claims instead of sifting through everything
published:

```bash
tt-model whitelist   you/mymodel                 # copy a LISTED, public bundle into Tenstorrent/
tt-model unwhitelist Tenstorrent/MyModel         # withdraw the copy from the catalog
```

**Whitelisting is the copy.** A bundle under `Tenstorrent/` has been reviewed by definition,
because only the DX team can write that namespace (`TT_ORG` in
[`tt_kernel/__init__.py`](../src/tt_kernel/__init__.py)). There is no index to fetch and nothing
to keep in sync: "has Tenstorrent reviewed this" reduces to "is this repo ours", which any
consumer can answer from the repo id alone. It is deliberately **not a repo tag** and **not a
manifest field** — a tag lives in the author's own README frontmatter, which they can edit from
the Hub UI, and the manifest is written by whoever publishes the model, so neither can carry a
review someone *else* granted. Consumers see it as `tt model list --community --whitelisted`.

The copy is named after the **weights** repo, not the bundle: a bundle published as
`someone/qwen3-32b-blackhole-v51` with weights `Qwen/Qwen3-32B` becomes
`Tenstorrent/Qwen3-32B`. That is the canonical model name a reader is looking for rather than an
author's packaging slug.

What the commands guarantee:

- **The author's repo is never touched.** `duplicate_repo` is a server-side copy — it reads the
  source and writes only the new repo. A review costs one request and moves no bundle data, even
  for a multi-GB image.
- **The copy is a snapshot.** It records the source repo and the exact revision it was copied at,
  in its own card frontmatter. Later commits to the original are *not* covered by the review, and
  `Tenstorrent/...` does not track upstream.
- **The author is credited in the card's top line.** A whitelisted copy opens with a block naming
  the community author and linking both their profile and their original repo, above the model's
  own heading. That is the bargain: the copy earns this org's traffic, so the credit goes where a
  reader lands rather than under the fold. Re-recording a review replaces that block, so resuming
  an interrupted `whitelist` never stacks a second credit on the first.
- **The whitelist is a subset of the catalog.** `whitelist` refuses a repo that is not listed
  (listing is the author's decision — it will never `publish` on their behalf) and refuses a
  private one. The copy is then listed explicitly rather than relying on it inheriting the tag,
  so a public-but-unlisted source cannot produce a copy nobody can find.
- **`push` neither grants nor drops a review**, and `unpublish` has nothing to say about one: the
  copy is its own repo, so delisting an original leaves it alone. That independence is the point.
- **A name collision refuses.** Two bundles of the same model — two board targets, or two authors
  packaging the same upstream weights — derive the same name. The second is refused, naming the
  existing copy and what it was made from; it is never overwritten, because a reviewed artifact
  someone may be relying on is not ours to replace silently.
- **A half-finished run resumes.** If the copy landed but its review record did not, re-running
  re-records rather than being permanently blocked by its own partial state.
- **Withdrawing keeps the artifact.** `unwhitelist` delists the copy and leaves the repo, so
  anyone who pinned it can still reach it, and the original reappears in listings. Delete the
  repo by hand on the Hub if it should be gone entirely. It takes the **copy's** id — passing the
  community bundle it came from is refused.

## How compatibility is checked

`tt-model` records the target arch/machine in the bundle's `tt_kernel_manifest.json` and reports
a verdict on `pull`/`serve`/`info` (`compare()` in [`compat.py`](../src/tt_kernel/compat.py)).
`arch` mismatch is fatal — binaries are a different ISA; other mismatches (for example too few
chips for the chosen profile) are non-fatal and overridable with `--force`. Because each bundle
builds its own venv, there is no host `ttnn`/vLLM version to gate against; v5.1 container
packages use the same check, since an image cannot carry anything else worth gating on.
`tt-model info <id>` prints the manifest and the required-vs-detected verdict declaratively.

Older bundles (pre-v5 schemas) are refused: *re-publish the bundle with a current tt-model.*
