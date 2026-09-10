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

Two rules make this predictable and safe:

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

## How compatibility is checked

`tt-model` records the target arch/machine in the bundle's `tt_kernel_manifest.json` and reports
a verdict on `pull`/`serve`/`info` (`compare()` in [`compat.py`](../src/tt_kernel/compat.py)).
`arch` mismatch is fatal — binaries are a different ISA; other mismatches (for example too few
chips for the chosen profile) are non-fatal and overridable with `--force`. Because each bundle
builds its own venv, there is no host `ttnn`/vLLM version to gate against; v5.1 container
packages use the same check, since an image cannot carry anything else worth gating on.
`tt-model info <id>` prints the manifest and the required-vs-detected verdict declaratively.

Older bundles (pre-v5 schemas) are refused: *re-publish the bundle with a current tt-model.*
