# One repo, several bundle formats: design proposal

> **Status: proposed.** Nothing here is implemented. This note records the design so it can be
> reviewed before any code lands. The implementation is planned as a short series of small PRs
> (see [Implementation plan](#9-implementation-plan)).

A model repo on Hugging Face can hold exactly one `tt-model` bundle today. This note proposes
the smallest change that lets one repo carry several **variants** of the same model: for
example a v5.1 container package and a v6 thin bundle side by side. A consumer picks one, and
an author can move a model from one format to the other without changing its repo id.

---

## 1. Why this matters

- **Migration.** An author moving a model from v5.1 to v6 today has two options. They can
  overwrite the v5.1 bundle, which breaks every consumer whose host has Docker but no SFPI. Or
  they can publish to a new repo, which splits the card, the Discussions page, the likes, and the
  catalog listing across two ids. Neither is a good outcome for a model people already use.
- **Different hosts.** A v5.1 package needs Docker and a Tenstorrent card. A v6 bundle needs SFPI
  and no Docker. Some users can run only one of them. One repo that offers both serves everyone.
- **Side-by-side checks.** An author should be able to serve the old and the new format on
  one box, benchmark both, and compare before switching the default.

## 2. What prevents it today

Four assumptions of "one repo = one bundle" are built into the code:

1. **One manifest slot.** Both formats write their manifest to the same root file,
   `tt_kernel_manifest.json` (`__init__.py`, `MANIFEST_NAME`). `pull` and `serve` read that
   one file to decide what the repo is, so the most recent push wins.
2. **`pull` downloads the whole repo.** `hub.download_bundle()` calls `snapshot_download` with no
   `allow_patterns`. Two formats in one repo would mean downloading both, including a multi-GB
   OCI image the consumer does not want.
3. **Local installs are keyed by repo id only.** A pulled container lives in
   `pulled/<org>__<name>` (`container_cli.pull_dir`), and a v6 install lives in
   `localdb[repo_id]` plus `models/<org>/<name>`. When both exist, `serve` checks the pulled
   container first. That is how a stale container shadowed a freshly pulled v6 bundle (fixed
   separately in #151).
4. **Pushes assume they own the repo root.** `push_folder` replaces `code/**` and `image/**`
   (`delete_patterns`), `_prune_removed` sweeps those two directories after a large-folder
   push, and every push rewrites `README.md` and the root manifest.

Branches look like an answer, and `pull org/name@branch` already works (`_split_revision`).
But `push` and `package-thin` always commit to `main`, the catalog lists only `main`, and the
card on `main` would name an install command that fetches the wrong format. See
[Alternatives](#10-alternatives-considered).

## 3. Goals and non-goals

**Goals**

- One repo id can carry several variants, each a complete self-contained bundle.
- Every `tt-model` release already in the wild keeps working, unchanged, against a repo that
  adopts variants: it sees the default variant exactly as it sees a bundle today.
- A consumer downloads only the variant they chose.
- Two variants of one repo can be installed, listed, served, stopped, and removed
  independently on one host.
- The design respects every invariant in [AGENTS.md](../AGENTS.md). In particular it adds no
  manifest schema (invariant 5), so `SUPPORTED_SCHEMAS` gating is unchanged.

**Non-goals (for the first version)**

- Choosing a variant automatically from the host's capabilities (Docker present, SFPI present).
- Listing variants separately in the community catalog.
- Variants on branches.
- Different weights per variant. All variants of a repo point at the same model; if the
  weights differ, it is a different model and belongs in a different repo.

Each non-goal can be added later without changing the layout below.

## 4. Hub layout

The repo root stays the **default variant**, byte for byte what a single-format repo looks
like today. Each other variant is a complete bundle in its own directory under `variants/`. A
small index file at the root names them all:

```
tt_kernel_manifest.json        default variant (unchanged: older clients read only this)
code/  image/  run.sh  ...     default variant's files
variants/
  v6/
    tt_kernel_manifest.json    a complete v6 thin bundle
    install.sh  run.sh  wheels/  requirements.txt  ...
tt_model_variants.json         the index (new)
README.md                      one card for the repo
```

The index:

```json
{
  "index_version": 1,
  "default": "container",
  "variants": {
    "container": {"path": ".", "schema": "5.1"},
    "v6":        {"path": "variants/v6", "schema": "6", "min_tt_model": "0.2.0"}
  }
}
```

Rules:

- **Variant names** use the same safe slug as a bundle `name` (`[a-z0-9][a-z0-9._-]*`), because
  they become part of local paths and container names.
- **Exactly one variant has `"path": "."`**, and it is the one named by `default`. Every other
  path is `variants/<name>`.
- **Each variant is self-contained.** Its manifest's relative paths (`code/`, `image/`,
  `wheels/`, `run.sh`) resolve against its own directory. No file is shared between variants,
  so the format code that reads a bundle does not change.
- **The index is advisory.** A repo with no `tt_model_variants.json` is a single-format repo,
  exactly as today. The index is a repo-level file, not a manifest, so no manifest schema
  changes and `SUPPORTED_SCHEMAS` still gates each variant's own manifest.
- **`min_tt_model`** (optional) names the oldest `tt-model` that can use the variant, so a
  newer client can tell an older one's user to upgrade instead of failing mid-install.

## 5. Consumer side

### Choosing a variant

```
tt-model pull  org/name                 # the default variant
tt-model pull  org/name --variant v6    # a named variant
tt-model pull  org/name@<rev>#v6        # same, with a revision, for tools that pass one string
tt-model serve org/name --variant v6
```

`#variant` is split off before repo-id validation (next to `_split_revision` in `cli.py`), so
a wrapper such as `tt` can forward the whole selection as one string. Whether `tt` accepts it
unchanged needs checking in that tool.

Resolution, in `hub.py`:

1. Fetch `tt_model_variants.json` for the revision (one small `hf_hub_download`). A 404 means a
   single-format repo: continue exactly as today.
2. With `--variant`, use it; an unknown name is an error that lists the available names.
3. Without it, use `default`. If this client cannot read the default's `schema`, or is older
   than its `min_tt_model`, fall back to the first variant it can use and say which one it
   picked. If none is usable, fail with a message naming the variants and what each needs.

### Downloading only one variant

`download_bundle(repo, rev, dest, *, path=".")` passes:

- `allow_patterns=[f"{path}/**"]` for a subdirectory variant, then treats that directory as
  the bundle root, or
- `ignore_patterns=["variants/**"]` for the root variant.

The container pull (`container_cli.pull_container`) and the v6 install path both go through
this one function, so both formats get the scoping for free.

### Local bookkeeping

A variant becomes part of an install's identity:

| | root (default) variant | named variant `v6` |
|---|---|---|
| `localdb` key | `org/name` (unchanged) | `org/name#v6` |
| container pull dir | `pulled/org__name` (unchanged) | `pulled/org__name#v6` |
| v6 install dir | `models/org/name` (unchanged) | `models/org/name#v6` |
| docker container name | unchanged | suffixed `-v6` |

Keeping the root variant's key unchanged means every existing install, and every open PR that
reads `localdb` by repo id, keeps working. Each entry also records a `variant` field.

- `list` shows the variant next to each install.
- `serve`, `stop`, `rm`, and `info` take `--variant` too. Without it:
  - **one install of the repo:** use it;
  - **several installs:** use the most recently pulled one and name it (the rule #151 adopts
    for the container-versus-v6 case), and print how to pick the other.
- The update check and `serve --refresh` compare the installed variant against the **same
  variant** on the Hub. If the root variant's schema changed since the install (because the
  author flipped the default), `serve` says so and offers `--refresh`. It never swaps formats
  silently.

### Weights are shared

All variants point at the same weights repo and revision. They should resolve to one download
on the host, not one per variant. This depends on where each format keeps its weights cache:
the container path uses the host Hugging Face cache, and #150 aligns the v6 `pull` with the
cache its `run.sh` reads. The variant work should follow whatever #150 settles, rather than
add a third location.

### Serving two variants at once

Ports already move apart: both paths walk upward from 20000. Chips need two fixes in progress
separately:

- a v6 `run.sh` that uses the chips it is granted instead of defaulting to chip 0 (#154);
- a container serve whose free-chip scan also sees host processes (a v6 server) that hold
  `/dev/tenstorrent/*` (#155).

`stop` also needs to reach a v6 server, which is a host process rather than a container (#152).

Until both land, run two variants side by side only on chips pinned by hand (`--device-id` for
the container, `TT_METAL_VISIBLE_DEVICES` for v6).

## 6. Publisher side

```
tt-model package --container tt-model.yaml org/name --variant container --default
tt-model package-thin org/name --variant v6 ...
tt-model push <staged-dir> org/name --variant v6
tt-model variants org/name                  # print the index
tt-model variants org/name --default v6     # switch the default
tt-model variants org/name --remove v6      # delete a variant
```

- **A named variant uploads into `variants/<name>/`.** Replacing and pruning (`delete_patterns`,
  `_prune_removed`) are scoped to that directory, so a variant push can never touch the root or
  another variant.
- **A root push keeps its current behavior** and leaves `variants/**` and the index alone. The
  current `delete_patterns` (`code/**`, `image/**`) do not match `variants/<x>/code/...`, and
  `_prune_removed` only considers top-level `code/` and `image/`. Both get a regression test so
  this stays true.
- **The index is updated in the same commit as the files it describes** (`create_commit` with the
  index added alongside the bundle's files, and `parent_commit` set to the revision it was read
  from). Two authors pushing at once then get a conflict instead of a lost update.
- **The first `--variant` push to an existing single-format repo** writes an index that names the
  current root bundle as the default. The variant name comes from `--root-name`, or from its
  schema: `container` for 5.1, `v6` for 6. Nothing already published moves.
- **Switching the default** moves the old default into `variants/<name>/` and the new one to the
  root in one commit, using `CommitOperationCopy` plus deletes. Large files are stored by
  content hash, so this re-points files on the Hub rather than uploading them again.
- **File ownership.** #149 refuses a push that would overwrite files the previous bundle did not
  own. With variants, "owned" is judged per variant: the root owns everything outside
  `variants/`, and each variant owns its own directory.

## 7. The model card

There is still one `README.md`, and it is rendered from the default variant, as today. Two
additions:

- A **Formats** section listing each variant with its schema, what the host needs (Docker, or
  SFPI), and the exact install command for each consumer flow (`tt` and `tt-model`).
- **Tags** become the union of every variant's tags, so a search for either format finds the repo.

A v6 bundle carries no rendered card today, so its row in the table is derived from its
manifest alone (schema, chip count, mesh). When #115's quickstart repointing runs on push, it
rewrites the commands in the Formats table the same way it rewrites the Quickstart.

## 8. Compatibility

| client | repo | behavior |
|---|---|---|
| existing release | repo without an index | unchanged |
| existing release | repo with an index | sees the root (default) variant only, unchanged; `variants/` is downloaded by its unscoped snapshot but never read |
| new release | repo without an index | unchanged |
| new release | repo with an index | full variant support |

One cost remains for existing releases: their unscoped `snapshot_download` fetches every
variant. The migration guide should say so, and authors should keep large images out of
non-default variants while many consumers are still on older releases. For a container
default, older clients already download the OCI image they need, so the added cost is only the
thin variant, which is small.

### Migration from v5.1 to v6

1. Publish v6 as a variant: `tt-model package-thin org/name --variant v6 ...`. Users of the
   container see nothing change.
2. Pull both variants on one box and serve them on separate chips. Run the same benchmark
   against both, then compare the output.
3. Switch the default: `tt-model variants org/name --default v6`. Set `min_tt_model` on
   the v6 variant so any client too old for it gets an "upgrade" message.
4. Keep the container variant for users without SFPI, or remove it once they have moved.
   While it stays, older clients pulling the new v6 default also download the container's
   image under `variants/` (see the cost above). Remove it, or accept that cost, once
   most consumers are on a release that scopes the download.

The repo id, card, Discussions page, and catalog listing stay the same throughout.

## 9. Implementation plan

Small PRs, each single-concern and independently reviewable, in this order:

1. **Read the index and scope the download.** `hub.fetch_variants`, the `path` argument to
   `download_bundle`, `--variant` and `#variant` on `pull` and `info`. Offline tests with a
   faked Hub, including a legacy repo and an unknown variant name.
2. **Key local installs by variant.** `localdb`, `pull_dir`, the models directory, the container
   name, and `list` / `serve` / `stop` / `rm` resolution. This rebases on #141 (`rm --all`)
   and #151 (serve precedence), whichever lands first.
3. **Push a named variant.** Scoped `delete_patterns` and pruning, the index write in the same
   commit, and `tt-model variants`. This builds on #149's ownership check.
4. **The card's Formats section and tag union.** Coordinated with #115.
5. **Docs.** `docs/publishing.md`, `docs/cli.md`, both end-to-end recipes, and a migration
   guide.

Hardware validation, once after step 3: publish a test repo with a container default and a v6
variant, pull both, serve both on disjoint chips, send a request to each, then switch the
default and repeat with an older client to confirm it still gets a working root bundle.

## 10. Alternatives considered

- **Separate repos per format** (`org/name` and `org/name-container`). This works today with no
  code change and is the recommended stopgap. The cost is splitting the card, Discussions,
  likes, and catalog listing, and asking consumers to change ids during a migration.
- **Branches** (`org/name@v6`). Pulling works today. But `push` and `package-thin` cannot
  target a branch, the catalog and Hub search see only `main`, and the card on `main` cannot
  describe a branch's install correctly. It would need changes in as many places as this
  design, with worse discoverability.
- **One manifest with several format blocks.** This is the most compact on the Hub, but it is
  a new manifest schema. Every existing release would refuse the repo (invariant 5), which
  breaks exactly the consumers a migration is meant to protect.

## 11. Open questions

1. Should the catalog list one entry per repo (proposed) or one per variant?
2. Should a new client pick a variant by host capability (Docker versus SFPI) when the user
   does not name one, or always take the default (proposed for the first version)?
3. Is `variants/` a safe reserved directory name? No published bundle is known to use it at
   the root, and `package` could refuse to stage one.
4. Should `whitelist` review a repo as a whole (proposed: the copy is a snapshot of every
   variant it contains), or each variant separately?
