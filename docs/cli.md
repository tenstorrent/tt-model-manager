# CLI reference

Every command below is `tt-model <command> --help` away from its full flag list. Global options:
`-v/--verbose` shows full per-step output instead of the collapsed summary, `--no-color`
disables styling, `--version` prints the version. This page groups the commands the same way
`tt-model --help` does.

```bash
tt-model login                                    # reuses huggingface_hub's token store
```

## Run a model

```bash
tt-model serve you/mymodel                         # install if needed, then launch the OpenAI server
tt-model serve you/mymodel --print                 # print the exact launch command + env instead of running it
tt-model serve you/mymodel --local-only            # require an installed bundle; never hit the Hub
tt-model serve you/mymodel -- --extra vllm-arg     # anything after the id is passed through to vLLM

tt-model curl "hello"                              # send a chat completion to the running model
tt-model curl "write a haiku" --temperature 0.7 --max-tokens 200
tt-model curl "hello" --print                      # emit the equivalent curl instead of sending

tt-model stop you/mymodel                          # container packages: stop the running server (SIGTERM first)
tt-model logs you/mymodel                          # container packages: show the server logs
```

`tt-model serve <id>` is the one-command path. For an already-installed bundle it runs the
bundle's `run.sh` directly from that bundle's own venv — the host toolchain is irrelevant
because the bundle ships or builds its own. For a bundle that isn't installed yet, `serve`
downloads it, runs its `install.sh` to build the per-model venv, then serves. `run.sh` wires the
engine env and launches the OpenAI-compatible vLLM server. Repeat invocations skip the install
and go straight to launch. For a v5.1 container package, `serve` runs the image instead (see
[container_packages.md](container_packages.md), "Consuming and serving").

`serve` also compares the installed revision to the Hub's tip and prints a non-blocking
advisory if a newer one exists; skip it with `--no-update-check`, `--local-only`, or a pinned
`@revision`.

### Checking it answers

`tt-model curl` builds the chat-completions request for whatever is being served, so
verifying a bring-up doesn't mean hand-writing JSON and matching the model id exactly. The
model id comes from the running server (`GET /v1/models`); with nothing serving yet,
`--print` falls back to the installed bundle's weights id so it still emits something
pasteable. Any option the command doesn't reserve (`--print`, `--model`, `--base-url`) goes
straight into the request body, so the whole vLLM sampling surface is available.

## Get models

```bash
tt-model pull   you/mymodel                         # download + install the bundle into its own venv
tt-model pull   you/mymodel --with-weights          # ...and pre-download the weights (default: skip)
tt-model pull   you/mymodel --force                 # reinstall regardless / push past a compat warning
tt-model info   you/mymodel                         # manifest + compatibility verdict
tt-model search gemma                               # discover published bundles
tt-model search gemma --catalog                     # only bundles listed in the community catalog
tt-model search --arch blackhole                    # only bundles tagged for an arch
tt-model list                                       # locally installed bundles, and whether each can serve now
tt-model profiles you/mymodel                       # container packages: serve profiles + the default
```

A plain `tt-model pull <id>` reinstalls a stale bundle in place and reuses an up-to-date one;
`--force` is not the update path. The compatibility verdict printed by `pull`/`info` is
described in [publishing.md](publishing.md#how-compatibility-is-checked).

## Publish models

```bash
tt-model package      you/mymodel ...               # author + push a v5 fat bundle
tt-model package      --container tt-model.yaml     # build a v5.1 container package (stages a dir)
tt-model package-thin you/mymodel ...               # author + push a v6 thin bundle (BETA, unsupported)
tt-model push         build/mymodel                 # push a staged v5.1 container package (repo id from its manifest)
tt-model publish      you/mymodel                   # list a public bundle in the community catalog
tt-model unpublish    you/mymodel                   # delist (repo untouched)
```

Flags per format: [self_contained_packages.md](self_contained_packages.md) (v5),
[container_packages.md](container_packages.md) (v5.1), [thin_packages.md](thin_packages.md)
(v6). Visibility and catalog rules: [publishing.md](publishing.md).

## Maintenance

```bash
tt-model rm      you/mymodel                        # remove an installed bundle and its index entry
tt-model version                                    # print the installed tt-model version
```
