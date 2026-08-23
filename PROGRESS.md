# hous/refactor progress tracker

Plan: ~/.claude/plans/melodic-popping-valley.md
Watch this file: `cat /home/ttuser/dev/tt-model-manager/PROGRESS.md`

## 1. Branch + demolition
- [x] Branch `hous/refactor` created from `main`
- [x] Delete 16 modules, 27 test files, web/, scripts/, docs (14,131 deletions)
- [x] Rename `src/tt_kernel` → `src/tt_model`, no compat shims
- [x] New `pyproject.toml` (pyyaml, >=3.10, single `tt-model` entry point)
- [x] Committed: 4d46b77

## 2. Schema + types
- [x] `manifest.py` — schema 1, serve_profiles deep-merge, mesh_device validation
- [x] `types/base.py` — ModelType protocol
- [x] `types/vllm.py` — stock vLLM + standalone plugin; `vllm serve`
- [x] `types/vllm_legacy.py` — the fork + in-tree plugin; `run_vllm_server`
- [x] `types/__init__.py` — TYPES registry
- [x] `examples/laguna-xs-2.1.yaml` (vllm, 2 device-target profiles)
- [x] `examples/ornith-1.0-35b.yaml` (vllm-legacy, 2 deployment profiles)
- [x] Smoke test: both manifests load; serve_argv matches serve_vllm.sh / quickstart
- [x] Commit: 259710d

## 3. Docker
- [x] `hardware.py`, `oci.py`, `hub.py`, `container.py` written
- [x] `docker/Dockerfile` (builder + runtime stages, cache mounts, type hooks) — src/tt_model/docker/
- [x] `docker/entrypoint.sh` (exec, PID-1 SIGTERM)
- [ ] Hand-driven real laguna image build on this box (2.5-4 h)

## 4. build.py + package
- [x] Provenance resolution (local path / git ref → sha, scm version)
- [x] Stage code/ allowlist, manifest rewrite + built: block
- [x] Live streamed `docker build` + tee to log + tail cmd up front
- [x] Interrupt guard (double-Ctrl-C, cleanup, resume message)
- [x] `tt-model package` command

## 5. push/pull
- [x] `tt-model push` (tri-state visibility, upload_large_folder, model card)
- [x] `tt-model pull` (snapshot → docker load, weights → host HF cache)

## 6. serve/stop/logs/list
- [x] `tt-model serve` (profile selection, --print, --follow, exclusivity)
- [x] `tt-model stop` (SIGTERM-first, mesh reset only after SIGKILL)
- [x] `tt-model logs`, `tt-model list`

## 7. Tests + docs
- [ ] tests: manifest validation matrix, golden serve_argv strings ×2 types
- [ ] tests: docker run composition via --print, profile selection, stop semantics
- [ ] tests: interrupt guard against a fake child; oci round-trip; hub error map
- [ ] tests: fix kept test_cli_output.py targets
- [ ] docs/model_types.md, docs/packaging.md, rewritten README.md
- [ ] Final commit(s)

## 8. On-hardware acceptance (long; requires the box + HF auth)
- [ ] package laguna → image builds, verify RUN passes
- [ ] serve local, curl OK; stop is clean
- [ ] HF round-trip; self-containment checks
