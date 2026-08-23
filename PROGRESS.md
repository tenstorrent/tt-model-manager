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
- [x] Real laguna image built + in-image verify PASSED (attempt 6; 10.4 GB, 2.2 GB as OCI blobs)

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
- [x] tests: manifest validation matrix (35), golden serve_argv strings ×2 types (5)
- [x] tests: docker run composition via --print, profile selection, stop semantics (16)
- [x] tests: interrupt guard (8); oci fake-docker round-trip (5); hub error map (16)
- [x] tests: kept test_cli_output.py retargeted (38) — 129 passing total
- [x] docs/model_types.md, docs/packaging.md, rewritten README.md
- [x] 16 commits; suite green (130)

## 8. On-hardware acceptance (long; requires the box + HF auth)
- [x] package laguna → image builds, verify RUN passes (imports, CPU torch, plugin registration, precision-config, SDPA)
- [x] serve local: READY in ~16 min, coherent completion (55 in / 32 out, finish=stop),
      ~29 tok/s incl. TTFT; `stop` = clean SIGTERM shutdown in 1.3 s, no mesh reset needed
- [x] two more field-found bugs fixed: sfpi cc1plus host libs (libmpc/mpfr/gmp/zstd),
      tt-owned metal tree (tracy mkdir on import), tests/ on laguna's serve path,
      and the PLUGIN DRIFT: main moved f3e4637a→e3fc849 and broke first-decode —
      the manifest now pins the validated sha (the whole point, proven on hardware)
- [x] self-containment: no host paths in image env/labels; USER tt; host pip has no
      vllm/ttnn; weights only via the mounted host HF cache
- [!] HF round-trip: BLOCKED — the stored HF token is role=read ('llama'); push needs a
      WRITE token. Run `hf auth login` with a write token, then:
      tt-model push build/laguna-xs-2.1 && docker image rm tt-model/laguna-xs-2.1:9b415f820
      && tt-model pull tt-hous/laguna-xs-2.1 && tt-model serve tt-hous/laguna-xs-2.1
- [!] Ornith on-hardware: BLOCKED — 77 GB gated weights not in cache, and the branch
      agentic-research/hous/ornith-1.0-35B is not visible on the public tt-metal remote.
      (Its manifest, type impl, and golden launch-command tests are done and green.)
