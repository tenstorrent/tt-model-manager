# Validation protocol

Run every step on the target hardware, in order. Keep the outputs and logs of each step; a step
that was not run is reported as not run.

1. **Solo baselines.** Each member alone, with exactly the group's settings (same fit hooks, KV
   cap, context, env). LLMs: greedy (`temperature 0`) on a fixed prompt set (16 prompts), saved as
   JSON. Audio: fixed `seed`, saved files. Speech to text: a fixed input clip.
2. **Fit check.** `python -m tt_cohost.fit` in the built image on the target chips. Pass: verdict
   `shared`. Keep its report: per member and phase DRAM, L1 and programs compiled after capture.
3. **Group, sequential.** `tt-model serve build/<name>/tt_kernel_manifest.json --profile <p>`; send
   each member the same requests one member at a time. Pass: identical to its baseline (16/16
   text; byte-identical audio).
4. **Group, interleaved.** All members' requests at the same time. Pass: identical again. This is
   the check that members do not corrupt each other's traces or buffers. tt-cohost's
   `python -m tt_cohost.conformance` runs steps 1, 3 and 4 for one member next to a reference
   neighbour.
5. **Load.** The package's benchmark flags (greedy, exact input length, warmup) at 1, 8 and 32
   users: each member alone (the others idle), then all members busy at once. Every request must
   be served. Add one long-prompt run per member.
6. **Greedy after the load.** Repeat step 3. Programs compiled after trace capture can corrupt a
   parked trace; this is where it would show.
7. **Quality subset** for every member whose numerics a fit hook changed: an eval subset (for an
   LLM, GSM8K-chat and IFEval, 250 prompts each) with and without the hook, same harness.
8. **Soak.** 60 minutes of mixed load on all members at once. Pass: 0 failed requests, 0 device
   errors in the log, greedy still identical afterwards.
9. **Clean stop.** `tt-model stop`: a clean SIGTERM shutdown, the chips usable by the next boot.
10. **Clean-host consumer check** after `tt-model push`: tt-model with a fresh `HOME` (tt-model
    keeps its state and caches under `~/.cache/tt-model`), the local image tag removed, `tt-model
    pull` then `serve`. Pass: ready, outputs identical, the image id is the one the manifest
    records. Report the ready time including downloads and cold caches.

Report each step as a number from its output files ("16/16 identical", "byte-identical",
"1591 tok/s, 0 failed", "ready in 1322 s"), never as "works".
