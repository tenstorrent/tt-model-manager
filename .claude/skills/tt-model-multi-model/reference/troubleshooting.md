# Failure → fix

| what you see | cause | fix |
| --- | --- | --- |
| a serve hangs at device open, no error | another process holds the chip (UMD's per-chip lock; a second opener waits) | `tt-model ps`; pick free chips or stop the holder. Never two servers on one chip |
| `deploy up` refuses: chip in use / port busy / duplicate name | the plan collides with a running server or itself | change `chips:`/`port:`/`name:`, or stop what holds them; `--print` shows the plan |
| a 2-chip profile lands on chips 1,2 | an old tt-model without whole-board placement | pin `chips: [2, 3]` (or `[0, 1]`) |
| a chip another user holds is planned as free | rootless Docker: their containers and `/proc/<pid>/fd` are invisible to you | pin `chips:` to the chips you own |
| a 4-chip mesh fails with `Fabric Router Sync: Timeout ... did not complete the remote (ethernet) handshake`, or hangs in the first all-reduce | an earlier dirty stop reset only some chips | reset every chip (`tt-smi -r`) once nothing uses them; a current `tt-model stop` resets the free ones itself |
| `deploy up` stopped everything | one instance failed to start or to become ready, so the deployment rolled back | `tt-model logs <org/name> --name <instance>` from the failed instance; fix, then `up` again. `--keep-going` keeps the healthy ones |
| a co-host instance's second or third model unreachable | another service sits in its port block | give the instance a base port with room for every member port plus the status port |
| two models on the same chips from two packages | not possible: one process per chip | one co-host package with both models (`tt-model-cohost-package` skill) |
| throughput far below the card | sampled or variable-length benchmark | the package's benchmark flags: greedy, exact input length, warmup |
| outputs differ between "alone" and "together" on separate chips | not seen so far; treat as a bug | keep both outputs, the deploy file and logs; report it |
