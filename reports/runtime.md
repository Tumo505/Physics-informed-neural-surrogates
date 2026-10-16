# CPU inference benchmark

Apple M5 Pro, float32, batch one, two PyTorch threads, 6784 nodes, 37056 graph edges. Modern pilot seed11 only (five-member uncertainty needs five passes). Ten warmups and 100 timed repetitions. Run with make benchmark in isolation; values vary with system load.

- Inference incl. inverse target normalization: median 12.118 ms, p95 13.511 ms.
- Resident-case preprocessing median 1.432 ms.
- Resident-case end-to-end median 12.777 ms, p95 13.959 ms.
- Model load 0.00327 s; case load 0.00531 s.

No matched Dalton FEM solve is available. No FEM speedup, amortization or break-even is claimed. Full reference GNN timings in reference_varying.json include JAX input transfer/materialization and inverse scaling, exclude preprocessing and are descriptive under concurrent CPU work, not an isolated speed comparison.
