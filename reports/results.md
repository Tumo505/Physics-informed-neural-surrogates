# Executed research results

All results are computational. Released-model inference uses unedited author equations under modern Flax, not an identical legacy runtime. Graph connectivity alone does not establish FE mechanics.

Released GNN: 567 held-out geometries, mean nodal L2 error 0.041037 cm.

Original 64-training-case pilot ensemble: mean nodal error 0.479376 cm on test0:64. Independent members were worse than their mean; the pilot gap reflects data, architecture and training-budget differences, not evidence that averaging hurts.

| Model | Main test geometries | Mean nodal L2 (cm) | Mean case component RMSE (cm) |
|---|---:|---:|---:|
| full_gnn_ensemble | 503 | 0.065840 | 0.041815 |
| full_point_baseline | 503 | 0.090279 | 0.056794 |
| full_gnn_seed11 | 503 | 0.097570 | 0.061320 |
| full_gnn_seed22 | 503 | 0.071563 | 0.045292 |
| full_gnn_seed33 | 503 | 0.086597 | 0.054750 |
| full_gnn_seed44 | 503 | 0.071306 | 0.045153 |
| full_gnn_seed55 | 503 | 0.072086 | 0.045652 |
| released_gnn_same503 | 503 | 0.041416 | 0.026530 |

Main evaluation uses test64:567, excluding the previously inspected pilot cohort. Full-data models train on all2250 original training cases. Checkpoint selection uses validation0:16; calibration uses16:150. Historical calibration reuse is disclosed; no coverage outcome selects a model. The same previously inspected64 pilot cases improved from0.479376cm to0.061811cm; this is a secondary same-case comparison, separate from the503-case main test.

| Nominal whole-geometry coverage | Empirical | Radius (cm) | Exact binomial95% interval |
|---|---|---:|---|
| 90% | 476/503 (94.63%) | 0.41981 | [0.9228595621530414, 0.9643315635531253] |
| 95% | 481/503 (95.63%) | 0.47805 | [0.9345273159783736, 0.9723901180891344] |

Ensemble raw-variance/error Spearman correlation: 0.8855. Geometry bootstrap intervals condition on the fixed trained ensemble and exclude retraining uncertainty. Paired comparisons, input-defined shape-tail analysis and per-case values are in full_evaluation.json. Exploratory shape-tail cases had90.5% coverage for nominal95% bounds (n=84), versus96.7% in central shapes (n=419). No shifted-distribution coverage guarantee.

Validation-only query-density experiment: one matched seed, same starting weights and 60 continuation epochs on all2250 training geometries; compare128 versus1024 nodes per geometry. Starting checkpoint remains eligible for selection. Calibration and test arrays were not opened.

| Candidate | Selected validation component RMSE(cm) | Selected continuation epoch |
|---|---:|---:|
| initialization | 0.0786016 | 50 |
| 128 | 0.0786016 | -1 |
| 1024 | 0.0786016 | -1 |

Improvement beyond relative numerical tolerance1e-6: False. More point queries did not establish an accuracy benefit in this continuation; this does not rule out a benefit when training from scratch or with a different architecture. Existing ensemble retained. These16 geometries are the reused selection set, so results are descriptive. Future comparisons on the already evaluated503-case test set must be labelled repeated benchmark evaluation.

![Validation-only query-density comparison](figures/query_density.png)

Two-level cardiac mesh sensitivity: uniform tetrahedral refinement of the same polyhedral domain, retaining boundary labels and interpolating the same continuous P1 fiber fields; original mechanics and timestep retained. This checks displacement discretization, without curved-geometry refinement or independent anatomy.

| Pressure(Pa) | Relative volume-L2 displacement difference | Volume-weighted component RMSE(µm) | Minimum refined endpoint detF |
|---|---:|---:|---:|
| 4017.74 | 2.815% | 1.090 | 0.994296 |
| 8035.48 | 2.820% | 2.190 | 0.988481 |
| 16070.95 | 2.827% | 4.412 | 0.976539 |

Mesh sizes: 17625 to 141000 tetrahedra. Boundary areas and reference volume match to numerical tolerance. Two levels do not establish an asymptotic convergence rate or a mesh-independent truth. Surrogate error against the original mesh does not include this reference discretization uncertainty.

![Two-level FE mesh sensitivity](figures/mesh_sensitivity.png)

Trusted cardiac first-step probe now converged: normalized residual 5.24e-11, minimum detF 0.9999762, inverted elements 0%. This is oneP1 dynamic timestep, not a full cardiac trajectory or matched Dalton FEM timing. The former native crash was isolated to PETSc LU option initialization; the workaround preserves governing forms.

Separate cardiac startup experiment:20 training,6 validation and6 test LOAD cases on ONE geometry; P1/fibers, passive anisotropic viscoelastic dynamics, zero initial histories. This is not a full-cycle benchmark or a shape-generalization result.

| Matched graph/POD model | Test component RMSE(m) | Normalized force RMS | Strain Frobenius error |
|---|---:|---:|---:|
| data_only | 5.92726e-07 | 0.000111396 | 0.000111019 |
| physics_lambda1 | 5.92969e-07 | 0.000111474 | 0.000111061 |

Physics weight selected on validation only: physics_lambda1; within10% displacement degradation constraint: True. One matched seed,500updates/model; no population-level significance claim. Full weak residual includes follower pressure, inertia, viscosity and natural epi/base Robin terms. POD rank4 is fitted on training cases only.

No residual improvement observed in this matched fixed-geometry startup pilot; do not claim H1 support.

Matched cardiac startup CPU/float64 timings: surrogate median 16.645ms; warm FEM median 1026.367ms; ratio 61.7×. Same host and two-thread settings; inference/corpus/training costs separate. This ratio applies only to the one-step fixed-geometry task and its measured errors, not Dalton or full cardiac cycles.

Full-node Dalton inference timing (batch1,100repetitions; inverse normalization included; resident preprocessed inputs):
- cpu: one model 3.363ms median; five-member ensemble 16.703ms median. No matched Dalton FEM speedup.
- mps: one model 1.381ms median; five-member ensemble 5.914ms median. No matched Dalton FEM speedup.

Observed training cost: full-data graph ensemble stages summed 1083.5s; cardiac data/physics validation grid 406.0s plus summed corpus solver steps 39.4s. These costs are separate from latency ratios and exclude installation/diagnostics.

![Released prediction on an actual graph](figures/reference_prediction.png)

![Full-data results](figures/full_results.png)
