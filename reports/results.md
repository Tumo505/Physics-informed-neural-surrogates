# Executed research results

All results are computational. Released-model inference uses unedited author equations under modern Flax, not an identical legacy runtime. Graph connectivity alone does not establish FE mechanics.

**Executed extension studies (2026-10-16).** The experiments below are separate from the earlier pilots. Negative and failed outcomes are retained. The full research claim of robust autonomous prediction across shapes and trajectories is not established.

**Full-cycle extension in progress (2026-10-17).** Reference trajectories completed: 11/47, including the full-cycle convergence variants. The preregistered cohort has32 numerical geometries and44 passive trajectories:8 training,4 selection,20 calibration,4 primary test,4 loading-only shift and4 combined shape/loading shift. All steps use strict positive SNES reason, finite residual and positive endpoint-Jacobian gates. The failed10ms attempt is retained and excluded; the main study uses5ms. A bounded causal graph/POD recurrence is trained through its complete predicted200-step history, with matched data/physics refinement and a separate one-geometry limited-data regime. No final full-cycle neural performance or convergence claim is made until evaluation artifacts exist.

**Dalton architecture study and updated five-seed ensemble.** All2250 original training geometries; selection remains validation0:16. Three seed11 candidates share the160-epoch maximum schedule; early stopping can shorten actual runs. This screens candidate designs, not a population-level architecture ranking.

| Candidate | Graph nodes | Parameters | Best selection RMSE(cm) | Epochs completed |
|---|---:|---:|---:|---:|
| control_v4 | 106 | 138563 | 0.068385 | 141 |
| capacity_v4 | 106 | 665091 | 0.085062 | 160 |
| fine_graph_v4 | 848 | 138563 | 0.079712 | 141 |

Selected control_v4, then trained all five seeds. Same503 previously inspected test geometries: mean nodal L2 error 0.065169cm versus 0.065840cm for the previous ensemble (1.02% lower). Released-model reference remains0.041416cm. Paired mean change -0.000671385cm, conditional geometry-bootstrap95% interval [-0.0013095797621645033, -8.012990183488018e-05]. This is a repeated benchmark comparison, not a new confirmatory test. The accuracy gap to the released model remains.

Updated uncertainty/error Spearman correlation 0.8961. Historical calibration134 is reused without coverage-based selection. Nominal90/95% joint-geometry coverage: 471/503 (93.64%), radius0.4025cm, 480/503 (95.43%), radius0.4541cm.

![Architecture and repeated benchmark comparison](figures/graph_study.png)

**New mechanics-complete numerical geometry cohort.** 64 independent uniform parameter draws within±5% of the documented ellipsoid radii, plus4 predeclared±10% extrapolation corners. This is a controlled numerical shape distribution from ONE benchmark family, not an anatomical or physiological population. The long axis is X as verified in the source. The original source microstructure algorithm is rerun on every scaled geometry. Exact affine nodal correspondence and unchanged tetrahedral indices are verified.

272 startup simulations converged on 68 geometries, four prescribed loads each, dt1ms, zero histories/activation. All4577 nodes and17625 tetrahedra retained. Minimum reference endpoint detF 0.964086; zero inversions; maximum normalized reference force RMS 1.02e-11; no exact duplicate parameter draws. Small perturbations deliberately create closely related shapes.

Geometry splits:20 training(80 loads),8 selection(32),20 calibration(80),16 test(64),4 extrapolation(16). Every load from a geometry stays in its split. This cohort has no historical calibration/test reuse. Train-only normalizers and rank12 POD use only the80 training fields. The graph encoder has64 pooled nodes, width32 and two message-passing rounds; a geometry/load-conditioned neural head predicts POD coefficients. Correspondence is specific to this affine family.

Five matched seeds,800 optimizer updates per model. Physics lambda screened at1/100 on seed11, then frozen at 1 using trusted float64 validation residuals subject to a10% displacement-RMSE constraint. Optimization/checkpoint scoring used the verified float32 GPU residual; final evaluation uses float64 arithmetic and trusted FE assembly. The CPU pilot and interrupted screening arm are preserved in cpu_geometry_pilot.json.

| Five-member ensemble | Mean geometry component RMSE(µm) | Mean normalized force RMS | Mean element strain error | Mean relative elastic-energy error |
|---|---:|---:|---:|---:|
| data | 0.256068 | 0.00016867885 | 6.82669e-05 | 0.0160336 |
| physics_lambda1 | 0.256391 | 0.00016888652 | 6.83458e-05 | 0.0160697 |

H1 is not supported in this startup study. Physics/data displacement-RMSE ratio 1.001262; paired mean normalized residual difference(physics minus data) 2.0767e-07, conditional geometry-bootstrap95% interval [3.305252562804678e-08, 3.5890349685049275e-07]. The approximately0.12% residual difference is small; it does not show a useful physics benefit. Bootstrap intervals condition on trained models and exclude retraining/implementation uncertainty. Individual matched-seed outcomes are in the JSON.

**Natural-boundary ablation.** All DOFs are free. Interior and boundary-DOF contributions partition the SAME full assembled weak residual, including inertia, viscosity, Robin terms and follower pressure; their losses sum exactly when normalized by total DOFs. Boundary-DOF loss is not a claim that external traction alone should vanish. Ablations use one matched seed only.

| Seed11 residual-loss region | Test component RMSE(µm) | Mean normalized FULL residual RMS |
|---|---:|---:|
| interior | 0.227562 | 0.00015266148 |
| boundary | 0.227851 | 0.00015292558 |

**Independent geometry calibration.** The simultaneous score is the maximum nodal L2 displacement error over ALL nodes AND ALL four loads in a geometry. Corrected split-conformal quantiles use20 calibration geometries. Exchangeability refers only to the defined uniform numerical parameter distribution.

| Ensemble | Nominal | Joint test coverage | Radius(µm) | Exact binomial95% interval |
|---|---:|---:|---:|---|
| data | 90% | 15/16 (93.75%) | 2.8578 | [0.6976792615654681, 0.998418888277234] |
| data | 95% | 16/16 (100.00%) | 3.4575 | [0.7940927857921773, 1.0] |
| physics_lambda1 | 90% | 15/16 (93.75%) | 2.8592 | [0.6976792615654681, 0.998418888277234] |
| physics_lambda1 | 95% | 16/16 (100.00%) | 3.4581 | [0.7940927857921773, 1.0] |

The16 test geometries give wide coverage intervals. Raw variance/error Spearman correlation was0.75 for both ensembles; selective-risk results are in the per-model JSON. This does not make raw variance a calibrated interval.

**Observed shift failure:** both nominal90% and95% balls covered0/4 extrapolation geometries for both models. Mean nodal error increased from about0.35µm in distribution to3.64µm at the shifted corners. The four deterministic corners have no exchangeability-based coverage guarantee and do not support a population-level shift estimate. H3 is not supported.

![Geometry mechanics and uncertainty comparison](figures/geometry_study.png)

![Actual FE boundary prediction and error](figures/geometry_prediction.png)

**Controlled refinement and fresh confirmation cohort.** The earlier v2 test remains reported above. Each of five original data-trained checkpoints now initializes BOTH matched refinement arms. Each arm receives400 updates at learning rate1e-4 to1e-5. Physics uses four mixed geometry/load cases per update; a training-only gradient diagnostic registered lambda100/1000/10000. Original training20/selection8 geometries are reused; no new target affects training or selection.

Validation selected lambda100 under the predeclared10% displacement constraint. This selects a physics candidate even if data-only has a lower validation residual; it does not itself demonstrate an advantage. All five models were frozen before evaluation. Fresh seed271828 supplies20 calibration and16 test geometries within the same numerical±5% distribution. Another16 random draws lie outside that support in the±5–10% corner regions. Original training-only POD, pooling and normalizers are reused without refitting. These are new numerical geometries, not independent anatomical sources.

All208 additional FE solves converged; minimum endpoint detF0.964567, no inversions, maximum reference force RMS9.82e-12.

| Refined five-member ensemble | Fresh test RMSE(µm) | Normalized force RMS | Mean nodal L2(µm) |
|---|---:|---:|---:|
| data | 0.210079 | 0.00013249044 | 0.292581 |
| physics_lambda100 | 0.216911 | 0.00013563106 | 0.301911 |

Physics/data RMSE ratio1.032523; paired force-RMS difference3.14062e-06, conditional geometry-bootstrap95% interval[2.1356860408155457e-06, 4.148290824321959e-06]. Descriptive H1 criterion satisfied:False. This criterion requires a lower mean residual and no more than10% RMSE degradation; it is not a population-level significance test. Geometry bootstrap conditions on five fitted members; matched individual-seed results are retained in the JSON.

| Refined ensemble | Nominal | Fresh joint test coverage | Radius(µm) | Shift coverage |
|---|---:|---:|---:|---:|
| data | 90% | 16/16 | 2.2773 | 0/16 |
| data | 95% | 16/16 | 2.6691 | 0/16 |
| physics_lambda100 | 90% | 16/16 | 2.2999 | 0/16 |
| physics_lambda100 | 95% | 16/16 | 2.8037 | 0/16 |

Coverage refers jointly to all4577 nodes and all four startup loads in a geometry; exact binomial intervals and selective-risk curves are in the evaluation JSON. Only16 test geometries keep inference imprecise. No conformal guarantee extends to the shifted draws.

![Fresh refined geometry comparison](figures/geometry_refinement_study.png)

![Fresh FE surface prediction](figures/geometry_refinement_prediction.png)

**Full passive reference trajectory and temporal sensitivity.** A pressure-only variant of the pinned dynamic benchmark, with P1 displacement/fibers, zero activation and initial u/v/a=0. Generalized-alpha histories are updated after every converged solve; pressure is evaluated at the source intermediate time. This differs from the default active/P2 benchmark.

999/999 steps completed over0.001–0.999s, dt1ms; minimum endpoint detF 0.629753, zero inverted elements across all stored states. Measured FE trajectory wall time 1477.2s. The40ms endpoint volume-L2 differences versus dt0.5ms were 0.00620% for dt1ms and 0.03329% for dt2ms. This short-interval check does not prove full-cycle timestep convergence. Earlier two-level mesh sensitivity also does not establish mesh independence.

![Reference trajectory and timestep sensitivity](figures/trajectory_validation.png)

**Temporal neural pilot, with explicit rollout failure.** One fixed geometry and one cycle. Train first600 timesteps, select on next200, evaluate last199. A64/64 MLP with train-only rank16 POD predicts acceleration increments from causal displacement/velocity/acceleration histories and prescribed pressure. Exact Newmark updates produce the displacement prediction. Matched seed11 and400 updates per condition. Time frames are dependent: no geometry/trajectory confidence interval or conformal claim is made.

The original lambda1/100 grid had little effect. A training-only parameter-gradient diagnostic on eight training frames registered the balanced grid1e5/1e6/1e7, then validation selected1e6. The same temporal block had already been evaluated by the original pilot; the balanced comparison is exploratory repeated temporal evaluation. No test outcome selected the grid.

| Model | True-history one-step component RMSE(µm) | Mean normalized force RMS | Recursive valid steps /199 | First invalid-J time(s) |
|---|---:|---:|---:|---:|
| data | 0.242987 | 6.7499844e-05 | 28/199 | 0.8290000000000001 |
| physics_lambda1e+06 | 0.221051 | 5.1236249e-05 | 25/199 | 0.8260000000000001 |

With true histories supplied, balanced physics weighting reduced conditional one-step displacement error by about9.0% and force residual by24.1%. BOTH recursive rollouts failed with inverted elements: physics after25 valid steps and data-only after28. Valid-prefix errors use different horizons and must not be interpreted as successful or directly comparable complete rollouts. The temporal model is not a validated autonomous full-cycle predictor. Scaling to a multi-geometry trajectory cohort is a no-go until closed-loop stability is addressed.

![One-step gains and recursive failures](figures/temporal_study.png)

**Matched speed and cost.** CPU float64, two threads, same P1 startup task, four held-out numerical geometries×four loads. Neural inference returns the full4577×3 field.

Warm FE median 998.184ms, five-member mean plus coordinate sample variance median 3.639ms: 274.3× for this startup task. Batch-one repetitions:32 timed FE solves and800 ensemble calls. No full-cycle, Dalton or clinical speedup follows. Including warm input-file reads, graph pooling and normalization, mean/variance inference costs 4.956ms (201.4×). Model load and geometry activation are recorded separately in geometry_runtime.json. Fiber-field generation is excluded from the resident-input inference comparison.

New cohort generation wall time 741.5s; GPU study training/validation grid plus ablations 394.2s. Measured partial-cost recovery about1143 comparable queries. This excludes the CPU pilot, verification, temporal experiments, preprocessing and software setup; it is not whole-project break-even.

Fresh-cohort refinement timing (selected physics-lambda100 five-member ensemble): mean plus variance 3.732ms with resident normalized inputs, or 5.178ms including warm file reads and input processing; matched warm FE 1017.522ms (196.5× including input processing). Same CPU float64/two threads;32 timed FE solves,800 resident ensemble calls,160 input-inclusive calls. Extra cohort generation258.3s and refinement screening/training/validation277.2s. This additional-cost accounting excludes the parent v2 training/corpus and all other research costs. Lower latency does not remedy the negative physics/shift results.

Physics-only backpropagation through an actual trained graph/POD model was verified: parameter-gradient norm 6.29037e-06; relative difference versus native float64 VJP through the same network graph 3.32e-05. Independent assembly checks include nonzero histories and a strongly deformed state. GPU optimization uses float32 after controlled precision checks; final mechanics uses float64.

**Go/no-go and reproducibility.** No external data/solver blocker remains for these numerical studies. The remaining scientific limitations include the released-model accuracy gap, the study-specific physics outcomes above, shift undercoverage and unstable recursive dynamics. The next experiment should train for closed-loop stability and test on independent trajectories before scaling the full trajectory cohort. Independent anatomical sources and full-cycle mesh/timestep convergence remain unvalidated.

Reproduce the extensions with `make graph-study`, `make geometry-study`, `make temporal-study`, `make geometry-confirmation` and `make report` after the original data/environment setup. Runtime requires the legacy solver environment; MPS training requires Apple GPU access. Configs, split IDs, checkpoint hashes, per-geometry/seed results and failed-rollout artifacts are retained. Large datasets/checkpoints/logs remain ignored.

Official conference requirements were verified and checksummed in experiments/manifests/conference.json: [CMBBE2027 submission page](https://www.cmbbe-symposium.com/2027/abstract-submission/). The mandatory template is one A4 page in two columns, including figures/references; deadline15November2026 at23:59CET. No manuscript or submission was created, respecting the user request to exclude paper/.


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
