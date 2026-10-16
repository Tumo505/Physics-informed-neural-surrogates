# CODEX MASTER IMPLEMENTATION BRIEF

## Project
**Physics-Informed Neural Surrogates for Rapid and Uncertainty-Aware Prediction of Left Ventricular Deformation**


**Scope:** Computational simulation and numerical experiments only. No wet lab, patient-specific clinical claims, or fabricated results.

---

## 0. Your operating contract (read first)

You are the lead computational research engineer collaborating with a cardiac biomechanist and machine-learning researcher. **Implement and run the experiment**, do not merely produce a plan or an attractive visualization. Make real, testable code, execute each attainable stage, and record evidence. Work autonomously through reversible steps. For costly downloads, verify disk and access first. For truly blocking external limitations (network unavailable, absent FEM geometry, incompatible legacy dependencies), document precisely and continue with the best valid alternate track; never silently substitute an easier but scientifically invalid result.

At the beginning:
1. Inspect the current repository and operating system, available CPU/GPU/RAM/disk, package manager, and existing data; preserve any user work. Do not assume the machine has CUDA. On macOS, check MPS/CPU support; on Linux, check CUDA.
2. Create `docs/STATUS.md` containing dated task progress, commands actually run, versions, errors, outcomes, remaining blockers, and immediate next actions. Update after each milestone.
3. Create `docs/SCIENTIFIC_DECISIONS.md` recording every important assumption, dataset availability finding, units, material model, boundary condition, loss scaling, and why the model is valid.
4. Set up reproducibility: Git commits, pinned dependencies/lockfiles, config files, deterministic seeds (as practical), manifests with checksums and source URLs, run metadata, and a smoke-test path.
5. Follow the acceptance gates below. Never claim an experiment is complete if only toy/generated demonstration data ran.

**Non-negotiables**:
- No made-up empirical scores, speedups, calibration, statistical significance, physical material parameters, medical findings, citations, patient cases, or FEM connectivity.
- Preserve original train/validation/test partitions; never fit normalizers on validation/test or use test results for model selection.
- Interpret each LV geometry/simulation as an independent unit for data splitting and statistics; audit possible repeated geometries, near duplicates, and fixed-geometry variants.
- A graph edge is **not** a tetrahedron/hexahedron. Edge smoothness is **not** a valid 3D hyperelastic equilibrium residual. If FE element connectivity, integration rule, boundary conditions, loads, or constitutive coefficients are unavailable, stop physics-equilibrium implementation on that data and switch to a mechanically complete benchmark or solver-generated cohort.
- Distinguish: (a) reconstruction/prediction of 3D nodal displacements, (b) valid FE-derived strain/stress, (c) solver-verified mechanics, (d) uncertainty indicators, and (e) empirically calibrated uncertainty.
- “Rapid” means measured inference runtime against matched FEM solution time on defined hardware/task. Training cost is separate; report amortization/break-even if appropriate. Do not infer speedup from paper timings on different hardware.
- Label illustrations, synthetic tests, and pilot results as such, never as medical/clinical validation.
- Do not assert complete physiological electromechanics; the primary Dalton dataset concerns passive diastolic LV deformation. Do not claim systolic contraction, active mechanics, regeneration, or patient-level digital twin capability.

## 1. Scientific question and contributions

### Primary question
Can a physics-informed graph neural surrogate predict the *3D displacement field and resulting left-ventricular deformation* more quickly than a matched finite-element solve, while maintaining reasonable displacement accuracy, improving physical consistency, and providing empirically evaluated uncertainty on unseen cases?

### Secondary questions
- **RQ1 physical validity:** When a valid residual can be computed, does physics guidance reduce nonlinear mechanical-equilibrium/boundary residuals at comparable displacement error?
- **RQ2 reliability:** Do ensembles and geometry-level calibration provide informative, calibrated error intervals in distribution? How do they behave under shift?
- **RQ3 generalization:** What happens under held-out ventricular shapes and, only if varied reference simulations exist, unseen material/loading conditions?
- **RQ4 speed:** What are end-to-end and inference-only speedups, including hardware context and failure cases?

### Falsifiable hypotheses
H1: Mechanics-informed training reduces normalized equilibrium residual compared with a matching data-only model without unacceptable displacement degradation.  
H2: Calibrated ensembles achieve approximately nominal empirical coverage on exchangeable held-out geometries and uncertainty is predictive of error.  
H3: Physics guidance may improve robustness to geometry/material/load changes, to be tested rather than assumed.  
H4: Inference is faster than a matched FE solve; benchmark before claiming.

**Novelty requirement:** Dalton et al. already demonstrated GNN emulation of passive LV mechanics. A replica alone is not original research. Contribution must be controlled mechanics validation plus reliable uncertainty (and ideally robust speed/shift evaluation) beyond the original emulator. Conduct a compact literature review of related physics-informed GNN/PINN/operator methods and uncertainty calibration; record citations/novelty map in `docs/RELATED_WORK.md` with actual source DOIs and access dates.

## 2. Verified sources, acquisition and licensing

**A. Primary data, Dalton et al. (2022):**
- Zenodo record and DOI: https://zenodo.org/records/7075055 ; DOI 10.5281/zenodo.7075055
- Download: archive `lvData.zip`, about 1.8 GB, contains approximately 3,000 synthetically generated LV geometry simulations of forward displacement from beginning to end of diastole, with author training/validation/test splits.
- SHA/MD5: verify archive against the MD5 displayed in Zenodo metadata (`96987da42e159c205fa4aeb30cd5526e`) and record computed checksum. Inspect archive without extracting blindly; protect against zip traversal; estimate expanded disk requirement.
- It may be missing the complete volumetric FE connectivity and physical solver metadata needed for FE equilibrium. **Do not assume** availability.

**B. Dalton reference code:**
- https://github.com/dodaltuin/passive-lv-gnn-emul
- Read README, `PAPER_REPLICATION.md`, `data/README.md`, and `data/DATA_FORMAT_REQUIREMENTS.md` before modifying or processing anything.
- Original implementation used Python 3.9.7, JAX 0.3.16, Flax 0.3.6 (legacy). It contains pretrained model parameters and a fixed-geometry dataset. Keep it in an isolated environment/container; do not destructively update legacy dependencies. Prefer original predictions for reproducibility first.
- Reference replication invocation, subject to reading the repository's current replication instructions:
  `python -m main --mode="evaluate" --data_path="lvData" --K=5 --n_shape_coeff=32 --dir_label="_paperReplication" --trained_params_dir="emulationResults/trainedParameters/lvData/"`
- Repository licence is MIT; capture the actual licence version and dataset use terms separately.

**C. Mechanics reference / solver:**
- https://github.com/finsberg/cardiac_benchmark
- Docs: https://finsberg.github.io/cardiac_benchmark/README.html
- Related simulation archive: https://zenodo.org/records/14260459 (large, around 23 GB; verify contents before downloading).
- Use this resource for verified cardiac mechanics benchmark configurations and/or generating FEM-consistent data with mesh connectivity, quadrature, loads, material parameters, and boundary conditions. Confirm exact mathematical correspondence of formulation and test case. Do not merge two datasets as if they share physics, mesh, units, boundary conditions, or train/test task.

**D. Conference:**
- https://www.cmbbe-symposium.com/2027/abstract-submission/
- Check latest deadline, template, abstract length, figure policy, topic category and submission mode directly. Use actual results only; never invent an accepted submission.

### Data inspection checklist (mandatory report)
For each data source, output `reports/data_audit.md` and a machine-readable `reports/data_audit.json` with: exact downloaded path, checksum, licence, number of cases, sizes/dtypes/shapes, coordinate/displacement units, names and distributions of node features, mapping of each global feature to a physical meaning **only if documentation/code supports it**, geometry and node/edge count, element connectivity availability, boundary-node labels, pressure/traction loading, material coefficients, constitutive law, reference/deformed configuration conventions, split integrity, duplicates, nonfinite values, extreme values, nodal correspondence, solver reproducibility, and expected use. Explicitly mark unknowns; do not assign invented labels.

`real-node-topology.npy` is graph **edge** topology, not evidence of 3D finite elements. Original documentation lists `real-node-coords.npy`, `real-node-displacement.npy`, `real-node-features.npy`, `global-features.npy`, optional `shape-coeffs.npy`, `real-node-topology.npy`, `representative-nodes.npy`. Inspect actual archive before hardcoding.

## 3. Two-track experiment design and decision gate

**Track A (must start first, fastest demonstrable task):** Reproduce Dalton passive-diastolic LV emulator predictions and construct a modern, data-driven GNN plus a five-member deep ensemble. Evaluate geometry-generalization, uncertainty and speed of inference. Label physics only where valid. Complete this even if Track B blocks.

**Track B (required to justify “Physics-Informed” in final title):** Use truly mechanics-complete FE examples from validated cardiac benchmark software/data (or recover original FE connectivity and boundary/material data if genuinely available). Implement differentiable finite-strain FE weak-form residuals, numerical verification, and a physics-guided surrogate matched to a data-only baseline. Demonstrate reduction in physical residual, not merely visually smooth fields. Evaluate true solver speedup on matched cases.

**Decision gate:** After A dataset audit, choose one:
1. **Complete mechanics on A:** Implement full physical constraints directly on same LV cohort; preferred if verified.
2. **No complete mechanics on A:** Keep A as external emulation/uncertainty benchmark. Develop B with benchmark/solver-generated cases. Report A and B separately, avoid conflating their distributions; maintain title only if B demonstrates bona fide physics-informed cardiac surrogate.
3. **No access to valid cardiac solver/data:** Build a verified finite-strain synthetic mechanical *unit* benchmark for debugging, but do NOT claim cardiac physics validation or use final title unqualified. Write blocker/contingency and prioritize reproducible A results.

## 4. Exact task definition, preprocessing and splits

Input: a reference 3D LV nodal geometry `X∈R^(N×3)`, supplied local features, supplied global features and (where supported) material coefficients and loads. Graph edges `(i,j)` carry relative position `X_j−X_i`, length and any documented edge features.

Output: predicted nodal displacement `u_hat∈R^(N×3)` and deformed points `x_hat=X+u_hat`. With a **valid volumetric mesh only**, calculate deformation gradient `F=I+Grad_X u`, Green-Lagrange strain `E=(F.T F−I)/2`, and appropriate stresses.

- Confirm whether preprocessing subtracts a reference geometry, uses normalized coordinates, shape coefficients, or augmented graph nodes; reproduce the author conventions exactly for Track A baseline.
- Normalize continuous predictors using *training split only*, save transformations, verify inverse transforms and physical units.
- Use original train/valid/test for standard benchmark. Define pre-registered geometry-shift split using shape descriptors available from inputs, without selecting threshold by testing performance. If constructing a new split, fit all preprocessing on its training subset and freeze before evaluation.
- No cross-case information leakage from graph augmentation, normalization, pretraining or calibration.
- Calibration must be separate from model-training/early-stopping data and the held-out test set. If original split is limited, partition original training/validation responsibly, record seed, and preserve untouched test.
- For benchmark-generated FE cases, use realistic parameter ranges justified by benchmark, not arbitrary physiological values. Record solver convergence and discard/fail invalid solves with reason and frequency; no silent cherry-picking.

## 5. Models and experimental conditions

Build a modular message-passing graph surrogate (PyTorch + PyTorch Geometric preferred for modern implementation; legacy JAX remains isolated). Use matched encoder/processor/decoder, same train/validation data, parameter counts approximately comparable, training budget, and optimizer settings when testing effect of losses.

- **B0:** original Dalton released pretrained GNN reproduction (Track A reference).
- **B1:** simple data-only baseline (e.g. MLP/graph-independent features or small GNN; document limitations).
- **B2:** modern matched message-passing GNN, supervised `L_data`.
- **P1:** B2 plus mechanically justified Dirichlet/Neumann boundary enforcement (Track B; not merely zeroing unknown nodes).
- **P2:** B2 plus verified FE equilibrium residual (Track B).
- **P3:** B2 plus both boundary and equilibrium (Track B).
- **U1:** five-seed B2 ensemble (Track A or B).
- **U2:** five-seed P3 ensemble (Track B).

Prioritize small-scale solver-verified P2/P3 over a large but mathematically unjustified physical loss. If time/compute is limited, use B0, B2, P3, U1, U2; run additional ablations later. Record seeds [11, 22, 33, 44, 55] (pre-specified example; modify only with documented reason), exact split IDs, training costs and checkpoints.

### Losses

Supervised displacement loss with consistent physical scaling:
`L_data = mean_{nodes} ||u_hat - u_ref||_2²` (report RMSE in physical units separately).

For a verified reference mesh/constitutive model under finite deformation:
- `F = I + Grad_X u`
- `P = dW(F; material_parameters)/dF`
- weak-form FE force balance, free DOFs `R_a(u) = ∫_{Omega0} P : Grad_X N_a dV − f_ext,a(u)`; account correctly for reference/current traction and any follower pressure, body forces, Neumann terms and reaction forces.
- `L_eq = mean_{free DOFs} || R_a(u_hat)/s_a ||²` with a physically sensible, documented nonzero force-scale normalization. Prevent huge mesh-density / scale effects.
- `L_bc` according to correct constrained/displacement boundary DOFs. If hard constraints used, show exact satisfaction. For pressure boundaries, use correct traction formulation, not an invented prescribed displacement.
- `L_total = L_data + λ_eq L_eq + λ_bc L_bc`, choose λ on **validation only**; report grid and selection rule.
- Check determinant `det(F)>0` for elements/quadrature points; report inverted elements and numerical failures. If using any soft penalty, declare it separately and do not misrepresent as equilibrium.
- Implement gradient flow through residual, or establish why mixed residual implementation is differentiable. Verify gradients with finite differences on a small mesh.
- Avoid inventing LV constitutive model or pressure if primary data does not document it. Reproduce the benchmark's documented constitutive law, fiber treatment and boundary conditions.

## 6. Mechanical verification tests (release blockers)

Before physics-GNN experiments pass:
1. Rigid-body translation produces zero internal strain/stress for a suitable unconstrained test.
2. Identity deformation produces physically expected stress response; check reference state/pre-stress assumptions.
3. A manufactured simple affine deformation yields the expected deformation gradient and strain to tolerance.
4. FE quadrature and basis gradients give correct element volumes and stable residuals on a documented test mesh.
5. On a converged reference FEM solution, free-DOF residual is near solver tolerance after appropriate normalization, and constrained reactions are handled correctly.
6. Verify residual direction and magnitudes against trusted solver assembly on at least one benchmark case.
7. Verify autodiff or analytic residual gradient versus central finite differences.
8. Assert no NaNs/Infs, no invalid Jacobians in normal benchmark examples, and correct treatment of non-positive determinant.
9. Verify that artificially perturbing a converged displacement typically increases mechanical residual under controlled conditions.
10. Separately test displacement/traction boundary enforcement and pressure load sign.

If any of these fail, **do not** label results physics-informed until corrected. Report failed tests and limitations transparently.

## 7. Uncertainty and calibration

Train five independently seeded networks for B2 and (when available) P3. Calculate mean prediction `ubar = (1/M)Σu_m`, coordinate-wise sample variance `(1/(M−1))Σ(u_m−ubar)²`, and a per-case uncertainty score such as mean across nodes of ensemble variance. Never claim this variance is a calibrated confidence or credible interval by itself.

Construct an explicitly separate calibration set. Prefer **geometry-level** prediction-interval construction: e.g. scalar score `s_g=max_nodes ||u_hat−u_ref||` for simultaneous per-geometry L2 displacement bounds, or a less conservative score with precisely stated coverage target. Use split conformal with corrected quantile `ceil((n_cal+1)(1−α))/n_cal` via valid implementation (include finite sample/order-statistic edge-case handling). Predict intervals/balls for test geometries. Test empirical coverage at α=.10 and .05 and interval size. Explain marginal exchangeable geometry coverage and its lack of guarantee under distribution shift; do not imply independent per-node marginal coverage equals a joint field guarantee.

Also report uncertainty-error association, selective risk when withholding high-uncertainty predictions, reliability diagrams if appropriate, and results on geometry shift clearly labelled exploratory/out-of-distribution. If adopting standardized residuals using ensemble variance, guard variance floor and validate. Do not tune calibration on final test.

## 8. Evaluation and statistics

**Displacement:** per-geometry mean nodal Euclidean displacement error, vector-component RMSE, 95th percentile error, maximum error, normalized error only with stated denominator and units. Aggregate over geometries (avoid treating nodes as independent cases).  
**Geometry:** deformed-coordinate error.  
**Mechanics (valid FE meshes only):** element/integration-point Green-Lagrange strain error, strain energy error if applicable, normalized free-DOF equilibrium residual, BC violation, min(det F), invalid element fraction.  
**Runtime:** compare warm model inference vs matching reference solver on same task, appropriate hardware and precision; measure batch-one latency (median/p95 across repetitions), preprocessing, model load, solver meshing/setup, solver step, GPU synchronization, and training/simulation corpus cost. Report CPU/CUDA/MPS hardware. Optionally compute break-even number of inference queries including training/data generation.  
**Uncertainty:** nominal vs empirical 90/95% coverage, interval radii, error–uncertainty rank correlation, selective error curves; both in-distribution and OOD with caveat.  
**Shift:** holdout shape-parameter extremes; only evaluate load/material shifts if corresponding features and reference solutions genuinely differ.

Analysis: 5 seeds if compute permits, state number of independent test geometries, per-case paired comparisons, geometry-cluster bootstrap 95% CIs (seed handling declared), preset primary endpoint (normalized equilibrium residual at comparable displacement RMSE), one-sided claims avoided without justified hypotheses; include failure/negative results. Report error distributions and calibration coverage CIs; avoid false p-value precision and post-hoc cherry-picking.

### Planned results tables
- Table 1: dataset source/size/splits/inputs/mesh physics availability.
- Table 2: data-only vs physics-guided displacement, strain, normalized residual, BC violations, invalid elements.
- Table 3: 90/95% calibration, interval widths, error–uncertainty ranking, in distribution and shifted.
- Table 4: reference FEM vs surrogate inference timing, training cost, amortized speedup.
- Figure 1: input geometry, reference displacement, surrogate displacement, error heat map on **actual model mesh**.
- Figure 2: displacement RMSE versus normalized equilibrium residual (matched models, uncertainty across cases/seeds).
- Figure 3: prediction-interval calibration and uncertainty-error relationship.
- Figure 4: inference-speed distributions and geometry-shift comparison.

Figures must be generated from real experiment outputs, include units and color bars, consistent visualization scaling, source/split labels and methods. No photorealistic substitute for scientific evidence. Use PyVista/ParaView for authentic mesh visualization; report when surface visualization is not physically equivalent to volume strain.

## 9. Reproducible repository deliverable

Create this layout while retaining existing files:

```
cardiac-lv-surrogate/
  README.md
  LICENSES_AND_ATTRIBUTION.md
  pyproject.toml
  environment.lock / requirements-lock.txt
  .gitignore
  configs/{data,baseline,gnn,physics,ensemble,eval}/*.yaml
  data/{downloads,raw,processed,splits}/  # ignore large data in git
  external/{passive-lv-gnn-emul,cardiac_benchmark}/
  src/lv_surrogate/
    data/{fetch.py,audit.py,dalton.py,benchmark.py,preprocess.py,splits.py}
    models/{message_passing.py,baseline.py,ensemble.py}
    physics/{kinematics.py,constitutive.py,assembly.py,bc.py,verification.py}
    train/{trainer.py,losses.py,experiment.py}
    eval/{accuracy.py,mechanics.py,calibration.py,shift.py,performance.py}
    vis/{mesh_plots.py,comparison.py,metrics.py}
    utils/{config.py,seeds.py,provenance.py}
  scripts/{audit_data.py,reproduce_dalton.py,run_smoke.py,train_models.py,evaluate_models.py,make_figures.py,benchmark_runtime.py}
  tests/{test_data.py,test_splits.py,test_kinematics.py,test_constitutive.py,test_equilibrium.py,test_gradients.py,test_calibration.py,test_inference.py}
  experiments/{manifests,checkpoints,metrics}/
  reports/{data_audit.md,data_audit.json,baseline_replication.md,mechanics_validation.md,results.md,runtime.md,limitations.md}
  docs/{STATUS.md,SCIENTIFIC_DECISIONS.md,RELATED_WORK.md,EXPERIMENT_PROTOCOL.md,REPRODUCE.md}
  paper/{abstract_draft.md,methods.md,figure_captions.md}
```

Provide simple documented CLIs or make targets for `setup`, `audit`, `smoke`, `reproduce`, `mechanics-verify`, `train`, `evaluate`, `figures`, `benchmark`, `report`; no hidden manual steps. Tests should pass in a small-data CPU smoke mode, without downloading the entire 1.8GB archive. Heavy jobs should be separately gated. Never commit raw copyrighted/large data or secrets. Use source licenses properly.

## 10. Execution stages and go/no-go gates

### Stage 0 — Audit resources and scaffold
- Detect host and hardware, available disk/network; document.
- Read originals; clone sources and pin commit SHAs; set up new code base and tests.
- Inspect small included fixed-geometry set; download full dataset only if resources permit.
- Produce `reports/data_audit.*`, explicit feasibility decision Track A/B, and runnable CPU smoke tests.
**Done when:** data format confirmed from actual files, checksums and split details recorded, a single graph sample loads with the correct target shape.

### Stage 1 — Faithful baseline
- Use author preprocessing / pretrained parameters in isolated legacy environment (potentially container, depending on architecture) to reproduce results; record differences if numerical environment incompatible.
- Create reference prediction vs ground truth mesh visualization, per-case accuracy metrics and replication report.
- Create modern data-only GNN pipeline (test tiny subset then scale).
**Done when:** baseline inference executes on held-out cases with recorded metrics and no leakage; reproducibility limitations documented.

### Stage 2 — Mechanically complete benchmark
- Inspect cardiac benchmark solver, mesh, constitutive and loading metadata; select a manageable documented case set.
- Verify weak-form FE residual with tests outlined in §6; build small valid solver-generated training/validation/test cohort if necessary.
- Train data-only and physics-guided variants using matched model architectures/optimization budgets.
**Done when:** valid reference solver residual and gradient checks pass, physics loss demonstrably backpropagates and its effect is empirically compared.

### Stage 3 — Main models, uncertainty and robustness
- Train selected 5-seed ensembles as compute permits. Quantify accuracy, mechanics, geometry shift, coverage and inference speed.
- Perform ablations `data`, `BC`, `equilibrium`, `BC+equilibrium` and calibrated/un-calibrated ensembles as feasible.
**Done when:** real results tables, confidence intervals, valid plots and reproducible run manifests exist; missing experiments flagged, not silently omitted.

### Stage 4 — Submission packaging
- Review official CMBBE page/template (https://www.cmbbe-symposium.com/2027/); write scientific abstract with background/method/results/conclusion strictly based on attained evidence.
- Include exact n values, primary quantitative comparisons with units and limitations; avoid retrospective inflation.
- Generate report figures and reproducibility instructions; highlight whether the original title is supported.
**Done when:** an abstract draft (not false claim of submission), results verification checklist, methods summary, citations and supervisor-review handoff are ready.


## 11. Reporting instructions on every progress summary

Return this structure:
1. **Completed and validated:** code paths, tests passed, actual sample/case counts, numerical scores if available, run artifacts.
2. **Next actionable step:** exact command or change.
3. **Scientific findings:** supported or unsupported claims, uncertainty/limitations.
4. **Blockers and contingency:** raw error/log summary and what was attempted.
5. **Updated files:** paths in repository.
6. **How to reproduce:** concise copy-pastable commands.

Do not stop at producing a plan. Execute Stage 0 and Stage 1 immediately, then advance to later stages if environment, resources, and time permit. Do not ask for repeated permission for ordinary read-only inspection, testing, source cloning or modest local experiments; follow host permissions and ask only when truly necessary for destructive actions or resource-heavy operations. Never promise unexecuted future results.

## 12. Acceptance rubric

- Reference LV GNN evaluated against real Dalton held-out displacements with verifiable data provenance.
- At least one truly mechanically valid physics-guided cardiac surrogate experiment with reference FE residual verification, *if retaining the physics-informed title*.
- Matched data-only versus physics-guided results with displacement error and mechanical residual.
- Measured inference benchmark and honest statement of whether speedup was observed.
- Evaluated uncertainty/calibration if retaining “uncertainty-aware” in title; at least one independent calibration and held-out test evaluation.
- Exact methods, datasets, limits and reproducible configurations.

## 13. Final command to the agent

**Start now.** Inspect my working directory and hardware, read and pin the source repositories, audit the available datasets, set up the reproducible project and tests, then attempt one actual reference LV deformation prediction using the original model. Continue stage by stage with demonstrable code and numerical results. Be scientifically skeptical: investigate and fix residual/geometry issues rather than cosmetically showing plausible colored meshes. Record all important evidence in the repository and tell me exactly what ran and what remains unverified.
