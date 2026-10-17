# Physics-informed neural surrogates for LV deformation

Computational research workspace. Quantitative outcomes are in [reports/results.md](reports/results.md), with per-case metrics, configs and provenance in `experiments/`. Original data and reference code remain separate. See [LICENSES_AND_ATTRIBUTION.md](LICENSES_AND_ATTRIBUTION.md).

## Reproduce

Use Python3.12. Original archives and reference repositories are expected beside this repository; source commits and data checksums are recorded in the audit/manifests.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
make smoke
make audit
make reproduce
make prepare-full
make train-full
make evaluate-full
make figures-full
make report
```

Full-data training uses all2250 author training geometries, five independent seeds, a106-node graph hierarchy and a point decoder predicting all6784 original nodes. Training samples nodes with a cyclic schedule; full-node validation selects checkpoints. Hyperparameters, splits and budgets are in configs/gnn/full.yaml; training is resumable. GPU execution requires available Apple MPS; CPU can be selected explicitly with `scripts/train_full.py --device cpu`. Do not run multiple training processes against the same checkpoints.

`make query-density` reproduces the matched validation-only128/1024-node continuation study and its figure. Both runs retain their initialization as an eligible checkpoint. This single-seed experiment did not improve validation accuracy; the existing ensemble remains selected. The503-case test cohort has already been evaluated, so future comparisons on it are repeated benchmark evaluations.

Legacy cardiac mechanics uses a separate `.fenics` environment. Recreate it with project-local micromamba: `.tools/bin/micromamba create -p "$PWD/.fenics" --file fenics-environment.lock`, then `.fenics/bin/python -m pip install -r fenics-pip-lock.txt`. The conda lock pins native packages; the pip lock contains only extra pure-Python packages. `make solver-probe` runs the archived LV mesh with benchmark constitutive/loading/time integration forms. Solver execution may require host permission for MPI. The repaired solver supports `make cardiac-cohort cardiac-verify physics-train cardiac-benchmark`. The cardiac experiment is a verified, matched startup-load pilot on one geometry, not a full trajectory or geometry-generalization study.

`make mesh-sensitivity` runs three matched startup pressures on the archived mesh and one uniform refinement, preserving the polyhedral domain, boundary labels and continuous fiber fields. It records volume-L2 differences and Jacobians and generates a figure. Two levels do not establish mesh independence. On this macOS host, MPI uses TCP on interface `en0` to avoid a VPN-interface finalization failure; override with `make mesh-sensitivity MPI_IFACE=<physical-interface>` on another host.

No matched Dalton FEM speedup or clinical validation is claimed. Raw data, trained checkpoints, environments, working notes and `paper/` are excluded from Git. No manuscript files are generated.

The executed extensions are available through `make graph-study`, `make geometry-study`, `make temporal-study`, and `make report` after the original setup/data preparation. They can take tens of minutes from scratch. Finished neural runs are skipped; FE trajectory restart states and per-geometry cohort files are retained. The geometry and temporal studies require the earlier `make cardiac-cohort` artifacts. `geometry-study` also generates the passive reference trajectory used by the extended verification gates.

The new cardiac cohort contains64 independent numerical radius-scale draws within one affine benchmark family and4 extrapolation corners, with four startup loads per geometry. Geometry-separated training/selection/calibration/test counts are20/8/20/16; the four corners are a separate shift test. This is not an anatomical population. Five matched data/physics graph-POD models are evaluated with trusted float64 mechanics. GPU optimization uses a verified float32 analytic-stress residual; the exact smoothed fiber-switch derivative is retained. All fields use the verified common nodal correspondence.

The full999-step passive FE trajectory converged without inverted elements. History-conditioned neural models are evaluated both with true histories and recursively. Their recursive rollouts failed, so they do not establish autonomous full-cycle prediction. The original temporal pilot's physics weights were too weak; the balanced grid is derived from training-only parameter-gradient norms and its repeated temporal evaluation is labelled exploratory. No new manuscript or Markdown files are created by these targets.

`make geometry-confirmation` requires the completed `geometry-study` checkpoints. It refines five matched data/physics pairs from shared data-trained weights using only the original training/validation geometries, then evaluates a fresh cohort of20 calibration,16 test and16 shifted numerical geometries. All208 additional FE solves converged. Data-only refinement achieved0.210µm mean geometry component RMSE versus0.217µm with physics weight100; the physics residual also increased. Both ensembles had16/16 joint test coverage and0/16 shifted coverage at nominal90/95%. These results do not support a startup physics advantage or robustness outside the training parameter range. All original and follow-up results remain in the existing consolidated report.

The full-cycle follow-up uses 60 FE trajectories across 48 numerical geometries, with 20 independent calibration and20 primary/confirmation test geometries. A graph encoder and bounded recurrent POD decoder are trained on complete predicted histories. With one labelled training geometry, physics refinement reduced test RMSE from 0.847 to 0.782 mm and sampled force residual by 76.8%; with four labelled training geometries, physics reduced residual but worsened RMSE from 0.509 to 0.630 mm. All four ensembles completed all 20 test cycles without inverted endpoint or generalized-alpha stress-state elements. These are idealized benchmark-family results, not anatomical validation.

`make dynamic-study` reuses finished local artifacts and executes the full-cycle cohort, convergence checks, matched five-seed studies, calibration, independent confirmation and FE routing tests. On a clean checkout, use `make dynamic-reproduce`: checkpoints are ignored, so missing weights must be retrained and the same-case evaluation is explicitly labelled repeated benchmark reproduction. `make dynamic-benchmark` reuses the recorded isolated timing; reproduction requests fresh timing queries. Native solver execution requires the isolated environment and host MPI access described above.

`make dynamic-dt001` checks the original 1 ms trajectory with a separate one-geometry/one-cycle recurrent pilot. A training-fit gate rejects the trivial zero predictor that low-motion temporal validation initially selected. Nontrivial models completed 999 steps without inverted elements, but their late low-motion forecast errors remain large relative to the small reference motion. Reference numerical sensitivity is reported separately: the 5 ms cohort target differs from the archived1ms baseline by 1.63% in space-time volume-L2, and 2.5 ms differs by 0.35%. Surrogate calibration is relative to its discrete FE target; continuum accuracy and coverage are not established.

The full-cycle spatial check at 2.5 ms changes displacement by 13.12% between 17,625 and 141,000 tetrahedra, failing the 1% tolerance. Numerical convergence is therefore unresolved. Reported neural improvements and uncertainty regions describe the coarse discrete benchmark; they do not include this spatial uncertainty. A further reference-refinement study is required before claiming mesh-independent accuracy.

`make dynamic-bundle` exports the exact 20 frozen model checkpoints and their preprocessing/source/configuration artifacts to an ignored local archive, with member and archive SHA256 checksums in `experiments/manifests/dynamic_replay_bundle.json`. This preserves the evaluated weights separately from retraining, which may differ because MPS accumulation is not bitwise deterministic. The bundle excludes the FE corpus and software environment and is not uploaded automatically.

`make dynamic-reference-capacity` probes a third linear mesh and a quadratic baseline with explicit memory/time guards. The quadratic 2.5 ms attempt failed at 0.2775 s after 110 accepted steps; trust-region and previous-state retries also failed. Two local 1.25 ms substeps passed, motivating a separate 800-step reference from rest. `make dynamic-quadratic-reference` now uses the explicitly configured 1.25 ms protocol, retaining the same polyhedral domain and P1 fiber fields. It checks sampled endpoint/stress-state Jacobians and exact volume norms against the linear meshes, with independent native integration checks. Because timesteps differ, the comparison changes both temporal and spatial discretization. Frozen models are evaluated against the changed reference on one known baseline case; no new population test or continuum-accuracy claim follows. The failed 2.5 ms configuration remains available through `QUADRATIC_CONFIG=configs/physics/dynamic_quadratic_reference.json`.
