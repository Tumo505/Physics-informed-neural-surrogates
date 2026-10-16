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
