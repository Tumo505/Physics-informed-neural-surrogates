# Research status — 2026-10-16 (Africa/Johannesburg)

Master specification read completely; preserved as MASTER_SPEC.md. Original user files retained.

## Stage 0 in progress
- Host: macOS Darwin 27 arm64, Apple M5 Pro, 18 CPU cores, 20 GPU cores, 48 GB RAM; 167 GiB free before extraction. CUDA inapplicable; MPS runtime check pending.
- Existing source commits: Dalton 449cc22459ba5ada344fe0ff238e6b586d5e9319; cardiac_benchmark c7d2d1ba6d680627dc465598dbf6b0363a75d604. Dalton has only untracked .DS_Store files; no source edits made.
- Read README, PAPER_REPLICATION.md, data/README.md, DATA_FORMAT_REQUIREMENTS.md, model/data loader/main/preprocessing code. Read cardiac README, environment, constitutive and problem formulation.
- Both original archives present: LV 1,765,469,988 bytes, expanded 2,577,977,800; cardiac 23,180,741,494 bytes, expanded 33,675,937,769. Inspected zip member tables without extraction. No new dataset downloads needed.
- Bundled Python 3.12.14 used for isolated .venv; system Python 3.14.8 is unsuitable for legacy JAX. No conda/docker commands installed.
- Initial pip attempt failed DNS under sandbox. Retrying package installation with network escalation into .venv.

## Decision gate
Track A graph edges are not FE connectivity. Track B uses dynamic viscoelastic anisotropic mechanics; static residual is not interchangeable. Solver availability and archive metadata still being audited. No physics-informed result claimed.

## Milestone: executed released inference and mechanics units
- Installed pinned modern environment (requirements-lock.txt), CPU backend. torch 2.14.1, JAX 0.11.2, Flax 0.12.10; MPS built but unavailable to this process; JAX CPU.
- Executed `scripts/reproduce_dalton.py` using bundled NumPy: 246 held-out fixed-geometry parameter cases, mean nodal L2 error 0.01817175 cm, mean per-case component RMSE 0.01139083 cm. One independent geometry only.
- Executed unedited author `models.py` with current Flax and released checkpoints. Adapter difference max 7.15e-7 normalized; author preprocessing agreement max 6.44e-6. Initial varying-geometry 16-case mean nodal error 0.03542131 cm. Expanded full test evaluation running.
- Attempted `pip install --dry-run jax==0.3.16 jaxlib==0.3.15 flax==0.3.6`: no matching jaxlib cp312 arm64 wheel. Original CLI imports removed `flax.optim`; no legacy-environment identity claim. Source remains unmodified.
- Archive MD5 checks both passed. Zenodo API metadata fetched: both datasets CC-BY-4.0. Initial audit hit missing cardiac_benchmark/LICENSE (README declares MIT); fixed missing-file handling and reran successfully.
- Actual varying splits are 2250/150/567, total 2967; fixed 85/15/246, total 346. These differ from source comments 3000/350. No excluded or reconstructed missing cases. All original partitions retained.
- Exact geometry duplicate groups: 0 in varying cohort. Shape descriptor nearest distances recorded; no near-duplicates under the prespecified 1e-6 rule. This is not a proof against all possible anatomical similarity.
- Selected authentic ellipsoid mesh (1.17 MB) from benchmark ZIP, no full 33.7 GB extraction. 4577 nodes, 17625 tetrahedra; affine F max absolute error 1.824e-14. Ten tests passed including volume/basis, rigid motion, affine strain, finite-difference force and residual-loss gradients, invalid Jacobians, conformal edges, GNN permutation/backprop.
- Synthetic neo-Hookean model is nondimensional test-only; no cardiac residual validation claimed. Trusted solver absent, cardiac pressure/Robin/dynamic residual not implemented.
- Five-seed modern 64-training-case pilot running; 16 validation, 134 calibration, first 64 test cases reserved before evaluation. See frozen protocol and config.

## Milestone: Track A pilot and native solver setup
- Released GNN completed 567/567 varying test cases: mean nodal L2 0.04103705 cm; mean case component RMSE 0.02623425 cm.
- Five 19,619-parameter graph models trained (seeds 11/22/33/44/55, 64 cases, 640 updates each). Mean ensemble nodal error 0.479376 cm on 64 test geometries; training-mean constant-vector baseline 0.989261 cm. Models underperform released GNN; retain negative finding.
- Separate 134-geometry calibration: 90% radius 2.322524 cm, coverage 59/64; 95% radius 2.832718 cm, coverage 63/64. Spearman raw variance/error 0.673626. Wide intervals; no practical accuracy claim.
- CPU batch-one modern single-member inference median 12.118 ms / p95 13.511 ms in current timing run; no matched FEM speedup.
- Figures generated from actual saved nodal predictions. Mesh graph shown as points/actual edges, not inferred FE surface/volume strain. Visual inspection performed.
- Fixed geometry matches varying test:468 exactly; cross-track leakage warning documented. Modern pilot never uses fixed data.
- Installed native FEniCS via micromamba (198 packages, resolved 333 MB download); isolated .fenics/.mamba ignored in Git. Dry-run resolution saved. Direct import first failed missing pkg-config in PATH; corrected PATH, then MPI sandbox OFI endpoint permission failure. Escalated solver run progressed to missing meshio. Installing pure-Python loader dependencies with working pip; conda pip startup stalled. No physics result yet.

## Final checkpoint for this execution
- Stage 0: validated, authentic graph loads and checksum/split audits complete.
- Stage 1: released model inference complete on all available held-out cases; current-runtime numerical reproduction qualified, original legacy CLI/paper table equality not established. Modern graph pipeline executed.
- Stage 2: kinematics/synthetic gradient tests pass. Native FEniCS installation succeeded, but corrected cardiac probe repeatedly exits 139 (segmentation fault), including full activation + fresh compiler cache. Last logs at Laplace/interpolation JIT; cause unresolved. See reports/solver_probe.md. No converged cardiac FE residual or physics-informed surrogate result.
- Stage 3: five-seed data-only pilot, independent conformal calibration, accuracy/bootstrap/coverage/uncertainty and latency results executed. Full-budget modern training, geometry-shift and mechanics-guided ensembles incomplete.
- Stage 4: compact novelty map, preliminary evidence abstract, methods/captions and conference source requirements recorded; no submission. Official template length/figure rules pending inspection; intended physics-informed title unsupported at this checkpoint.
- Final smoke: 12 tests passed; included held-out graph CPU inference passed; source/scripts compileall succeeded. Pinning local Git snapshot of code, small evidence and reports (large data/checkpoints and environments ignored).
- Immediate next action: diagnose native solver with minimal Laplace/interpolation and native traceback, then verify independent cardiac residual; keep modern main experiment protocol frozen until stronger validation-only training study is designed. No final test-based hyperparameter tuning.

Commands actually run include scripts/audit_data.py, inspect_benchmark.py, reproduce_dalton.py, reproduce_author_forward.py --limit 0, train_models.py, evaluate_models.py, benchmark_runtime.py, verify_mechanics.py, make_figures.py, write_reports.py, `make smoke`, compileall and the logged isolated solver-probe invocations. Detailed configs, source hashes, checkpoints and package locks retained. Original files preserved.

Reproducibility snapshot created locally; original copied guide retains its intentional Markdown trailing spaces (diff whitespace warnings only there). Small metric JSON and scientific PNGs versioned. Example geometry/target arrays and checkpoints remain on disk but are excluded from Git, alongside all raw data and environments. No remote push or submission performed.
