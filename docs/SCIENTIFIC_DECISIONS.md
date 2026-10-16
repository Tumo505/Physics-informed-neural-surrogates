# Scientific decisions

2026-10-16:
- Preserve existing sources; link them rather than alter user checkouts. Pin commit SHAs and hash data/checkpoints.
- Primary LV output is passive beginning-to-end diastolic nodal displacement. main.py explicitly states evaluation units are cm. Report cm, with no patient/clinical claim.
- Fixed geometry cases share a single geometry, including across author splits. They can validate released decoder inference and material-parameter interpolation only, not unseen-geometry generalization or geometry conformal coverage.
- Variable LV archive only includes edge topology; no FE elements, quadrature, full constitutive/BC/load specification. No edge-smoothness loss will be called physics.
- Cardiac benchmark is elastodynamics with pressure, epicardial/base springs/friction, inertia, viscosity and optional activation. Need displacement history, exact FE space/DOF mapping, fibers, velocities/accelerations, time integration and boundary metadata to validate residual.
- Reference preprocessing uses provided fixed-geometry topology and normalization statistics for varying geometries. Do not regenerate random clustering for pretrained reproduction. Treat original normalization as released B0 artifact; new models fit transformations on their training cases only.
- Original code's mean_rmse is mean nodal Euclidean error, not component RMSE. Report both separately.
- Modern pretrained-weight translation is permissible as an explicitly qualified numerical reproduction if legacy runtime cannot run; validate against original Flax model forward pass where possible.
- Pilot subset/budget chosen before test evaluation; pilot test is diagnostic, never used to choose architecture or loss. Full untouched test evaluation must be separated from pilot development claims.

- Full audit confirms the fixed-geometry dataset exactly equals varying test geometry index 468 (max coordinate difference 0 cm). Never pool fixed-data training with varying-geometry generalization claims; its pretrained transfer experiment includes a geometry from varying test by design. Modern pilot uses no fixed data or pretrained parameters.
- Modern pilot uses first 64 varying test geometries; B0 is now evaluated on all 567 available test cases. Comparisons must state different sample sets and training budgets. Published-paper table reproduction is not yet established, though all released inference is executed.
- Native conda-forge osx-arm64 fenics-dolfin 2019.1.0 packages now exist (verified 2026-10-16), so the original README's macOS incompatibility warning is outdated for this host. Installed isolated 333 MB environment. Direct invocation needed environment PATH; MPI endpoint setup blocked under sandbox, escalated single-process probe. Keep raw errors in status.
- Cardiac first-step probe uses authentic ellipsoid mesh, benchmark material/springs/friction/rho/generalized-alpha parameters, zero activation, pressure evaluated from benchmark waveform at t=0.0006 s (alpha_f=0.4, dt=0.001). P1 displacement and P1 fibers selected for manageable verification; this is a documented discretization variant, not reproduction of default P2 published trajectory.
- Cardiac residual must be assembled before updating old state; call solver.solve directly rather than Problem.solve (which updates histories). Normalize by initial force norm for this load step. No pressure magnitude invented.
- The original common raw graph topology is a supplied parametric discretization, not learned from target displacements; modern model uses those invariant graph indices and per-case relative coordinates. New node/edge/global/target scalers fit only the selected training cases. Constant physical vector mean baseline uses the same training target mean.
- Mean case component RMSE and pooled component RMSE differ by aggregation and are explicitly separated; original mean_rmse is nodal L2, not either component metric.
