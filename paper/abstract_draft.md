# Preliminary evidence draft — NOT submission-ready

Working evidence title: **Reproducing passive left-ventricular graph emulation and evaluating geometry-level uncertainty: an initial computational study**

The proposed study title “Physics-Informed Neural Surrogates for Rapid and Uncertainty-Aware Prediction of Left Ventricular Deformation” remains conditional on successful cardiac equilibrium verification and matched physics-guided experiments.

**Background.** Passive left-ventricular deformation surrogates require independent assessment of displacement accuracy, uncertainty and mechanical consistency. Existing graph emulators provide a reference for reproducible comparison.

**Methods.** We audited public Dalton data and executed its released graph model on 567 available held-out geometries using the unmodified model equations under modern JAX/Flax. Separately, five small supervised message-passing models were trained on a prespecified 64-geometry subset, selecting checkpoints on 16 validation geometries. A disjoint set of 134 geometries calibrated simultaneous nodal L2 displacement balls; 64 test geometries assessed coverage. Synthetic mechanics unit checks and manufactured affine kinematics on an independent cardiac mesh were distinguished from solver validation.

**Results.** The released model achieved mean nodal error 0.0410 cm on 567 test geometries. The five-model pilot ensemble achieved 0.479 cm mean nodal error (geometry bootstrap 95% CI 0.387–0.592 cm), compared with 0.989 cm for a training-mean baseline. Nominal 90% and 95% simultaneous balls covered 59/64 and 63/64 geometries with radii 2.32 and 2.83 cm. Single-model CPU inference had median latency 12.1 ms on an Apple M5 Pro. No matched FEM speedup was established. Affine deformation-gradient error on 17,625 actual cardiac tetrahedra was below 1.9×10⁻¹⁴, which verifies kinematics only.

**Conclusion.** Released-model inference and independent uncertainty assessment are reproducible, but the small modern pilot is substantially less accurate and its calibrated bounds are broad. Cardiac equilibrium guidance, matched FEM timing and geometry-shift robustness remain necessary before the intended physics-informed contribution can be claimed.

Author identities, affiliations and supervisor review pending. Format/figure allowance must be checked against the official mandatory template before submission. No submission performed.
