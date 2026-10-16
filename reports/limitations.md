# Limitations and gates

- Original JAX 0.3/Flax 0.3 runtime not reproduced; current-runtime unedited author forward validated numerically.
- Actual varying archive 2967 cases, fixed set 346 cases. Reasons for missing cases unknown; no exclusions made here.
- Graph edges are not volume elements. Dalton governing residual, strain/stress and matched FEM solution unavailable; physical parameter channel meanings remain unknown pending detailed paper mapping.
- Native isolated FEniCS installed, but cardiac probe exits139 (native segmentation fault), including fully activated fresh-cache retry; see solver_probe.md. No new cardiac physics-informed surrogate validated yet. Synthetic neo-Hookean checks and affine cardiac-mesh checks cannot establish cardiac solver consistency.
- Pilot modern model undertrained: 64 train cases and ten epochs. Five seeds and separate calibration are real, but intervals are broad; not evidence of deployable performance.
- No OOD evaluation yet; no conditional or shifted-distribution coverage guarantee. Bootstrap describes geometry sampling only, not full training uncertainty.
- MPS built but unavailable to this process; all reported experiments CPU. Data generation cost unknown. No matched FEM speedup.
- Conference template linked but full format/figure requirements not yet parsed; no final abstract submission made. Proposed physics-informed title not yet supported.
