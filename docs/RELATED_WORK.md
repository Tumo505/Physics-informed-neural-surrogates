# Related work and novelty map

Accessed 2026-10-16. Compact starting review; not a systematic or exhaustive novelty claim.

| Reference / verified DOI | Relation | Required distinction in this project |
|---|---|---|
| Dalton, Gao, Husmeier (2022), *Emulation of cardiac mechanics using Graph Neural Networks*, CMA 401, 115645. https://doi.org/10.1016/j.cma.2022.115645 ; author-hosted record https://eprints.gla.ac.uk/279057/ | Existing multiscale passive LV graph emulator and pretrained baseline | Replication is not novelty. Need verified equilibrium and independent geometry calibration beyond this baseline. |
| Pfaff et al., *Learning Mesh-Based Simulation with Graph Networks*, ICLR 2021. https://doi.org/10.48550/arXiv.2010.03409 ; https://openreview.net/forum?id=roNqYL0_XP | Mesh message passing for simulation | Graph connectivity is not enough to define FE weak-form residual. |
| Raissi, Perdikaris, Karniadakis (2019), *Physics-informed neural networks*, JCP. https://doi.org/10.1016/j.jcp.2018.10.045 | PDE-informed neural training | Need a specified nonlinear mechanical BVP and tested residual; a smoothness loss does not qualify. |
| Li et al., *Fourier Neural Operator for Parametric Partial Differential Equations*. https://doi.org/10.48550/arXiv.2010.08895 | Parametric solution operators | Cartesian Fourier representation does not directly supply irregular LV mesh physics. Operator comparison optional after matched GNN baseline. |
| Lakshminarayanan, Pritzel, Blundell, NeurIPS 2017. https://doi.org/10.48550/arXiv.1612.01474 | Deep ensembles as uncertainty indicators | Our pilot uses independently seeded deterministic regressors; raw sample variance is not a confidence interval and does not reproduce their probabilistic training recipe. |
| Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*. https://doi.org/10.48550/arXiv.2107.07511 ; author code https://github.com/aangelopoulos/conformal-prediction | Split conformal prediction | Whole-geometry max-node L2 scores, distinct calibration cases, corrected order statistic; exchangeability needed, no OOD guarantee. |
| Arostica et al., *A software benchmark for cardiac elastodynamics*, CMA 435 (2025), 117485. https://doi.org/10.1016/j.cma.2024.117485 | Multi-solver cardiac verification and complete benchmark formulation | Dynamic/viscous/active options differ from Dalton passive endpoint task. Separate tracks and distributions. |
| Finsberg & Sundnes software release. https://doi.org/10.5281/zenodo.10875818 | Trusted implementation for mechanics validation | Commit pinned locally; original results must not be conflated with our runs. |

Proposed contribution remains unproven: a controlled mechanics-guided cardiac surrogate plus geometry-level uncertainty assessment and matched FEM timing. Current evidence is faithful released-weight inference, an undertrained modern pilot with calibrated wide balls, and mechanics unit verification. It does not yet establish a publishable physics-informed improvement.
