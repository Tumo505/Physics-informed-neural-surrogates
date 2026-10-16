# Methods record

Data provenance, preprocessing and checksum audit: reports/data_audit.json. Sources and licenses: LICENSES_AND_ATTRIBUTION.md. Primary pretrained model source pinned at Dalton commit 449cc22459ba5ada344fe0ff238e6b586d5e9319. Actual supplied splits: varying 2250/150/567; fixed 85/15/246. Fixed geometry is exactly varying test:468, so do not pool those observations as independent geometries.

Released inference preserves the author graph augmentation, clustering labels, normalization, 32 shape coefficients, five message-passing steps, 40 latent channels and 128–128 hidden MLPs with tanh and LayerNorm epsilon 1e-6. Native author models.py is unedited. Modern Flax runtime differs from original environment; compatibility tests and exact errors recorded.

Pilot: see frozen docs/EXPERIMENT_PROTOCOL.md and configs/gnn/pilot.yaml. Native PyTorch mean aggregation, width32, two message rounds, training-only normalization, five independently seeded networks. Per-case errors aggregate by geometry. Bootstrap resamples test geometries for the fixed ensemble. Split conformal scores are max nodal L2 error in cm, with finite-sample ceil((n+1)(1-alpha)) order statistic and infinity when rank exceeds calibration size. No OOD guarantee.

Cardiac mechanics: separate benchmark mesh in m. Linear tetrahedral basis gradients and F are verified against manufactured affine fields. Neo-Hookean unit constitutive law is dimensionless and never assigned to cardiac tissue. Benchmark-specific trusted solver and residual comparison remain a distinct gate.
