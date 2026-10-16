# Released-model reproduction

Executed the unedited author models.py with modern JAX 0.11.2/Flax 0.12.10. Original CLI cannot import deprecated flax.optim under current runtime; legacy jaxlib wheel resolution failed on cp312 arm64. No original-environment identity claim.

Fixed decoder: 246 actual held-out parameter cases, one geometry; mean nodal error 0.01817175 cm. NumPy numerical forward cross-check versus author Flax max normalized difference 7.15e-07.

Varying model: all 567 available test geometries; mean nodal error 0.04103705 cm. Reconstructed author clustering/statistics preprocessing max differences {'augmented-node-features': 2.384185791015625e-07, 'edge-features': 6.4373016357421875e-06, 'global-features': 1.1920928955078125e-07, 'shape-coeffs': 1.1920928955078125e-07}. Target inverse normalization uses released training summary statistics. No model selection performed here. No undocumented material coefficient assignments.

Actual case counts differ from source comments; missing cases were not fabricated. Paper numerical table equality has not been established because the published table and missing-case filtering need detailed reconciliation. Per-case quantitative results and first held-out field arrays saved in experiments/metrics.
