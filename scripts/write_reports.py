from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parents[1]
def load(n):return json.loads((P/f'experiments/metrics/{n}.json').read_text())
a=load('reference_fixed');b=load('reference_varying');u=load('ensemble_pilot');r=load('runtime');training=load('training')
rows=['# Executed results — 2026-10-16','','All values computed from real Dalton FEM displacement data; no clinical claim. B0 is a released-weight numerical reproduction on modern Flax, not an identical legacy runtime. B2/U1 is a prespecified small data-only pilot, not a full-scale replacement or physics-informed result.','','| Model/task | Train cases | Test cases/geometries | Mean nodal L2 (cm) | Mean case component RMSE (cm) |','|---|---:|---:|---:|---:|',f'| Released fixed decoder | author pretrained | {a["n_cases"]} / 1 | {a["metrics_mean_over_cases"]["mean_nodal_l2_cm"]:.6f} | {a["metrics_mean_over_cases"]["component_rmse_cm"]:.6f} |',f'| Released varying GNN | author pretrained | {b["n_cases"]} / {b["n_cases"]} | {b["metrics_mean_over_cases"]["mean_nodal_l2_cm"]:.6f} | {b["metrics_mean_over_cases"]["component_rmse_cm"]:.6f} |',f'| Five-seed modern pilot | 64 | 64 / 64 | {u["mean_nodal_l2_cm"]:.6f} | {u["mean_case_component_rmse_cm"]:.6f} |',f'| Training-fitted constant vector mean baseline | 64 | 64 / 64 | {u["training_mean_baseline_mean_nodal_l2_cm"]:.6f} | — |','','The released GNN is evaluated on a larger test cohort and uses a much larger training budget. These rows are not a matched architecture/budget comparison. Pilot is far less accurate; no superiority claim.',f'Pilot geometry bootstrap 95% CI for mean nodal L2: {u["geometry_bootstrap_95CI_mean_nodal_l2_cm"]}; resample test geometries with fixed trained ensemble, 2000 bootstrap replicates. This excludes training-seed variability. Five individual seed errors are in ensemble_pilot.json.', '', '| Nominal simultaneous coverage | Test coverage | Radius (cm) | Exact binomial 95% interval |','|---|---|---|---|']
for c in u['coverage']:rows.append(f'| {c["nominal"]:.0%} | {c["successes"]}/{c["n"]} ({c["coverage"]:.2%}) | {c["radius_cm"]:.4f} | {c["exact_binomial_95CI"]} |')
rows +=['',f'Separate calibration: {u["n_calibration"]} geometries. Whole-geometry max-node L2 score and corrected split-conformal order statistic. Wide balls reflect poor pilot accuracy. Raw ensemble variance is not itself calibrated. Spearman variance/error association {u["spearman_uncertainty_mean_error"]:.4f}; selective-risk values saved. Exchangeability assumed; shift calibration/generalization not established.', '',f'Total five-seed training wall time (sum of separate seed timings): {sum(t["train_seconds"] for t in training):.2f} s. Each seed: 19,619 parameters; 640 optimizer updates. Costs exclude preprocessing/install/data generation.', '', 'Physics-guided comparison, FE strain/stress errors, matched FEM timing, OOD shift and final submission evidence: not complete. Kinematic verification and synthetic unit checks are reported separately; no equilibrium gain claimed.','','![Real held-out field](figures/reference_prediction.png)','','![Pilot uncertainty](figures/uncertainty_pilot.png)']
(P/'reports/results.md').write_text('\n'.join(rows))
(P/'reports/baseline_replication.md').write_text(f'''# Released-model reproduction

Executed the unedited author models.py with modern JAX {b['jax']}/Flax {b['flax']}. Original CLI cannot import deprecated flax.optim under current runtime; legacy jaxlib wheel resolution failed on cp312 arm64. No original-environment identity claim.

Fixed decoder: {a['n_cases']} actual held-out parameter cases, one geometry; mean nodal error {a['metrics_mean_over_cases']['mean_nodal_l2_cm']:.8f} cm. NumPy numerical forward cross-check versus author Flax max normalized difference {b['numpy_vs_unedited_author_fixed_max_abs_normalized']:.3g}.

Varying model: all {b['n_cases']} available test geometries; mean nodal error {b['metrics_mean_over_cases']['mean_nodal_l2_cm']:.8f} cm. Reconstructed author clustering/statistics preprocessing max differences {b['preprocessing_max_abs']}. Target inverse normalization uses released training summary statistics. No model selection performed here. No undocumented material coefficient assignments.

Actual case counts differ from source comments; missing cases were not fabricated. Paper numerical table equality has not been established because the published table and missing-case filtering need detailed reconciliation. Per-case quantitative results and first held-out field arrays saved in experiments/metrics.
''')
(P/'reports/runtime.md').write_text(f'''# CPU inference benchmark

Apple M5 Pro, float32, batch one, two PyTorch threads, 6784 nodes, 37056 graph edges. Modern pilot seed11 only (five-member uncertainty needs five passes). Ten warmups and 100 timed repetitions. Run with make benchmark in isolation; values vary with system load.

- Inference incl. inverse target normalization: median {r['inference_ms_median']:.3f} ms, p95 {r['inference_ms_p95']:.3f} ms.
- Resident-case preprocessing median {r['preprocess_ms_median']:.3f} ms.
- Resident-case end-to-end median {r['end_to_end_resident_case_ms_median']:.3f} ms, p95 {r['end_to_end_resident_case_ms_p95']:.3f} ms.
- Model load {r['model_load_seconds']:.5f} s; case load {r['case_load_seconds']:.5f} s.

No matched Dalton FEM solve is available. No FEM speedup, amortization or break-even is claimed. Full reference GNN timings in reference_varying.json include JAX input transfer/materialization and inverse scaling, exclude preprocessing and are descriptive under concurrent CPU work, not an isolated speed comparison.
''')
(P/'reports/limitations.md').write_text('''# Limitations and gates

- Original JAX 0.3/Flax 0.3 runtime not reproduced; current-runtime unedited author forward validated numerically.
- Actual varying archive 2967 cases, fixed set 346 cases. Reasons for missing cases unknown; no exclusions made here.
- Graph edges are not volume elements. Dalton governing residual, strain/stress and matched FEM solution unavailable; physical parameter channel meanings remain unknown pending detailed paper mapping.
- Native isolated FEniCS installed, but cardiac probe exits139 (native segmentation fault), including fully activated fresh-cache retry; see solver_probe.md. No new cardiac physics-informed surrogate validated yet. Synthetic neo-Hookean checks and affine cardiac-mesh checks cannot establish cardiac solver consistency.
- Pilot modern model undertrained: 64 train cases and ten epochs. Five seeds and separate calibration are real, but intervals are broad; not evidence of deployable performance.
- No OOD evaluation yet; no conditional or shifted-distribution coverage guarantee. Bootstrap describes geometry sampling only, not full training uncertainty.
- MPS built but unavailable to this process; all reported experiments CPU. Data generation cost unknown. No matched FEM speedup.
- Conference template linked but full format/figure requirements not yet parsed; no final abstract submission made. Proposed physics-informed title not yet supported.
''')
(P/'README.md').write_text('''# Cardiac LV surrogate research

Executed research workspace for CMBBE 2027. Start with [results](reports/results.md), [status](docs/STATUS.md), [decisions](docs/SCIENTIFIC_DECISIONS.md), [protocol](docs/EXPERIMENT_PROTOCOL.md) and [reproduction](docs/REPRODUCE.md).

Stage 0 completed. Stage 1 released-weight forward inference executed on 567 varying and 246 fixed cases, with qualified modern-runtime reproduction. Five-seed small data-only GNN pilot and independent calibration executed. Physics-informed cardiac validation remains incomplete; the proposed final title is not yet justified.

Run make smoke, make reproduce, make train, make evaluate, make benchmark, make figures, make report. Original data/source are preserved; no raw data/checkpoints committed. See LICENSES_AND_ATTRIBUTION.md.
''')
# Small checkpoint/source provenance; raw artifacts remain local and reproducible.
manifest={'sha256':{str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for pattern in ['experiments/checkpoints/*','scripts/*.py','src/**/*.py','configs/gnn/*.yaml','*lock*','docs/EXPERIMENT_PROTOCOL.md'] for p in P.glob(pattern) if p.is_file()},'date':'2026-10-16'}
(P/'experiments/manifests/artifacts.json').write_text(json.dumps(manifest,indent=2))
print('reports written')
