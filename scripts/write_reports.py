"""Generate the single consolidated results report and artifact manifest."""
from pathlib import Path
import json,hashlib,math
P=Path(__file__).resolve().parents[1]
def load(name):return json.loads((P/f'experiments/metrics/{name}.json').read_text())
b=load('reference_varying');pilot=load('ensemble_pilot')
rows=['# Executed research results','','All results are computational. Released-model inference uses unedited author equations under modern Flax, not an identical legacy runtime. Graph connectivity alone does not establish FE mechanics.','',f'Released GNN: {b["n_cases"]} held-out geometries, mean nodal L2 error {b["metrics_mean_over_cases"]["mean_nodal_l2_cm"]:.6f} cm.','',f'Original 64-training-case pilot ensemble: mean nodal error {pilot["mean_nodal_l2_cm"]:.6f} cm on test0:64. Independent members were worse than their mean; the pilot gap reflects data, architecture and training-budget differences, not evidence that averaging hurts.']
full=P/'experiments/metrics/full_evaluation.json'
if full.exists():
 d=json.loads(full.read_text());rows+=['','| Model | Main test geometries | Mean nodal L2 (cm) | Mean case component RMSE (cm) |','|---|---:|---:|---:|']
 for key,m in d['models'].items():rows.append(f'| {key} | {d["n_test"]} | {m["mean_nodal_l2_cm"]:.6f} | {m["mean_case_component_rmse_cm"]:.6f} |')
 rows+=['','Main evaluation uses test64:567, excluding the previously inspected pilot cohort. Full-data models train on all2250 original training cases. Checkpoint selection uses validation0:16; calibration uses16:150. Historical calibration reuse is disclosed; no coverage outcome selects a model. The same previously inspected64 pilot cases improved from0.479376cm to0.061811cm; this is a secondary same-case comparison, separate from the503-case main test.','','| Nominal whole-geometry coverage | Empirical | Radius (cm) | Exact binomial95% interval |','|---|---|---:|---|']
 for c in d['coverage']:rows.append(f'| {c["nominal"]:.0%} | {c["covered"]}/{d["n_test"]} ({c["coverage"]:.2%}) | {c["radius_cm"]:.5f} | {c["ci95"]} |')
 rows+=['',f'Ensemble raw-variance/error Spearman correlation: {d["spearman_uncertainty_error"]:.4f}. Geometry bootstrap intervals condition on the fixed trained ensemble and exclude retraining uncertainty. Paired comparisons, input-defined shape-tail analysis and per-case values are in full_evaluation.json. Exploratory shape-tail cases had90.5% coverage for nominal95% bounds (n=84), versus96.7% in central shapes (n=419). No shifted-distribution coverage guarantee.']
query=P/'experiments/metrics/query_density.json'
if query.exists():
 q=json.loads(query.read_text());rows+=['','Validation-only query-density experiment: one matched seed, same starting weights and 60 continuation epochs on all2250 training geometries; compare128 versus1024 nodes per geometry. Starting checkpoint remains eligible for selection. Calibration and test arrays were not opened.','','| Candidate | Selected validation component RMSE(cm) | Selected continuation epoch |','|---|---:|---:|']
 for name,r in q['results'].items():rows.append(f'| {name} | {r["validation"]["component_rmse_cm"]:.7f} | {r["epoch"]} |')
 rows+=['',f'Improvement beyond relative numerical tolerance1e-6: {q["improves_initialization"]}. More point queries did not establish an accuracy benefit in this continuation; this does not rule out a benefit when training from scratch or with a different architecture. Existing ensemble retained. These16 geometries are the reused selection set, so results are descriptive. Future comparisons on the already evaluated503-case test set must be labelled repeated benchmark evaluation.','','![Validation-only query-density comparison](figures/query_density.png)']
mesh=P/'experiments/metrics/mesh_sensitivity.json'
if mesh.exists():
 m=json.loads(mesh.read_text());rows+=['','Two-level cardiac mesh sensitivity: uniform tetrahedral refinement of the same polyhedral domain, retaining boundary labels and interpolating the same continuous P1 fiber fields; original mechanics and timestep retained. This checks displacement discretization, without curved-geometry refinement or independent anatomy.','','| Pressure(Pa) | Relative volume-L2 displacement difference | Volume-weighted component RMSE(µm) | Minimum refined endpoint detF |','|---|---:|---:|---:|']
 for c in m['cases']:rows.append(f'| {c["pressure_Pa"]:.2f} | {c["relative_displacement_volume_L2_difference"]:.3%} | {1e6*c["displacement_L2_difference_m_power_2_5"]/math.sqrt(3*m["mesh_volume_m3"][1]):.3f} | {c["minimum_refined_endpoint_detF"]:.6f} |')
 rows+=['',f'Mesh sizes: {m["meshes"]["coarse"]["tetrahedra"]} to {m["meshes"]["refined"]["tetrahedra"]} tetrahedra. Boundary areas and reference volume match to numerical tolerance. Two levels do not establish an asymptotic convergence rate or a mesh-independent truth. Surrogate error against the original mesh does not include this reference discretization uncertainty.','','![Two-level FE mesh sensitivity](figures/mesh_sensitivity.png)']
probe=P/'experiments/metrics/cardiac_probe.json'
if probe.exists():
 d=json.loads(probe.read_text());rows+=['',f'Trusted cardiac first-step probe now converged: normalized residual {d["normalized_residual"]:.3g}, minimum detF {d["min_detF"]:.7f}, inverted elements {d["inverted_element_fraction"]:.0%}. This is oneP1 dynamic timestep, not a full cardiac trajectory or matched Dalton FEM timing. The former native crash was isolated to PETSc LU option initialization; the workaround preserves governing forms.']
physics=P/'experiments/metrics/physics_experiment.json'
if physics.exists():
 d=json.loads(physics.read_text());rows+=['','Separate cardiac startup experiment:20 training,6 validation and6 test LOAD cases on ONE geometry; P1/fibers, passive anisotropic viscoelastic dynamics, zero initial histories. This is not a full-cycle benchmark or a shape-generalization result.','','| Matched graph/POD model | Test component RMSE(m) | Normalized force RMS | Strain Frobenius error |','|---|---:|---:|---:|']
 for name,m in d['results'].items():rows.append(f'| {name} | {m["mean_component_rmse_m"]:.6g} | {m["mean_normalized_residual_rms"]:.6g} | {m["mean_strain_frobenius_error"]:.6g} |')
 rows+=['',f'Physics weight selected on validation only: {d["lambda_selection"]["selected"]}; within10% displacement degradation constraint: {d["lambda_selection"]["meets_validation_RMSE_10percent_constraint"]}. One matched seed,500updates/model; no population-level significance claim. Full weak residual includes follower pressure, inertia, viscosity and natural epi/base Robin terms. POD rank4 is fitted on training cases only.']
 if 'H1_observed' in d:rows+=['',d['H1_observed']['conclusion']]
rt=P/'experiments/metrics/cardiac_runtime.json'
if rt.exists():
 d=json.loads(rt.read_text());rows+=['',f'Matched cardiac startup CPU/float64 timings: surrogate median {d["surrogate_ms_median"]:.3f}ms; warm FEM median {d["solver_ms_median"]:.3f}ms; ratio {d["warm_median_speedup"]:.1f}×. Same host and two-thread settings; inference/corpus/training costs separate. This ratio applies only to the one-step fixed-geometry task and its measured errors, not Dalton or full cardiac cycles.']
rt=P/'experiments/metrics/full_runtime.json'
if rt.exists():
 d=json.loads(rt.read_text());rows+=['','Full-node Dalton inference timing (batch1,100repetitions; inverse normalization included; resident preprocessed inputs):']
 for device,m in d['results'].items():rows.append(f'- {device}: one model {m["timing"]["one_member"]["median_ms"]:.3f}ms median; five-member ensemble {m["timing"]["five_member_ensemble"]["median_ms"]:.3f}ms median. No matched Dalton FEM speedup.')
cost=P/'experiments/metrics/training_costs.json'
if cost.exists():
 c=json.loads(cost.read_text());rows+=['',f'Observed training cost: full-data graph ensemble stages summed {c["full_multiscale_v1"]["graph_training_seconds_sum"]+c["full_multiscale_v2"]["graph_training_seconds_sum"]:.1f}s; cardiac data/physics validation grid {c["cardiac"]["training_and_validation_grid_seconds_sum"]:.1f}s plus summed corpus solver steps {c["cardiac"]["corpus_solver_seconds_sum"]:.1f}s. These costs are separate from latency ratios and exclude installation/diagnostics.']
rows+=['','![Released prediction on an actual graph](figures/reference_prediction.png)']
if (P/'reports/figures/full_results.png').exists():rows+=['','![Full-data results](figures/full_results.png)']
from report_research_extension import extension_rows
rows=rows[:3]+extension_rows(P)+rows[3:]
(P/'reports/results.md').write_text('\n'.join(rows)+'\n')
patterns=['experiments/checkpoints/**/*.pt','scripts/*.py','src/**/*.py','configs/**/*.yaml','configs/**/*.json','*lock*']
manifest={'sha256':{str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for pattern in patterns for p in P.glob(pattern) if p.is_file()},'note':'Working notes and paper directory intentionally ignored; no manuscript generated.'}
(P/'experiments/manifests/artifacts.json').write_text(json.dumps(manifest,indent=2));print('consolidated report written')
