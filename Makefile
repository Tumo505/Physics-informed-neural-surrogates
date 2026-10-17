PY := .venv/bin/python
MPI_IFACE ?= en0
.PHONY: setup audit smoke reproduce inspect-benchmark mechanics-verify train evaluate figures benchmark report
setup:
	$(PY) -m pip install -r requirements-lock.txt
audit:
	$(PY) scripts/audit_data.py
inspect-benchmark:
	$(PY) scripts/inspect_benchmark.py
smoke:
	$(PY) -m pytest -q tests
	$(PY) scripts/run_smoke.py
reproduce:
	$(PY) scripts/reproduce_dalton.py
	$(PY) scripts/reproduce_author_forward.py --limit 0
mechanics-verify:
	$(PY) -m pytest -q tests
	$(PY) scripts/verify_mechanics.py
train:
	$(PY) scripts/train_models.py
evaluate:
	$(PY) scripts/evaluate_models.py
figures:
	$(PY) scripts/make_figures.py
benchmark:
	$(PY) scripts/benchmark_runtime.py
report:
	$(PY) scripts/write_reports.py
solver-probe:
	env XDG_CACHE_HOME=$(CURDIR)/.cache DIJITSO_CACHE_DIR=$(CURDIR)/.fenics-cache-activated MAMBA_ROOT_PREFIX=$(CURDIR)/.mamba .tools/bin/micromamba run -p $(CURDIR)/.fenics python scripts/solve_cardiac_probe.py
prepare-full:
	$(PY) scripts/prepare_full_data.py
train-full:
	$(PY) scripts/train_full.py
	$(PY) scripts/train_full.py --config configs/gnn/full_low_lr.yaml
evaluate-full:
	$(PY) scripts/finalize_full_checkpoints.py
	$(PY) scripts/evaluate_full.py
figures-full:
	$(PY) scripts/plot_full.py

cardiac-cohort:
	env XDG_CACHE_HOME=$(CURDIR)/.cache MAMBA_ROOT_PREFIX=$(CURDIR)/.mamba .tools/bin/micromamba run -p $(CURDIR)/.fenics python scripts/generate_cardiac_cohort.py
cardiac-verify:
	$(PY) scripts/verify_cardiac_assembly.py
physics-train:
	$(PY) scripts/train_physics.py
	$(PY) scripts/augment_physics_energy.py
cardiac-benchmark:
	$(PY) scripts/benchmark_cardiac.py
benchmark-full:
	$(PY) scripts/benchmark_full.py

query-density:
	$(PY) scripts/train_full.py --config configs/gnn/query_density_128.yaml --kind gnn --seed 11
	$(PY) scripts/train_full.py --config configs/gnn/query_density_1024.yaml --kind gnn --seed 11
	$(PY) scripts/analyze_query_density.py
mesh-sensitivity:
	env FI_PROVIDER=tcp FI_TCP_IFACE=$(MPI_IFACE) XDG_CACHE_HOME=$(CURDIR)/.cache DIJITSO_CACHE_DIR=$(CURDIR)/.fenics-cache-activated MAMBA_ROOT_PREFIX=$(CURDIR)/.mamba OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .tools/bin/micromamba run -p $(CURDIR)/.fenics python scripts/check_mesh_sensitivity.py
	$(PY) scripts/plot_mesh_sensitivity.py

# Requires the original data caches and isolated .fenics environment above.
FEM_RUN = env FI_PROVIDER=tcp FI_TCP_IFACE=$(MPI_IFACE) XDG_CACHE_HOME=$(CURDIR)/.cache DIJITSO_CACHE_DIR=$(CURDIR)/.fenics-cache-activated MAMBA_ROOT_PREFIX=$(CURDIR)/.mamba OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .tools/bin/micromamba run -p $(CURDIR)/.fenics python
.PHONY: graph-study geometry-cohort cardiac-trajectory extended-mechanics geometry-study temporal-study

graph-study:
	$(PY) scripts/prepare_graph_resolution.py
	$(PY) scripts/train_full.py --config configs/gnn/control_v4.yaml --kind gnn --seed 11
	$(PY) scripts/train_full.py --config configs/gnn/capacity_v4.yaml --kind gnn --seed 11
	$(PY) scripts/train_full.py --config configs/gnn/fine_graph_v4.yaml --kind gnn --seed 11
	$(PY) scripts/select_graph_study.py --expand
	$(PY) scripts/train_full.py --config configs/gnn/selected_v4.yaml --kind gnn
	$(PY) scripts/evaluate_graph_study.py
	$(PY) scripts/plot_graph_study.py
geometry-cohort:
	$(FEM_RUN) scripts/generate_geometry_cohort.py
	$(PY) scripts/audit_geometry_cohort.py
	env OPENBLAS_NUM_THREADS=2 $(PY) scripts/prepare_geometry_data.py
cardiac-trajectory:
	$(FEM_RUN) scripts/generate_cardiac_trajectory.py
	$(FEM_RUN) scripts/generate_cardiac_trajectory.py --dt 0.002 --end 0.0401 --name cardiac_dt002
	$(FEM_RUN) scripts/generate_cardiac_trajectory.py --dt 0.0005 --end 0.0401 --name cardiac_dt0005
	$(PY) scripts/analyze_trajectory.py
extended-mechanics: geometry-cohort cardiac-trajectory
	$(PY) scripts/verify_extended_mechanics.py
	$(PY) scripts/check_physics_backend.py
	$(PY) scripts/check_temporal_backend.py
geometry-study: geometry-cohort cardiac-trajectory extended-mechanics
	$(PY) scripts/train_geometry_models_gpu.py
	$(PY) scripts/evaluate_geometry_models.py
	$(PY) scripts/verify_neural_physics_gradient.py
	$(PY) scripts/plot_geometry_study.py
	$(PY) scripts/benchmark_geometry_study.py
temporal-study: cardiac-trajectory extended-mechanics
	env OPENBLAS_NUM_THREADS=2 $(PY) scripts/prepare_temporal_data.py
	$(PY) scripts/train_temporal_model.py
	$(PY) scripts/evaluate_temporal_model.py
	$(PY) scripts/diagnose_temporal_loss_scale.py
	$(PY) scripts/train_temporal_model.py --config configs/physics/temporal_balanced.json
	$(PY) scripts/evaluate_temporal_model.py --run cardiac_temporal_balanced_v2
	$(PY) scripts/plot_temporal_study.py

.PHONY: geometry-confirmation
# Prerequisites: completed geometry-study, its checkpoints and immutable train cache.
geometry-confirmation:
	$(FEM_RUN) scripts/generate_geometry_cohort.py --config configs/physics/geometry_confirmation.json
	$(PY) scripts/audit_geometry_cohort.py --cohort cardiac_confirmation_v1 --output confirmation_cohort_audit
	$(PY) scripts/diagnose_geometry_loss_scale.py
	$(PY) scripts/train_geometry_models_gpu.py --config configs/physics/geometry_refinement.json
	$(PY) scripts/prepare_confirmation_data.py
	$(PY) scripts/evaluate_geometry_models.py --run cardiac_geometry_refinement_v3 --data-root data/processed/cardiac_confirmation_ml --raw-root data/processed/cardiac_confirmation_v1
	$(PY) scripts/plot_geometry_study.py --run cardiac_geometry_refinement_v3 --raw-root data/processed/cardiac_confirmation_v1 --prefix geometry_refinement
	$(PY) scripts/verify_confirmation_artifacts.py

.PHONY: dynamic-cohort dynamic-convergence dynamic-train dynamic-evaluate dynamic-study
# These are passive numerical experiments. Dataset/training configs define all splits.
dynamic-cohort:
	$(PY) scripts/run_dynamic_cohort.py --group cohort --workers 5
dynamic-convergence:
	$(PY) scripts/run_dynamic_cohort.py --group convergence --workers 1
	$(PY) scripts/analyze_dynamic_convergence.py
	$(FEM_RUN) scripts/analyze_dynamic_mesh.py
dynamic-train: dynamic-cohort
	$(PY) scripts/verify_dynamic_backend.py
	env OPENBLAS_NUM_THREADS=2 $(PY) scripts/prepare_dynamic_study.py --regime full
	$(PY) scripts/train_dynamic_study.py --regime full
	env OPENBLAS_NUM_THREADS=2 $(PY) scripts/prepare_dynamic_study.py --regime limited
	$(PY) scripts/train_dynamic_study.py --regime limited
dynamic-evaluate: dynamic-train dynamic-convergence
	$(PY) scripts/audit_dynamic_study.py --require-complete
	$(PY) scripts/verify_dynamic_neural_gradient.py
	$(PY) scripts/evaluate_dynamic_study.py
dynamic-study: dynamic-confirmation dynamic-dt001
	$(PY) scripts/execute_dynamic_fallback.py
	$(PY) scripts/plot_dynamic_study.py
	$(PY) scripts/plot_dynamic_confirmation.py
	$(PY) scripts/audit_dynamic_jacobians.py
	$(PY) scripts/verify_dynamic_artifacts.py
	$(PY) scripts/write_reports.py

.PHONY: dynamic-benchmark
dynamic-benchmark:
	$(PY) scripts/benchmark_dynamic_study.py
	$(PY) scripts/write_reports.py

.PHONY: dynamic-confirmation
dynamic-confirmation: dynamic-evaluate
	$(PY) scripts/run_dynamic_confirmation.py
	$(PY) scripts/evaluate_dynamic_confirmation.py
	$(PY) scripts/plot_dynamic_confirmation.py

.PHONY: dynamic-reproduce
# Clean checkouts include recorded metrics but omit weights. Retrain missing weights;
# label the repeated benchmark explicitly and preserve published results in Git.
dynamic-reproduce: dynamic-train dynamic-convergence dynamic-dt001
	$(PY) scripts/evaluate_dynamic_study.py --repeat-benchmark
	$(PY) scripts/run_dynamic_confirmation.py
	$(PY) scripts/evaluate_dynamic_confirmation.py
	$(PY) scripts/execute_dynamic_fallback.py
	$(PY) scripts/audit_dynamic_jacobians.py
	$(PY) scripts/verify_dynamic_artifacts.py
	$(PY) scripts/benchmark_dynamic_study.py --repeat-benchmark
	$(PY) scripts/plot_dynamic_study.py
	$(PY) scripts/plot_dynamic_confirmation.py
	$(PY) scripts/write_reports.py

.PHONY: dynamic-dt001
dynamic-dt001: cardiac-trajectory
	env OPENBLAS_NUM_THREADS=2 $(PY) scripts/train_dynamic_dt001_pilot.py
	$(PY) scripts/plot_dynamic_dt001_pilot.py

.PHONY: dynamic-inputs
dynamic-inputs:
	mkdir -p data/processed experiments/checkpoints experiments/logs .cache
	@if [ ! -f data/processed/cardiac/cohort.npz ]; then $(FEM_RUN) scripts/generate_cardiac_cohort.py; fi
	$(PY) scripts/prepare_dynamic_hierarchy.py
dynamic-cohort: dynamic-inputs
dynamic-convergence: cardiac-trajectory dynamic-inputs
cardiac-trajectory: dynamic-inputs
