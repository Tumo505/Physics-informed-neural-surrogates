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
