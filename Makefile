PY := .venv/bin/python
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
